"""HC06 CPU-only changed-path and historical UID-trace verification; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from tools import exp086_dinov2s_v2 as base
    from training.adapters import dinov2_patch_mil as adapter
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base.install_gold_blocker(ROOT)
    protocol, ordered = base.validate_protocol(ROOT/base.PROTOCOL/'manifest.json', ROOT, True)
    assert not torch.cuda.is_initialized()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model = base.build_offline_model(base.ARCH, protocol['initial_asset'])
    initial_hash = base.state_hash(model.state_dict())
    assert initial_hash == protocol['arm']['initial_state_sha256']
    parameters = {k: id(v) for k, v in model.named_parameters()}
    keys = list(model.state_dict())
    rng = torch.get_rng_state().clone()
    head_method = model.head.__func__
    original_class_encode = type(model).encode
    adapter.install(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert parameters == {k: id(v) for k, v in model.named_parameters()}
    assert keys == list(model.state_dict()) and model.head.__func__ is head_method
    assert type(model).encode is original_class_encode
    assert base.state_hash(model.state_dict()) == initial_hash
    gen = torch.Generator().manual_seed(20261007)
    x = torch.randn(1, 2, 3, 224, 224, generator=gen)
    model.eval()
    with torch.no_grad():
        tokens = model.backbone.forward_features(x.flatten(0, 1))
        features = model.encode(x)
        assert features.shape == (1, 512, 384)
        assert torch.equal(features, tokens[:, 1:].reshape(1, 512, 384))
        assert torch.equal(model(x), model.head(features))
    model.backbone.set_grad_checkpointing(True)
    model.train()
    logits = model(x[:, :1])
    torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), torch.full((1, 12), .5)).backward()
    assert torch.isfinite(logits).all()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)
    assert model.backbone.pos_embed.grad.abs().sum() > 0
    assert base.state_hash(model.state_dict()) == initial_hash
    # New adapter has the same state keys as CLS. Verify explicit reconstruction,
    # since the legacy generic checkpoint loader would silently select CLS.
    from tools.patch_mil import load_patch_checkpoint
    model.eval()
    with torch.no_grad(): expected_logits = model(x[:, :1])
    with tempfile.TemporaryDirectory(prefix='hc06-checkpoint-') as temp:
        checkpoint = Path(temp)/'synthetic.pt'
        torch.save(dict(model=model.state_dict(),arch=base.ARCH,res=224,
                        aggregation_policy=adapter.POLICY),checkpoint)
        restored, metadata = load_patch_checkpoint(checkpoint)
        restored.eval()
        with torch.no_grad(): actual_logits = restored(x[:, :1])
        assert torch.equal(expected_logits,actual_logits)
        assert restored._hc06_aggregation_policy == adapter.POLICY
        del restored
    del model, x, tokens, features, logits
    # One lightweight worker preserves positive-worker persistent DataLoader RNG
    # behavior. UID ordering is checked against the actual stored historical trace.
    loader = torch.utils.data.DataLoader(list(range(len(ordered['train']))), batch_size=8,
        shuffle=True, drop_last=True, num_workers=1, persistent_workers=True,
        generator=torch.Generator().manual_seed(42))
    trace = hashlib.sha256(); epoch_hashes = []; batches = 0
    try:
        for epoch in range(8):
            eh = hashlib.sha256()
            for indexes in loader:
                uids = [ordered['train'][i] for i in indexes.tolist()]
                payload = json.dumps([epoch, uids], separators=(',', ':')).encode()
                trace.update(payload); eh.update(payload); batches += 1
            epoch_hashes.append(eh.hexdigest())
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
    historic = base.read(ROOT/'artifacts/experiments/EXP086_DINOV2S_G1_V2/trace.json')
    assert batches == 3448 and trace.hexdigest() == historic['training_trace_sha256']
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS', schema='hc06-cpu-probe-v1', seconds=time.monotonic()-started,
        initial_state_sha256=initial_hash, aggregation_policy=adapter.POLICY,
        install_preserves_parameters_keys_rng_and_head=True, class_method_unchanged=True,
        normalized_patch_features_exact=True, features_shape=[1, 512, 384],
        all_trainable_gradients_finite=True, canonical_position_gradient_nonzero=True,
        explicit_checkpoint_roundtrip_logits_exact=True,
        training_trace_sha256=trace.hexdigest(), historical_trace_exact=True,
        epoch_trace_sha256=epoch_hashes, batches=batches, cuda_initialized=False,
        cpu_threads=1, loader_workers=1, historical_loader_workers=12,
        scope='Synthetic wiring and actual recorded UID order; not historical tensor/dropout equality or GPU feasibility.',
        source_sha256={str(p.relative_to(ROOT)):base.sha256(p) for p in [Path(__file__), ROOT/'training/adapters/dinov2_patch_mil.py']})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base.write_new(args.out, result)
    print(json.dumps({k: result[k] for k in ['status','seconds','historical_trace_exact','cuda_initialized']}))


if __name__ == '__main__':
    main()
