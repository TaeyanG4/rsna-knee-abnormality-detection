"""H216 v1.0.1 actual-OAI CPU input mechanics; immutable v1 remains intact.

No fit, decode, render, checkpoint load, CUDA or running-output access.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.l1_rco_stage0_v1 import need, read, write, sha

VERSION = '1.0.1'
OUT = ROOT / '.local/scratch/slot-b/h216-oai-pair-core-v101'
PILOT = ROOT / '.local/scratch/slot-b/h216-oai-v4pilot8-v1'
INPUT = ROOT / '.local/scratch/slot-b/h216-oai-pair-inputs-v1'


def exact_windows(ct, blocks, px, meta, logical_index, epoch=0, seed=22601):
    """Reference original Studies semantics without the sparse adapter's offsets."""
    import numpy as np
    rng = np.random.default_rng(seed + 1000003 * epoch + logical_index)
    bs = [dict(b, views=[(0, v, off) for v, off in b['views']]) for b in blocks]
    rows = ct.window_rows(bs, lambda f, r: meta[r], 'train', 3, 3, 3, 3.5,
                          rng, .25, 'cover')
    if len(rows) > 64:
        rows = [rows[j] for j in np.sort(rng.choice(len(rows), 64, replace=False))]
    rows = np.asarray(rows, np.int64)
    need(len(rows) > 0 and rows.shape[1] == 5, 'REFERENCE_WINDOWS')
    x = np.stack([np.asarray(px[rows[:, 2 + d]]) for d in range(3)], 1)
    m = np.asarray(meta[rows[:, 1]][:, [1, 3, 4, 5, 6, 7]], np.float32)
    return x, m


def compare_item(item, expected, logical_index):
    import numpy as np
    x, m, i = item
    need(i == logical_index, 'LOGICAL_INDEX')
    need(np.array_equal(x.numpy(), expected[0]) and
         np.array_equal(m.numpy(), expected[1]), 'ACTUAL_WINDOW_PARITY')
    need(len(x) >= 2, 'CPU_WINDOW_COUNT')


def prepare():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_ISOLATION')
    need(not OUT.exists(), 'OUTPUT_EXISTS')
    from tools.job_runtime import LOGS, job_status
    need(job_status(LOGS, 'H216_OAI_V4_PILOT8_V1')['state'] == 'SUCCESS', 'PILOT_NOT_TERMINAL')
    from tools.h216_oai_pair_runtime_v101 import inputs
    from tools.h216_oai_pair_inputs_v1 import ScheduledStudies
    from tools import h216_oai_pair_core_v1 as core
    import numpy as np
    import torch
    torch.set_num_threads(1)
    need(not torch.cuda.is_initialized(), 'CUDA_INITIALIZED')
    def forbidden(*args, **kwargs):
        raise ValueError('CHECKPOINT_LOAD_FORBIDDEN')
    old_load = torch.load
    torch.load = forbidden
    started = time.perf_counter()
    try:
        binding = inputs()
        result = read(PILOT / 'RESULT.json'); review = read(PILOT / 'B-RESULT-CHECK.json')
        need(result['status'] == 'V4_PILOT_PREPARED' and result['parity'] == 'PASS' and
             review['verdict'] == 'PASS_B_DIRECT' and review['result_sha256'] == sha(PILOT / 'RESULT.json'), 'PILOT_CHECK')
        selected = binding['pilot_reusable_ids']
        need(len(selected) == len(set(selected)) == 3, 'FIXED_PILOT_INTERSECTION')
        for uid in selected:
            token = hashlib.sha256(uid.encode()).hexdigest()[:24]
            for name in ('receipt.json', 'px.npy', 'meta.npy'):
                rel = f'shards/{token}/{name}'
                need(sha(PILOT / rel) == review['reusable_files'][rel], 'PILOT_FILE_DRIFT')
        # Only committed pilot payloads, never the running bulk destination.
        for name, h in result['hashes'].items():
            need(sha(PILOT / name) == h, 'PILOT_PAYLOAD_DRIFT')
        os.environ['HF_HUB_OFFLINE'] = '1'
        sys.path.insert(0, str(ROOT / 'tools/exp189_s6_src'))
        import compact_train as ct
        ds = ScheduledStudies(ct, PILOT, binding['oai_logical_roster'], selected)
        index = read(PILOT / 'index.json')
        px = np.load(PILOT / 'px.u8.npy', mmap_mode='r')
        meta = np.load(PILOT / 'meta.f32.npy', mmap_mode='r')
        items = []; windows = 0; payload = hashlib.sha256()
        for uid in selected:
            logical = binding['oai_logical_roster'].index(uid)
            item = ds[logical]
            expected = exact_windows(ct, index['blocks'][uid], px, meta, logical)
            compare_item(item, expected, logical)
            windows += len(item[0])
            payload.update(item[0].numpy().tobytes()); payload.update(item[1].numpy().tobytes())
            items.append((item[0][:2], item[1][:2], logical))
        img, m, mask, idx = ct.collate(items)
        img = core.crop(img, torch.Generator().manual_seed(21673))
        target = torch.from_numpy(np.load(INPUT / 'native-targets.private.npy'))[idx]
        need(target.shape == (3, 30) and bool((target >= 0).any(1).all()), 'TARGET_ALIGNMENT')
        torch.manual_seed(21600)
        base = ct.Net('convnext_small', None, crop=352, in_chans=3, spatial=True, extent=False)
        model = core.TemporaryAdapter(base, 120)
        core.configure_trainable(model, True); core.train_mode(model)
        z = model(img, m, mask)
        loss = core.native_loss(z, target)
        need(tuple(z.shape) == (3, 120) and bool(torch.isfinite(z).all()) and
             bool(torch.isfinite(loss)), 'FORWARD_FINITE')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        need(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), 'GRADIENTS')
        need(all(p.grad is None for n, p in model.enc.named_parameters() if not n.startswith('stages.3.')), 'FROZEN_GRADIENT')
        need(not torch.cuda.is_initialized(), 'CUDA_INITIALIZED')
        paths = [Path(__file__), ROOT / 'tests/test_h216_oai_pair_prepare_v101.py',
                 ROOT / 'docs/research/h216-oai-pair-prepare-v101.md',
                 ROOT / 'tools/h216_oai_pair_inputs_v1.py', ROOT / 'tools/h216_oai_pair_core_v1.py',
                 ROOT / 'tools/exp189_s6_src/compact_train.py']
        record = dict(version=VERSION, status='PASS_ACTUAL_OAI_CPU_INPUT_MECHANICS',
            source_hashes={str(p): sha(p) for p in paths}, input_contract=binding['contract_id'],
            pilot_result_sha256=sha(PILOT / 'RESULT.json'), studies=3,
            full_training_windows_compared=windows, graph_windows=6,
            window_tensor_sha256=payload.hexdigest(), exact_window_parity=True,
            native_output_shape=[3, 120], gradients_finite=True, early_encoder_frozen=True,
            gradient_tensors=len(gradients), optimizer_steps=0, random_weights=True,
            checkpoint_loads=0, cuda=False, clinical_reading=False, efficacy_evaluated=False,
            full800_qualified=False, training_ready=False, seconds=time.perf_counter()-started)
        OUT.mkdir(); write(OUT / 'RESULT.json', record)
        return {k: record[k] for k in ('status', 'studies', 'full_training_windows_compared',
                 'graph_windows', 'optimizer_steps', 'checkpoint_loads', 'cuda', 'training_ready')}
    finally:
        torch.load = old_load


if __name__ == '__main__':
    import json
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    try:
        print(json.dumps(prepare()))
    except Exception as exc:
        code = str(exc) if isinstance(exc, ValueError) and str(exc).replace('_', '').isalnum() else 'PRIVATE_DETAILS_SUPPRESSED'
        print(json.dumps(dict(status='FAILED',error_type=type(exc).__name__,code=code)))
        raise SystemExit(1)
