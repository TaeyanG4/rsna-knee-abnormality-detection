"""VAST_PARITY_3090_V1 training shim v1.0.0.

Changelog: 1.0.0 - immutable EXP127 runner, saved-state gate, mapped paths,
separate dropout-off 32-update diagnostic. Imports never initialize CUDA.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import runpy
import sys


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _path(root, value):
    """Map only project-root Windows paths, and reject any path escaping stage."""
    value = str(value).replace('\\', '/')
    if PureWindowsPath(value).is_absolute():
        prefix = 'H:/kaggle/competitions/rsna-knee-abnormality-detection/'
        if not value.lower().startswith(prefix.lower()):
            raise ValueError('unregistered absolute path')
        value = value[len(prefix):]
    p = (root / value).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('path escapes stage')
    return p


def verify_initial(path, keys_path):
    """Load existing full state on CPU; verify every dtype/shape/value and combined hash."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Original saved full initial state required; RNG regeneration forbidden')
    import torch
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        elif isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
    expected = _read(keys_path)
    if not isinstance(state, dict) or set(state) != set(expected['keys']):
        raise ValueError('initial state tensor set mismatch')
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError('non-tensor initial state entry')
        value = value.detach().cpu().contiguous()
        raw = value.numpy().tobytes()
        rec = [hashlib.sha256(raw).hexdigest(), str(value.dtype).replace('torch.', ''), list(value.shape)]
        if rec != expected['keys'][key]:
            raise ValueError('initial state per-key mismatch: ' + key)
        h.update(key.encode()); h.update(raw)
    if h.hexdigest() != expected['combined_sha256']:
        raise ValueError('initial combined state mismatch')
    return state


def _check_protocol(stage, cfg):
    original = _read(_path(stage, cfg['protocol']))
    p = _read(_path(stage, cfg['normalized_protocol']))
    # Rewriting may change path spellings only. Every scalar numerical rule stays fixed.
    def normalized(x):
        if isinstance(x, dict):
            return {normalized(k): normalized(v) for k, v in x.items()}
        if isinstance(x, list):
            return [normalized(v) for v in x]
        if isinstance(x, str) and PureWindowsPath(x).is_absolute():
            return _path(stage, x).relative_to(stage).as_posix()
        return x
    if normalized(original) != p:
        raise ValueError('normalized manifest differs beyond project path spelling')
    rewrites = cfg.get('rewritten_bindings', {})
    bindings = dict(original['source_bindings'])
    bindings.update({'training/src/' + k: v for k, v in original['training_code_sha256'].items()})
    bindings[original['label_set']['path']] = original['label_set']['sha256']
    bindings[original['paths']['corpus'] + '/manifest.json'] = original['corpus_manifest_sha256']
    bindings[original['init_weights']['local']] = original['init_weights']['sha256']
    for field in ('ordered_ids', 'groups'):
        rec = original['folds']['0'][field]
        bindings[rec['path']] = rec['sha256']
    for name, digest in bindings.items():
        mapped = _path(stage, name)
        rewrite = rewrites.get(name)
        if rewrite:
            # Only target-only CSV derivatives can differ. Code and manifests stay byte exact.
            if not name.startswith('storage/labels/') or not name.endswith('.csv'):
                raise ValueError('non-label source binding rewrite forbidden')
            if rewrite['original_sha256'] != digest:
                raise ValueError('original label receipt mismatch')
            digest = rewrite['staged_sha256']
        if _sha(mapped) != digest:
            raise ValueError('source binding mismatch: ' + name)
    if p['variant'] != 'AUG1' or p['seed'] != 42 or p['common_args']['epochs'] != 20:
        raise ValueError('wrong training protocol')
    return p


def _diagnostic(stage, out, p, make_model, guard):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from raptor_common import seed_all, study_rng, select_centers
    from raptor_data import collate
    from raptor_exp127 import loader_factory
    from tools.parity_runtime import configure_parity, runtime_info
    out.mkdir(parents=True, exist_ok=False)
    runtime = configure_parity(amp='fp16', sdpa='efficient', strict=True, threads=1, seed=42)
    torch.use_deterministic_algorithms(True, warn_only=False)
    policy = runtime['policy']
    runtime = runtime_info()
    runtime['policy'] = dict(policy, deterministic_warn_only=False)
    runtime['deterministic_warn_only'] = torch.is_deterministic_algorithms_warn_only_enabled()
    from tools.vast_parity_3090 import validate_runtime
    validate_runtime(runtime)
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2, default=str))
    a, fold = p['common_args'], p['folds']['0']
    ids = _read(_path(stage, fold['ordered_ids']['path']))['train']
    labels, digest = guard.load_label_csv(_path(stage, p['paths']['labels']))
    if digest != p['label_set']['semantic_sha256']:
        raise ValueError('label semantic mismatch')
    seed_all(42)
    ds = loader_factory('AUG1')(_path(stage, p['paths']['corpus']), ids, labels,
        a['res'], a['k'], True, a['norm'], 42,
        window_policy=a['window_policy'], input_policy=a['input_policy'])
    ds.set_epoch(0)
    loader = DataLoader(ds, batch_size=a['bs'], shuffle=True, num_workers=0,
        drop_last=True, collate_fn=collate, generator=torch.Generator().manual_seed(42))
    model, _ = make_model(a['arch'], _path(stage, p['init_weights']['local']), p['init_weights']['sha256'])
    initial_combined, initial_keys = guard.state_hashes(model)
    (out / 'initial-state-receipt.json').write_text(json.dumps(dict(
        combined_sha256=initial_combined, keys=initial_keys,
        source='loaded saved full initial state; no RNG-generated tensor accepted'), indent=2))
    model = model.cuda().train()
    # Match the runner's eval-mode warmup before training.
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, 3, a['res'], a['res'], device='cuda'))
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.
        if hasattr(m, 'drop_prob'):
            m.drop_prob = 0.
    bb = [v for k, v in model.named_parameters() if k.startswith('backbone.')]
    head = [v for k, v in model.named_parameters() if not k.startswith('backbone.')]
    opt = torch.optim.AdamW([{'params': bb, 'lr': a['bb_lr']}, {'params': head, 'lr': a['head_lr']}], weight_decay=a['wd'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a['bb_lr'], a['head_lr']], total_steps=fold['steps'], pct_start=.15)
    scaler = torch.amp.GradScaler('cuda', init_scale=a['amp_init_scale'])
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(fold['pos_weight'], device='cuda'))
    records, losses, norms, scales = [], [], [], []
    opt.zero_grad(set_to_none=True)
    for i, (x, y, uids) in enumerate(loader, 1):
        studies = []
        for uid in uids:
            vol, mask = ds.corpus.volume(uid), ds.corpus.mask(uid)
            if vol.dtype != np.uint8 or mask.dtype.kind not in 'bui':
                raise ValueError('diagnostic integer-stage dtype changed')
            centers = np.asarray(select_centers(mask, ds.corpus.depth, a['k'], ds.corpus.spec,
                a['window_policy'], study_rng(42, 0, uid)), dtype=np.int64)
            studies.append(dict(uid=uid, volume_sha256=hashlib.sha256(vol.tobytes()).hexdigest(),
                mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                centers_sha256=hashlib.sha256(centers.tobytes()).hexdigest()))
        records.append(dict(step=i, uids_sha256=hashlib.sha256('\n'.join(uids).encode()).hexdigest(),
            x_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(), y_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest(),
            integer_studies=studies))
        with torch.autocast('cuda', dtype=torch.float16):
            loss = lossf(model(x.cuda()).float(), y.cuda())
        if not torch.isfinite(loss):
            raise ValueError('nonfinite diagnostic loss')
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
        scale = scaler.get_scale(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if scaler.get_scale() >= scale:
            sched.step()
        losses.append(float(loss.item())); norms.append(float(norm.item())); scales.append(float(scale))
        if i == 32:
            break
    if len(losses) != 32:
        raise ValueError('insufficient diagnostic batches')
    np.savez(out / 'train32.npz', loss=losses, grad_norm=norms, scale=scales)
    (out / 'inputs.json').write_text(json.dumps(records, indent=2))
    del model, opt, scaler
    torch.cuda.empty_cache()


def run_training(stage: Path, out: Path, diagnostic_only=False):
    stage, out = Path(stage).resolve(), Path(out).resolve()
    cfg = _read(stage / 'vast-config.json')
    if not cfg.get('initial_state'):
        raise FileNotFoundError('Part 2 BLOCKED: exact saved original full initial state missing; no CUDA started')
    from tools.vast_parity_3090 import pins_ok, write
    pins_ok()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    p = _check_protocol(stage, cfg)
    state = verify_initial(_path(stage, cfg['initial_state']), _path(stage, p['initial_state']['keys_file']))
    if _read(_path(stage, p['initial_state']['keys_file']))['combined_sha256'] != p['initial_state_sha256']:
        raise ValueError('manifest initial state mismatch')
    sys.path[:0] = [str(stage), str(stage / 'training/src')]
    from tools import exp127_radr50_single as guard
    import raptor_radimagenet
    original_builder = raptor_radimagenet.build_radimagenet_model
    original_validator = guard.validate_arguments
    def builder(*args, **kwargs):
        model, report = original_builder(*args, **kwargs)
        model.load_state_dict(state, strict=True)
        if guard.state_hashes(model)[0] != p['initial_state_sha256']:
            raise ValueError('loaded model initial state mismatch')
        return model, report
    def validate(args, root):
        if Path(root).resolve() != stage or args.variant != 'AUG1' or args.seed != 42 or args.fold != 0 or args.mode != 'full' or args.validate_only:
            raise ValueError('wrong shim scope')
        for key, expected in p['common_args'].items():
            if getattr(args, key) != expected:
                raise ValueError('frozen training argument changed: ' + key)
        if args.tag != p['folds']['0']['tag'] or not 0 <= args.workers <= 12:
            raise ValueError('wrong tag/workers')
        for key in ('labels', 'eval_labels', 'corpus'):
            if Path(getattr(args, key)).resolve() != _path(stage, p['paths'][key]):
                raise ValueError('training path changed')
        if Path(args.init_weights).resolve() != _path(stage, p['init_weights']['local']):
            raise ValueError('initial backbone path changed')
        if Path(args.protocol).resolve() != _path(stage, cfg['normalized_protocol']) or Path(args.output_root).resolve() != out:
            raise ValueError('protocol/output path changed')
        return p, p['folds']['0'], _read(_path(stage, p['folds']['0']['ordered_ids']['path'])), out
    argv = [str(stage / 'training/runners/exp127_radr50_single.py')]
    args = dict(p['common_args'], variant='AUG1', seed=42, fold=0, mode='full', tag=p['folds']['0']['tag'],
        protocol=str(_path(stage, cfg['normalized_protocol'])), output_root=str(out), workers=cfg.get('workers', 12),
        init_weights=str(_path(stage, p['init_weights']['local'])))
    args.update({k: str(_path(stage, p['paths'][k])) for k in ('labels', 'eval_labels', 'corpus')})
    for key, value in args.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    old_argv, old_relax = sys.argv, os.environ.get('KNEE_REMOTE_RELAX_ENV')
    try:
        raptor_radimagenet.build_radimagenet_model = builder
        guard.validate_arguments = validate
        _diagnostic(stage, out / 'diagnostic32', p, builder, guard)
        write(out / 'diagnostic32/binding.json', {
            'inventory_sha256': _sha(stage / 'inventory.json'),
            'initial_state_file_sha256': _sha(_path(stage, cfg['initial_state'])),
            'normalized_protocol_sha256': _sha(_path(stage, cfg['normalized_protocol']))})
        if not diagnostic_only:
            local_diag = stage / 'reference-diagnostic' / 'diagnostic32' / 'inputs.json'
            left, right = _read(local_diag), _read(out / 'diagnostic32' / 'inputs.json')
            if len(left) != 32 or len(right) != 32:
                raise ValueError('missing 32-step local/remote diagnostic inputs')
            for a_step, b_step in zip(left, right):
                for key in ('step', 'uids_sha256', 'y_sha256', 'integer_studies'):
                    if key not in a_step or a_step[key] != b_step.get(key):
                        raise ValueError('diagnostic integer/order/label mismatch; refusing full fit')
            # Diagnostics intentionally force efficient SDPA; restore the actual
            # local EXP127 recipe, not whatever the caller process inherited.
            import torch
            baseline = _read(stage / 'artifacts/experiments/EXP127_AUG1_S42_F0/provenance.json')
            backend = baseline['runtime']['backends']
            for flag, setter in (('flash_sdp_enabled', torch.backends.cuda.enable_flash_sdp),
                                 ('mem_efficient_sdp_enabled', torch.backends.cuda.enable_mem_efficient_sdp),
                                 ('math_sdp_enabled', torch.backends.cuda.enable_math_sdp)):
                setter(backend[flag])
            # All model/optimizer/BN state is newly loaded by the runner; it resets
            # random seeds and runtime before construction. Diagnostic peak memory
            # must not contaminate its own memory receipt either.
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            os.environ['KNEE_REMOTE_RELAX_ENV'] = '1'
            sys.argv = argv
            runpy.run_path(argv[0], run_name='__main__')
    finally:
        raptor_radimagenet.build_radimagenet_model = original_builder
        guard.validate_arguments = original_validator
        sys.argv = old_argv
        if old_relax is None:
            os.environ.pop('KNEE_REMOTE_RELAX_ENV', None)
        else:
            os.environ['KNEE_REMOTE_RELAX_ENV'] = old_relax
