"""H-B-112 v1.0.0: isolated, no-gradient, matched BN replay; local data only.

Changelog: exact retained-SWA recovery, immutable registration, complete model
state guards, independent-review admission and fixed descriptive contrasts.
No original training source, checkpoint, label, exposure or recipe is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from torch.utils.data import DataLoader
from training.arms import coatnet as c, exp190f as e, lifecycle as life
from training.arms.exp190f_data import Dataset, mixed
from training.arms.coatnet_fit import observation, verify_studies
from tools.exp112_full_swa import (batchnorm_modules, recompute_bn, state_dict_sha256,
                                  swa_average, uid_hash as bn_uid_hash)
from tools import multi_lens_readout as ml

VERSION = '1.0.0'
SCRATCH = ROOT / '.local/scratch/sol-4/bn-only'
REG = ROOT / 'artifacts/experiments/HB112_BN_ONLY_V1/registration.json'
ORIGINAL = '.local/exp190f-cpu-v1'
OUTPUTS = ORIGINAL + '/collected-full/outputs'
EPOCHS = ORIGINAL + '/train/epochs'
SPEC_SHA = '9094da258f0d4f230a36372394c9778a20ab7c59b2d51fa1e58a025d00444550'
PRE_SHA = 'fa7b0b67fc5d13fc73c573e61ee3b76e7c5390fc7b36b4df02c4edad2710826c'
FINAL_SHA = '924e9fad150bde7366d729c1577ecedde9fc0722fcd2715f846f3c533d236849'
SWA = [(14, '82bbf76b6cb517a1cb78654e5facf37d5ca896365eb3ed8d86bc765a5ffb8d18'),
       (11, 'd044578bb24f325fd796669e14c813a54b97657b39a6b0c5e557704b2dfa2770'),
       (12, '5f8541176aa6360c6abbf9d739057cb5bba559e2175267a7e0246850118f802e')]
VIEWS = ('plain', 'mirror', 'mean')
POLICY = dict(seed=43112, study_seed=43, draw_epoch=16, workers=0, threads=2,
              batch_size=2, shuffle=False, drop_last=False, augment=False,
              mode='train', bn_precision='bf16', eval_precision='fp16',
              bn_resolution=384, bn_k=12, eval_resolution=320, eval_k=94,
              eval_chunk=16, gradients=False, checkpoint_selection=False,
              order='original RSNA prefix, then original OAI order; RSNA keeps prefix',
              backend='deterministic algorithms; cudnn benchmark off; cudnn TF32 on; matmul TF32 off')
CONTRASTS = [('rsna-minus-mixed', 'rsna', 'mixed', False),
             ('mixed-minus-CONTROL43', 'mixed', 'control', True),
             ('rsna-minus-CONTROL43', 'rsna', 'control', True),
             ('matched-mixed-minus-original-EXP190F', 'mixed', 'original', False)]
need = life._need
sha = life.file_sha256


def setup():
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 2:
        torch.set_num_interop_threads(2)
    import cv2
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')


def read(path):
    return life.parse_json(Path(path).read_bytes())


def put(path, value, *, scratch=False):
    path = Path(path)
    raw = life.canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if scratch:
        need(path.resolve().is_relative_to(SCRATCH), 'scratch outside fixed lane')
        path.write_bytes(raw)
    elif path.exists():
        need(path.read_bytes() == raw, 'immutable artifact already exists with different bytes')
    else:
        life.create_new(path, raw)


def tensor_hash(items):
    """Framed tensor bytes, including dtype/shape; supports nonpersistent buffers."""
    h = hashlib.sha256()
    for name, value in sorted(items):
        t = value.detach().cpu().contiguous()
        h.update(life.canonical_bytes([name, str(t.dtype), list(t.shape)]))
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def bn_keys(model):
    return {name + '.' + key for name, _ in batchnorm_modules(model)
            for key in ('running_mean', 'running_var', 'num_batches_tracked')}


def non_bn(model):
    allowed = bn_keys(model)
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    # Include ordinary tensor attributes too. Module metadata is checked below.
    for name, module in model.named_modules():
        for key, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                tensors[name + '.attribute.' + key] = value
    return tensor_hash((k, v) for k, v in tensors.items() if k not in allowed)


def module_metadata(model):
    def simple(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, (tuple, list)):
            return [simple(x) for x in v]
        return str(type(v))
    return life.content_sha256({name: dict(type=type(m).__module__ + '.' + type(m).__qualname__,
        extra=m.extra_repr(), attributes={k: simple(v) for k, v in vars(m).items()
            if not k.startswith('_') and not isinstance(v, torch.Tensor)})
        for name, m in model.named_modules()})


def recover():
    states = []
    for epoch, expected in SWA:
        path = ROOT / EPOCHS / f'swa-{epoch:02d}.pt'
        need(sha(path) == expected, 'retained SWA member changed')
        ck = torch.load(path, map_location='cpu', weights_only=True)
        need(ck['epoch'] == epoch, 'SWA epoch differs')
        states.append(ck['model'])
    state = swa_average(states)
    need(state_dict_sha256(state) == PRE_SHA, 'exact pre-BN recovery failed')
    return state


def inspect_state(state):
    model = c.build_model().eval()
    model.load_state_dict(state, strict=True)
    final = torch.load(ROOT / OUTPUTS / 'model_best.pt', map_location='cpu', weights_only=True)['model']
    different = [k for k in state if not torch.equal(state[k], final[k])]
    allowed = bn_keys(model)
    need(len(batchnorm_modules(model)) == 25 and len(different) == 75
         and set(different) == allowed, 'original changed non-BN state or BN audit differs')
    receipt = read(ROOT / EPOCHS / 'epoch-16.receipt.json')
    epoch_path = ROOT / EPOCHS / 'epoch-16.pt'
    need(sha(epoch_path) == receipt['sha256'], 'epoch RNG checkpoint differs')
    ck = torch.load(epoch_path, map_location='cpu', weights_only=True)
    need(ck['epoch'] == 16 and ck['identity'] == receipt['identity'], 'epoch identity differs')
    need(receipt['identity']['spec_sha256'] == SPEC_SHA, 'epoch spec differs')
    need(receipt['swa_members'] == {f'swa-{n:02d}.pt': h for n, h in SWA}, 'SWA receipt differs')
    result = dict(pre_bn_sha256=PRE_SHA, non_bn_sha256=non_bn(model),
        bn_modules=25, changed_bn_keys=len(different), non_bn_differences=0,
        module_metadata_sha256=module_metadata(model),
        nonpersistent_buffers=[k for k, _ in model.named_buffers() if k not in state],
        stochastic=[dict(name=n, p=m.p) for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Dropout) and m.p > 0],
        retained_rng=dict(cpu=tensor_hash([('cpu', ck['torch_rng'])]),
                          cuda=tensor_hash([(str(i), x) for i, x in enumerate(ck['cuda_rng'])])),
        original_reproduction='UNPROVED: epoch-boundary RNG retained; original workers12 and nondeterministic backend. New shared seed policy; no historical replay claim.')
    modules = {inspect.getsourcefile(type(m)) for m in model.modules()}
    modules.update([inspect.getsourcefile(torch.optim.swa_utils.update_bn),
                    inspect.getsourcefile(DataLoader)])
    return result, {str(Path(f).resolve()): sha(f) for f in modules if f}


def ruler_bindings(ids):
    # Numeric labels/masks are bound, never scored, selected or printed here.
    ctx = ml.make_context(ids, ROOT)
    arrays = {r: dict(labels=tensor_hash([('labels', torch.as_tensor(y))]),
                      mask=tensor_hash([('mask', torch.as_tensor(mask))]), min_class=minimum)
              for r, (y, mask, minimum) in ctx.refs.items()}
    return dict(files=ctx.bindings, arrays=arrays)


def make_registration():
    need(not torch.cuda.is_initialized(), 'CPU registration initialized CUDA')
    spec_path = ROOT / ORIGINAL / 'spec.json'
    need(sha(spec_path) == SPEC_SHA, 'original spec changed')
    spec = e.Spec(spec_path.read_bytes())
    p = e.validate_inputs(spec, ROOT)
    receipt = read(ROOT / ORIGINAL / 'train-receipt.json')
    need(receipt['spec_sha256'] == SPEC_SHA and receipt['exit_code'] == 0, 'original receipt differs')
    need(receipt['outputs']['model_best.pt'] == FINAL_SHA, 'original final binding differs')
    files = dict(spec.inventory)
    files.update({OUTPUTS + '/' + k: v for k, v in receipt['outputs'].items()})
    files.update({EPOCHS + f'/swa-{n:02d}.pt': h for n, h in SWA})
    files.update(read(ROOT / e.PRESET)['control_predictions'])
    for path in (ORIGINAL+'/spec.json', ORIGINAL+'/train-receipt.json',
                 EPOCHS+'/epoch-16.receipt.json', EPOCHS+'/epoch-16.pt'):
        files[path] = sha(ROOT / path)
    life._verify_files(ROOT, files)
    audit, package_sources = inspect_state(recover())
    rulers = ruler_bindings(p['val_ids'])
    files.update(rulers['files'])
    # Hash only imported project modules; do not traverse the workspace.
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f:
            path = Path(f).resolve()
            if path.suffix == '.py' and any(path.is_relative_to(ROOT / d) for d in ('training', 'tools')):
                files[path.relative_to(ROOT).as_posix()] = sha(path)
    files['tools/hb112_bn_only_v1.py'] = sha(__file__)
    files['tests/test_hb112_bn_only_v1.py'] = sha(ROOT/'tests/test_hb112_bn_only_v1.py')
    need(len(p['train_ids']) == 4959 and len(p['rsna_train_ids']) == 3455
         and len(p['val_ids']) == 888 and p['train_ids'] == p['rsna_train_ids'] + p['oai_train_ids'],
         'frozen cohort order differs')
    return dict(schema='hb112-bn-only/v1', version=VERSION, builder='Codex sol-4',
        original_spec_sha256=SPEC_SHA, inputs=files, package_sources=package_sources,
        cohorts={k: p[k] for k in ('train_ids', 'rsna_train_ids', 'oai_train_ids', 'val_ids')},
        ordered_uid_sha256={k: e.uid_hash(p[k]) for k in ('train_ids', 'rsna_train_ids', 'oai_train_ids', 'val_ids')},
        policy=POLICY, audit=audit, rulers=rulers, targets=list(e.control.LAB),
        contrasts=[list(x) for x in CONTRASTS], target=spec.document['target_environment'],
        interpretation='Conditional BN-cohort contrast at fixed EXP190F weights; no universal OAI inference, holdout-stat fitting, routing or checkpoint selection.',
        original_bn_seconds=read(ROOT / OUTPUTS / 'summary.json')['bn']['seconds'])


def validate(reg):
    need(reg['schema'] == 'hb112-bn-only/v1' and reg['version'] == VERSION
         and reg['policy'] == POLICY and reg['contrasts'] == [list(x) for x in CONTRASTS], 'registration policy differs')
    life._verify_files(ROOT, reg['inputs'])
    for path, digest in reg['package_sources'].items():
        need(sha(path) == digest, 'installed model/helper source changed')
    spec = e.Spec((ROOT / ORIGINAL / 'spec.json').read_bytes())
    need(sha(ROOT / ORIGINAL / 'spec.json') == SPEC_SHA, 'original spec changed')
    need(all(reg['inputs'].get(k) == v for k, v in spec.inventory.items()), 'original bindings omitted/changed')
    need(reg['target'] == spec.document['target_environment'], 'target changed')
    p = e.population(spec, ROOT)
    for k, ids in reg['cohorts'].items():
        need(ids == p[k] and e.uid_hash(ids) == reg['ordered_uid_sha256'][k], 'cohort or order changed')
    need(ruler_bindings(p['val_ids']) == reg['rulers'], 'ruler labels/masks changed')
    need(reg['audit']['pre_bn_sha256'] == PRE_SHA and reg['targets'] == list(e.control.LAB), 'state/targets differ')
    return p


def prepare(reg):
    validate(reg)
    path = SCRATCH / 'prepared/pre-bn.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=True)['model']
        need(state_dict_sha256(state) == PRE_SHA, 'prepared state changed')
    else:
        state = recover()
        with path.open('xb') as stream:
            torch.save(dict(model=state, scope='HB112_PRE_BN_FIXED_SWA'), stream)
    receipt = dict(registration_sha256=life.content_sha256(reg), checkpoint_sha256=sha(path),
                   pre_bn_sha256=PRE_SHA, status='PREPARED_CPU_NO_GPU')
    put(SCRATCH / 'prepared/receipt.json', receipt)
    return receipt


def admit(reg, review, evidence):
    expected = life.content_sha256(reg)
    blockers = []
    if review.get('verdict') != 'PASS': blockers.append('Independent reviewer PASS is required before ANY CUDA execution')
    if not review.get('reviewer') or review.get('reviewer') == reg['builder']:
        blockers.append('Reviewer identity must differ from builder')
    if review.get('registration_sha256') != expected: blockers.append('Review must bind this registration')
    if review.get('evidence') != evidence: blockers.append('Review must bind CPU evidence hashes')
    if review.get('approve_new_matched_policy') is not True: blockers.append('Matched policy needs explicit review')
    if review.get('b_gpu_authorized') is not True: blockers.append('B must authorize the idle GPU window separately')
    return dict(status='BLOCKED' if blockers else 'REVIEWED_B_EXECUTION_ONLY', blockers=blockers,
                launch_authorized=not blockers, registration_sha256=expected)


class VerifiedDataset(Dataset):
    def __getitem__(self, index):
        uid = self.ids[index]
        if not hasattr(self, '_verified_corpora'):
            self._verified_corpora = (c.base.Corpus(self.rsna.path), c.base.Corpus(self.oai.path))
        corpus = self._verified_corpora[0 if uid in self.rsna_ids else 1]
        verify_studies(corpus, [uid])
        return super().__getitem__(index)


def replay(model, loader, device):
    model.eval().requires_grad_(False)
    need(all(p.grad is None for p in model.parameters()), 'preexisting gradient')
    before, metadata = non_bn(model), module_metadata(model)
    c.base._cpu_seed(POLICY['seed'])
    if device == 'cuda': torch.cuda.manual_seed_all(POLICY['seed'])
    rng_before = tensor_hash([('cpu', torch.get_rng_state())])
    gpu_before = tensor_hash([('cuda', torch.cuda.get_rng_state())]) if device == 'cuda' else None
    with torch.no_grad():
        bn = recompute_bn(model, mixed.bn_batches(loader), device,
                          c.base.amp_context(device, 'bf16' if device == 'cuda' else 'none'))
    need(non_bn(model) == before, 'non-BN tensor changed (including nonpersistent buffers)')
    need(module_metadata(model) == metadata, 'module configuration/mode changed')
    need(all(p.grad is None and not p.requires_grad for p in model.parameters()), 'gradient/parameter flags changed')
    return dict(bn=bn, non_bn_before=before, non_bn_after=non_bn(model),
                cpu_rng_before=rng_before, cuda_rng_before=gpu_before,
                cpu_rng_after=tensor_hash([('cpu', torch.get_rng_state())]))


def selection(p, ids):
    need(len(set(ids)) == len(ids) and set(ids) <= set(p['train_ids'])
         and not set(ids) & set(p['val_ids']), 'invalid BN training selection')
    chosen = dict(p)
    chosen.update(train_ids=ids, rsna_train_ids=[u for u in ids if u in set(p['rsna_train_ids'])],
                  oai_train_ids=[u for u in ids if u in set(p['oai_train_ids'])],
                  train_labels={u: p['train_labels'][u] for u in ids},
                  train_masks={u: p['train_masks'][u] for u in ids})
    return chosen


def predict(state, corpus, ids, device):
    model = c.build_model(res=320).to(device).eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    need(state_dict_sha256(model.state_dict()) == state_dict_sha256(state), 'inference reload differs')
    before = non_bn(model)
    values = []
    for uid in ids:
        verify_studies(corpus, [uid])
        values.append(c.infer_pair(model, c.Config(), *corpus.get(uid), device=device,
                                  precision='fp16' if device == 'cuda' else 'none',
                                  chunk=16 if device == 'cuda' else 2))
    need(non_bn(model) == before, 'inference mutated non-BN state')
    plain, mirror = (np.asarray(v) for v in zip(*values))
    return dict(plain=plain, mirror=mirror, mean=(plain+mirror)/2)


def compare(ctx, predictions):
    out = {}
    for view in VIEWS:
        for name, arm, control, label_change in CONTRASTS:
            a, b = predictions[arm][view], predictions[control][view]
            out[name+'/'+view] = dict(
                descriptive_only=True, no_selection_or_routing=True,
                max_abs_probability_delta=float(np.max(np.abs(a-b))),
                mean_abs_probability_delta=float(np.mean(np.abs(a-b))),
                multi_lens=ml.evaluate(ctx, {'43': a}, {'43': b}, label_change=label_change))
    return out


def execute(reg, *, smoke, review=None):
    p = validate(reg)
    device = 'cpu' if smoke else 'cuda'
    if smoke:
        need(not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
        seen = observation('cpu')
        state = recover()
    else:
        decision = admit(reg, review or {}, evidence_files())
        need(decision['launch_authorized'], '; '.join(decision['blockers']))
        pr = read(SCRATCH / 'prepared/receipt.json')
        need(pr['registration_sha256'] == life.content_sha256(reg)
             and sha(SCRATCH/'prepared/pre-bn.pt') == pr['checkpoint_sha256'], 'prepared receipt changed')
        state = torch.load(SCRATCH/'prepared/pre-bn.pt', map_location='cpu', weights_only=True)['model']
        # CUDA is first touched ONLY after the independent/B admission checks.
        seen = observation('cuda')
        life.check_environment(reg['target'], hostname=seen['hostname'], runtime=seen['runtime'])
        need(torch.cuda.is_bf16_supported(), 'native bf16 required')
    need(state_dict_sha256(state) == PRE_SHA, 'initial state differs')
    out = SCRATCH / ('smoke' if smoke else 'production-v1')
    out.mkdir(parents=True, exist_ok=smoke)
    ids = p['rsna_train_ids'][:1] if smoke else p['val_ids']
    corpus = c.base.Corpus(ROOT/c.CORPUS)
    counts, predictions = {}, {}
    started = time.monotonic()
    for cohort in ('mixed', 'rsna'):
        train = (p['rsna_train_ids'][:1] + (p['oai_train_ids'][:1] if cohort == 'mixed' else [])) if smoke else (
            p['train_ids'] if cohort == 'mixed' else p['rsna_train_ids'])
        need(not set(train) & set(p['val_ids']), 'holdout in BN cohort')
        ds = VerifiedDataset(ROOT/c.CORPUS, ROOT/e.OAI, selection(p, train), 43, augment=False)
        ds.set_epoch(16)
        generator = torch.Generator().manual_seed(POLICY['seed'])
        loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=False, num_workers=0,
                            collate_fn=mixed.collate, generator=generator, pin_memory=device=='cuda')
        model = c.build_model().to(device).eval()
        model.load_state_dict(state, strict=True)
        need(non_bn(model) == reg['audit']['non_bn_sha256'], 'initial non-BN bytes differ')
        result = replay(model, loader, device)
        need(result['bn']['studies'] == result['bn']['unique_studies'] == len(train)
             and result['bn']['studies_sha256'] == bn_uid_hash(sorted(train)), 'replay roster differs')
        result['ordered_uid_sha256'] = e.uid_hash(train)
        final = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        path = out / f'{cohort}-final.pt'
        with path.open('wb' if smoke else 'xb') as stream:
            torch.save(dict(model=final, registration_sha256=life.content_sha256(reg), cohort=cohort,
                            scope='CPU_ENGINEERING_ONLY' if smoke else 'MATCHED_BN_ONLY'), stream)
        del model
        reloaded = torch.load(path, map_location='cpu', weights_only=True)['model']
        need(state_dict_sha256(reloaded) == state_dict_sha256(final), 'saved state differs')
        t0 = time.monotonic()
        predictions[cohort] = predict(reloaded, corpus, ids, device)
        result.update(checkpoint_sha256=sha(path), checkpoint_reloaded=True,
                      readout_seconds=time.monotonic()-t0, readout_rows=len(ids))
        for view in VIEWS:
            fp = out / f'{cohort}-{view}.csv'
            raw = life.prediction_csv_bytes(ids, predictions[cohort][view], reg['targets'])
            if smoke: fp.write_bytes(raw)
            else: life.create_new(fp, raw)
            predictions[cohort][view] = life.load_prediction_csv(fp, ids, reg['targets'])
        counts[cohort] = result
    need(counts['mixed']['cpu_rng_before'] == counts['rsna']['cpu_rng_before']
         and counts['mixed']['cuda_rng_before'] == counts['rsna']['cuda_rng_before'], 'initial replay RNG differs')
    sensitivity = None
    if smoke:
        # Training image only: exercises baseline sensitivity without holdout stats.
        old = torch.load(ROOT/OUTPUTS/'model_best.pt', map_location='cpu', weights_only=True)['model']
        original = predict(old, corpus, ids, 'cpu')
        sensitivity = {v: float(np.max(np.abs(predictions['mixed'][v]-original[v]))) for v in VIEWS}
        need(not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
    receipt = dict(schema='hb112-bn-result/v1', registration_sha256=life.content_sha256(reg),
        scope='CPU_ENGINEERING_ONLY' if smoke else 'MATCHED_BN_ONLY', status='COMPLETE',
        cohorts=counts, seconds=time.monotonic()-started, optimizer_steps=0,
        gpu_seconds=0 if smoke else None, cuda_initialized=torch.cuda.is_initialized(),
        original_sensitivity_training_only=sensitivity,
        observation=seen,
        files={f.name: sha(f) for f in out.iterdir() if f.name.endswith(('.pt', '.csv'))})
    put(out/'receipt.json', receipt, scratch=smoke)
    return receipt


def evidence_files():
    paths = [SCRATCH/'smoke/receipt.json', SCRATCH/'pytest.txt', SCRATCH/'prepared/receipt.json']
    smoke = read(paths[0])
    reg_hash = sha(REG)
    need(smoke['scope'] == 'CPU_ENGINEERING_ONLY' and smoke['status'] == 'COMPLETE'
         and smoke['optimizer_steps'] == 0 and smoke['cuda_initialized'] is False
         and smoke['registration_sha256'] == reg_hash, 'CPU smoke evidence differs')
    need(read(paths[2])['registration_sha256'] == reg_hash, 'prepared evidence differs')
    need('PYTEST_EXIT_CODE=0' in paths[1].read_text(encoding='utf-8'), 'successful pytest evidence required')
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def screen(reg):
    validate(reg)
    out = SCRATCH/'production-v1'
    receipt = read(out/'receipt.json')
    need(receipt['scope'] == 'MATCHED_BN_ONLY' and receipt['status'] == 'COMPLETE'
         and receipt['registration_sha256'] == life.content_sha256(reg), 'complete production replay required')
    life._verify_files(out, receipt['files'])
    ids = reg['cohorts']['val_ids']
    predictions = {}
    for name, folder in [('mixed', out), ('rsna', out), ('control', ROOT/e.CONTROL), ('original', ROOT/OUTPUTS)]:
        predictions[name] = {v: life.load_prediction_csv(folder/(f'{name}-{v}.csv' if name in ('mixed','rsna')
                                else f'predictions-{v}.csv'), ids, reg['targets']) for v in VIEWS}
    result = dict(registration_sha256=life.content_sha256(reg), rows=888,
        scope=reg['interpretation'], contrasts=compare(ml.make_context(ids, ROOT), predictions))
    put(out/'screen.json', result)
    return dict(status='DESCRIPTIVE_SCREEN_COMPLETE', contrasts=12, rows=888)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['register','validate','prepare','smoke','admit','run','screen'])
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    if args.command != 'run': os.environ['CUDA_VISIBLE_DEVICES'] = ''
    setup()
    if args.command in ('register', 'smoke'):
        reg = make_registration()
        if args.command == 'register':
            put(REG, reg)
            result = dict(status='REGISTERED_BEFORE_GPU_OUTCOMES', sha256=sha(REG))
        else:
            result = execute(reg, smoke=True)
    else:
        reg = read(REG)
        if args.command == 'validate':
            validate(reg); result = dict(status='VALIDATED_CPU', sha256=sha(REG))
        elif args.command == 'prepare': result = prepare(reg)
        elif args.command == 'admit':
            validate(reg)
            evidence = evidence_files()
            review = read(args.review) if args.review else dict(verdict='HOLD', reviewer=None,
                registration_sha256=life.content_sha256(reg), evidence=evidence,
                approve_new_matched_policy=False, b_gpu_authorized=False)
            if args.review is None: put(SCRATCH/'review-template.json', review, scratch=True)
            result = admit(reg, review, evidence)
            put(SCRATCH/'admission.json', result, scratch=True)
        elif args.command == 'run': result = execute(reg, smoke=False, review=read(args.review) if args.review else {})
        else: result = screen(reg)
    # Never print IDs, label rows, or per-study predictions.
    print(json.dumps({k: v for k, v in result.items() if k not in ('cohorts','files')}, indent=2))


if __name__ == '__main__': main()
