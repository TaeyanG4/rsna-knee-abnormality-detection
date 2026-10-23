"""EXP123 (H-B-43): Qwen3-VL-2B-Instruct fold-0 pilot registration and queue entries.

  python tools/exp123_qwen3vl_pilot.py prepare            # writes EXP123_QWEN3VL_PILOT_PROTOCOL_V1/{manifest.json, queue-entry.json, queue-entry-probe.json}
  python tools/exp123_qwen3vl_pilot.py queue-json         # prints the training queue entry (does not start anything)
  python tools/exp123_qwen3vl_pilot.py argv               # prints the registered runner argv
  python tools/exp123_qwen3vl_pilot.py status

Fold 0 of EXP099_KFOLD_PROTOCOL_V1 = G1 T3455 / V888, i.e. the same train/val as EXP099 fold 0 (EXP094L_B8_SEED42) and the CoAtNet
fold-0 K36 OOF (EXP096_K_SWEEP/B_K36.csv). Training labels Dread (labels_llm_soft.csv, float32 soft targets), pos_weight = the frozen
EXP099 fold-0 vector (clip(mean, .03, .7) -> clip((1-p)/p, 1, 10)). Inputs: RAPTOR64_336_C140_S02 via raptor_common Corpus/select_centers
(within-slot-valid)/windows_from, 24 three-slice windows per study resized 336 -> 384, packed as 12 two-frame temporal patches
(slot-aware pairing, training/runners/exp123_qwen3vl_pilot.py:pair_indices). Model, recipe, gate and pre-registered evaluation are frozen in
the manifest; the runner refuses any other argument value in --mode full. Nothing here trains, starts a queue or touches CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 'exp123-qwen3vl-pilot-v1'
PLAN_ITEM = 'H-B-43'
PROTOCOL = 'artifacts/experiments/EXP123_QWEN3VL_PILOT_PROTOCOL_V1'
RUNNER = 'training/runners/exp123_qwen3vl_pilot.py'
PROBE = 'tools/exp123_fp16_probe.py'
EVAL = 'tools/exp123_qwen3vl_eval.py'
TAG = 'EXP123_QWEN3VL_PILOT_F0'
PROBE_TAG = 'EXP123_QWEN3VL_FP16_PROBE_F0'
AFTER = 'EXP113_RADR50_LOCAL_F4'
VLM_PYTHON = ROOT / '.venv-vlm' / 'Scripts' / 'python.exe'
WEIGHTS = 'storage/weights/qwen__Qwen3-VL-2B-Instruct'
WEIGHT_FILES = ('config.json', 'preprocessor_config.json', 'video_preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json',
                'chat_template.json', 'provenance.json', 'model.safetensors')
HF_REPO = 'Qwen/Qwen3-VL-2B-Instruct'
HF_SHA = '89644892e4d85e24eaac8bacfd4f463576704203'
K99 = 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1'
FOLD = 0
DREAD = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
COATNET_F0_K36 = 'artifacts/experiments/EXP096_K_SWEEP/B_K36.csv'
COATNET_F0_RUN = 'artifacts/experiments/EXP094L_B8_SEED42'
RADR50_OOF_K36 = 'artifacts/experiments/EXP107_RADR50_DREAD_OOF/oof_k36.csv'
RADR50_LOCAL_F0_K36 = 'artifacts/experiments/EXP113_RADR50_LOCAL_F0/oof_k36.csv'
SOURCE_CODE = ('training/src/raptor_common.py', RUNNER, 'tools/exp123_qwen3vl_pilot.py', PROBE, EVAL)
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
REGISTERED_ENV = {**THREAD_ENV, 'HF_HUB_OFFLINE': '1'}

# Registered recipe: every runner argument below must equal these values in --mode full (feasibility P1, vlm_feasibility.md section 4).
RECIPE = dict(
    fold=FOLD, seed=42, device='cuda', epochs=3, k_train=24, k_eval=24, k_eval_extra=36, res=384, accum=8,
    lr_vit=2e-5, lr_lora=2e-4, lr_head=1e-3, wd=0.01, warmup_frac=0.03, clip=1.0,
    lora_r=16, lora_alpha=32, lora_dropout=0.05, amp='bf16', eval_amp='bf16', grad_ckpt=True, jitter=0.2, head_dropout=0.0,
    workers=4, gate_steps=48, gate_max_hours=4.0, gate_max_peak_gib=14.5, gate_eval_fraction=0.35,
    text_layers=0, vision_layers=0, max_train=0, max_val=0, max_steps=0, save_weights=True,
)
STOP_RULES = [
    'consensus-OOF (D-B-035 pairs, VLM K24 fold-0 predictions) more than 0.01 below CoAtNet fold 0 (EXP096_K_SWEEP/B_K36.csv) -> STOP '
    '(point estimate; decided by tools/exp123_qwen3vl_eval.py)',
    'projected T4 serving for 1,300 studies > 3 h -> STOP (tools/exp123_fp16_probe.py reports local timing; T4 factor is an assumption)',
    'AMENDED 2026-10-07 21:23 KST before any run: a non-finite fp16 output on 100 validation studies switches serving to fp32 '
    '(fp32 T4 timing must be projected separately) instead of stopping (tools/exp123_fp16_probe.py)',
]
GATE = ('G0 inside the run (vlm_feasibility.md section 4): after gate_steps micro-steps of epoch 1 the runner projects training + K24/K36 '
        'evaluation from the measured mean micro-step time (steps gate_steps//3 .. gate_steps) and requires projected <= gate_max_hours and '
        'peak allocated CUDA memory <= gate_max_peak_gib; otherwise it writes gate-receipt.json GATE_FAILED and exits 3 (no retry; fall back '
        'to P0 = K12/6 pairs only by a new registered protocol)')


# ------------------------------------------------------------------ small helpers
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def uid_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def rel(root, value):
    p = Path(str(value).removeprefix('workspace://'))
    return p if p.is_absolute() else Path(root) / p


def fold_paths(root=ROOT):
    p = read(Path(root) / K99 / 'manifest.json')
    f = p['folds'][str(FOLD)]
    return p, f


def fold_ids(root=ROOT):
    """(train ids, val ids) of EXP099 fold 0 in canonical order (no validation; the runner validates in full mode)."""
    _, f = fold_paths(root)
    o = read(rel(root, f['ordered_ids']['path']))
    return list(o['train']), list(o['val'])


def runner_args(recipe=RECIPE):
    """Runner flags for the registered recipe (booleans use --name / --no-name)."""
    out = []
    for k, v in recipe.items():
        flag = '--' + k.replace('_', '-')
        if isinstance(v, bool):
            out.append(flag if v else '--no-' + k.replace('_', '-'))
        else:
            out += [flag, str(v)]
    return out


def runner_argv(root=ROOT, python=None):
    root = Path(root)
    return [str(python or VLM_PYTHON), str(root / RUNNER), '--mode', 'full', '--protocol', str(root / PROTOCOL / 'manifest.json'), '--tag', TAG,
            '--weights', str(root / WEIGHTS), '--corpus', str(root / CORPUS), '--labels', str(root / DREAD), *runner_args()]


def probe_argv(root=ROOT, python=None):
    root = Path(root)
    return [str(python or VLM_PYTHON), str(root / PROBE), '--checkpoint', str(root / 'artifacts/experiments' / TAG / 'weights_epoch3.pt'),
            '--n-studies', '100', '--device', 'cuda', '--out', str(root / 'artifacts/experiments' / PROBE_TAG)]


def queue_entries(root=ROOT):
    # gpu_queue.py runs entries serially; a G0 failure blocks the dependent probe.
    return [dict(tag=TAG, command=runner_argv(root), after=AFTER, continue_on_failure=False, exclusive=True),
            *probe_queue_entries(root)]


def probe_queue_entries(root=ROOT):
    # Runs only after a successful pilot (continue_on_failure false): it needs the trained weights.
    return [dict(tag=PROBE_TAG, command=probe_argv(root), after=TAG, continue_on_failure=False, exclusive=True)]


# ------------------------------------------------------------------ prepare (CPU, writes a new protocol directory only)
def bindings(root=ROOT):
    root = Path(root)
    k99, f0 = fold_paths(root)
    paths = [K99 + '/manifest.json', f0['ordered_ids']['path'], f0['groups']['path'], DREAD, CORPUS + '/manifest.json', *SOURCE_CODE,
             *[WEIGHTS + '/' + n for n in WEIGHT_FILES]]
    return {p: sha256(root / p) for p in paths}


def prepare(root=ROOT):
    root = Path(root)
    dest = root / PROTOCOL
    if dest.exists():
        raise SystemExit('Refusing to overwrite existing registration: ' + PROTOCOL)
    k99, f0 = fold_paths(root)
    if sha256(rel(root, f0['ordered_ids']['path'])) != f0['ordered_ids']['sha256'] or sha256(rel(root, f0['groups']['path'])) != f0['groups']['sha256']:
        raise SystemExit('EXP099 fold-0 files changed')
    if k99['paths']['corpus'] != CORPUS or sha256(root / CORPUS / 'manifest.json') != k99['corpus_manifest_sha256']:
        raise SystemExit('corpus differs from EXP099')
    if k99['label_sets']['dread']['path'] != DREAD or sha256(root / DREAD) != k99['label_sets']['dread']['sha256']:
        raise SystemExit('Dread file differs from EXP099')
    train, val = fold_ids(root)
    if (len(train), len(val)) != (f0['n_train'], f0['n_val']) or (len(train), len(val)) != (3455, 888):
        raise SystemExit('fold 0 is not T3455/V888')
    if uid_hash(train) != f0['ordered_ids']['train_sha256'] or uid_hash(val) != f0['ordered_ids']['val_sha256']:
        raise SystemExit('fold 0 ordered ids changed')
    for need in (COATNET_F0_K36, RADR50_OOF_K36):
        if not (root / need).is_file():
            raise SystemExit('missing comparator ' + need)
    t0 = time.time()
    b = bindings(root)
    st = root / WEIGHTS / WEIGHT_FILES[-1]
    weight_provenance = read(root / WEIGHTS / 'provenance.json')
    if weight_provenance['weights_sha256'] != b[WEIGHTS + '/model.safetensors'] or weight_provenance['license'].lower() != 'apache-2.0':
        raise SystemExit('weight provenance does not match the local snapshot')
    n_steps = len(train)  # batch 1 micro-steps per epoch
    updates_per_epoch = -(-n_steps // RECIPE['accum'])
    manifest = dict(
        schema=SCHEMA, plan_item=PLAN_ITEM, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status='REGISTERED_NOT_RUN',
        tag=TAG, after=AFTER, fold=FOLD, n_train=len(train), n_val=len(val),
        split=dict(source=K99, fold=FOLD, ordered_ids=f0['ordered_ids'], groups=f0['groups'],
                   same_as=['EXP099 fold 0', 'EXP094L_B8_SEED42 (reused as EXP099 fold 0)', 'EXP084_B_G1 / EXP096_K_SWEEP B_K36.csv']),
        labels=dict(train=DREAD, train_rule='float32 soft targets (as EXP099)', semantic_sha256=k99['label_sets']['dread']['semantic_sha256'],
                    pos_weight=f0['pos_weight'], pos_weight_rule=k99['pos_weight_rule']),
        corpus=dict(path=CORPUS, manifest_sha256=k99['corpus_manifest_sha256'], window_policy='within-slot-valid',
                    selection='raptor_common.select_centers; train: study_rng(seed, epoch, uid) sample of k_train then sorted; eval: evenly spaced k_eval',
                    windows='raptor_common.windows_from (three consecutive cached planes, bilinear 336 -> res), [0,1]',
                    jitter='train only: x * (1 + (u - .5) * jitter), u = study_rng(seed, epoch, uid, "intensity"), clip [0,1] (as raptor_data.StudyWindows)',
                    normalisation='Qwen processor mean/std read from preprocessor_config.json (0.5/0.5); no ImageNet normalisation',
                    pairing='slot-aware: consecutive windows of one slot form a two-frame temporal patch; odd leftovers of different slots pair in slot order',
                    packing='each pair -> (2,3,res,res) -> Qwen3-VL video patchify order (C, t, p, p) -> 576 patches x 1536 at 384 px, grid [1,24,24], 144 LM tokens'),
        model=dict(hf_repo=HF_REPO, hf_sha=HF_SHA, weights=WEIGHTS, safetensors_bytes=st.stat().st_size,
                   licence='Apache-2.0 (README.md model card; no LICENSE in snapshot)',
                   cls='transformers Qwen3VLModel (no lm_head)', base_dtype='bfloat16 (frozen language model)',
                   vision='vision tower fully trainable, fp32 master weights',
                   lora='peft LoRA on language-model linear layers only: q/k/v/o_proj, gate/up/down_proj',
                   head='masked mean of last-layer hidden states over image tokens ++ last-token state -> LayerNorm(4096) -> Linear(4096, 12), fp32',
                   prompt='fixed ChatML user turn with the image pairs and a short instruction; see runner PROMPT_TEXT'),
        optimisation=dict(loss='BCEWithLogits(pos_weight) on fp32 logits', optimizer='AdamW, groups vit/lora/head', schedule='linear warmup then cosine to 0 per optimizer update',
                          micro_batch=1, updates_per_epoch=updates_per_epoch, total_updates=updates_per_epoch * RECIPE['epochs'],
                          precision='bf16 autocast, no GradScaler, non-finite loss stops the run', checkpoint='fixed-final (epoch 3); weights saved every epoch'),
        args=RECIPE, gate=GATE, registered_env=REGISTERED_ENV,
        outputs=dict(primary='oof_k24.csv (888 rows, UID + 12 targets, fold-0 val canonical order; the format of oof_k36.csv files)',
                     diagnostic='oof_k36.csv (18 pairs; never used for the stop decision)',
                     weights='weights_epoch{1,2,3}.pt (trainable parameters fp32: vision tower, LoRA, head)',
                     receipts=['provenance.json', 'history.json', 'gate-receipt.json', 'summary.json', 'output-inventory.json']),
        evaluation=dict(script=EVAL, primary='consensus-pair macro AUC (D-B-035) of VLM K24 minus CoAtNet fold 0 K36',
                        comparators=dict(coatnet_f0_k36=COATNET_F0_K36, radr50_k36_fold0_rows=RADR50_OOF_K36, radr50_local_f0_k36=RADR50_LOCAL_F0_K36 + ' (if present)'),
                        references=['rint(Dread)', 'flight_v4 >= 0.5', 'relabel_v1 >= 0.5', 'consensus pairs of Dread/Steven v4/Pilkwang v2/relabel_v1 (all >= 0.5 agree)'],
                        bootstrap='component bootstrap over VALIDATION_GROUP_CONTRACT_V1 components, seed 20261003',
                        metric_access='none inside the runner (fixed-final, deferred single evaluation)'),
        stop_rules=STOP_RULES,
        backbone_change='Qwen3.5-2B hybrid attention -> Qwen3-VL-2B standard attention with DeepStack; 384 px, 12 temporal frame pairs, 144 tokens per pair unchanged',
        fp16_inference_risk='BF16 training does not establish fp16 serving safety. Probe only after pilot SUCCESS; non-finite fp16 output switches serving to fp32.',
        estimated_runtime=dict(local_rtx4070_ti_super_hours='2.0-3.5 total, unmeasured engineering range; G0 <=4h registered cutoff',
                               kaggle_2xt4_1300_studies_fp16_minutes='10-40, unmeasured; two independent workers, ~650 studies each; decoding excluded'),
        source_bindings=b, binding_seconds=round(time.time() - t0, 1),
        caveats=['one fold, one seed; fold-0 val is the exposed G1 V888', 'Dread weak labels', 'feasibility numbers are estimates (vlm_feasibility.md)',
                 'the author recipe (D-B-013) is unknown beyond model, 384 px and LoRA-on-LM + joint vision fine-tune',
                 'Qwen3-VL has DeepStack standard attention; first local RTX 4070 Ti SUPER timing is G0 only',
                 'fp16 serving safety unverified until a successful pilot is probed'],
    )
    dest.mkdir(parents=True)
    dump(dest / 'manifest.json', manifest)
    dump(dest / 'queue-entry.json', queue_entries(root))
    dump(dest / 'queue-entry-probe.json', probe_queue_entries(root))
    print(json.dumps(dict(status='PREPARED', protocol=PROTOCOL, binding_seconds=manifest['binding_seconds'],
                          total_updates=manifest['optimisation']['total_updates']), indent=1))
    return manifest


# ------------------------------------------------------------------ runner guard (called before training)
def supply_registered_env(environ=None):
    """Thread/offline variables are supplied when absent (a queue runner may lack them, D-B-017); different values are refused later."""
    environ = os.environ if environ is None else environ
    supplied = [k for k in REGISTERED_ENV if k not in environ]
    for k in supplied:
        environ[k] = REGISTERED_ENV[k]
    return supplied


def validate_arguments(args, root=ROOT, hash_weights=True):
    """Full-mode guard: protocol identity, frozen bindings, registered recipe/tag/environment. Returns (manifest, train ids, val ids)."""
    root = Path(root)
    mpath = Path(args.protocol)
    if mpath.resolve() != (root / PROTOCOL / 'manifest.json').resolve():
        raise ValueError('wrong protocol path')
    m = read(mpath)
    if m.get('schema') != SCHEMA or m.get('tag') != TAG:
        raise ValueError('wrong protocol schema/tag')
    if args.tag != TAG:
        raise ValueError('tag differs from the registered tag')
    for k, v in m['args'].items():
        if getattr(args, k, None) != v:
            raise ValueError(f'frozen argument changed: {k} = {getattr(args, k, None)!r} (registered {v!r})')
    for k, v in REGISTERED_ENV.items():
        if os.environ.get(k) != v:
            raise ValueError('registered environment required: ' + k)
    for name, want in (('weights', WEIGHTS), ('corpus', CORPUS), ('labels', DREAD)):
        if Path(getattr(args, name)).resolve() != (root / want).resolve():
            raise ValueError(f'--{name} differs from the registered path')
    for path, expected in m['source_bindings'].items():
        if path.endswith('.safetensors') and not hash_weights:
            if (root / path).stat().st_size != m['model']['safetensors_bytes']:
                raise ValueError('weights size changed')
            continue
        if sha256(root / path) != expected:
            raise ValueError('frozen binding changed: ' + path)
    o = read(rel(root, m['split']['ordered_ids']['path']))
    train, val = list(o['train']), list(o['val'])
    if uid_hash(train) != m['split']['ordered_ids']['train_sha256'] or uid_hash(val) != m['split']['ordered_ids']['val_sha256']:
        raise ValueError('fold-0 ordered ids changed')
    if (len(train), len(val)) != (m['n_train'], m['n_val']):
        raise ValueError('fold-0 sizes changed')
    return m, train, val


def validate_membership(root, m, corpus_ids, train, val):
    """EXP099 fold-membership rules (exact cover of the group file, canonical order, no component across train/val)."""
    from tools import exp099_kfold as k99
    rows = k99.read_rows(rel(root, m['split']['groups']['path']))
    base = read(Path(root) / k99.PROTOCOL / 'manifest.json')
    canonical = k99.canonical_order(read(rel(root, base['paths']['base_ordered_ids'])))
    return k99.validate_fold_membership(list(corpus_ids), dict(train=train, val=val), rows, canonical)


# ------------------------------------------------------------------ CLI
def status():
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from raptor_common import LOGS
    out = {}
    for tag in (AFTER, TAG, PROBE_TAG):
        d = LOGS / tag
        out[tag] = 'SUCCESS' if (d / 'SUCCESS').exists() else 'FAILED' if (d / 'FAILED').exists() else 'running' if (d / 'job.json').exists() else 'absent'
    for f in ('gate-receipt.json', 'summary.json'):
        p = ROOT / 'artifacts/experiments' / TAG / f
        if p.exists():
            out[f] = read(p).get('status', 'present')
    print(json.dumps(out, indent=1))


def cpu_check(root=ROOT):
    """Load the complete backbone on CPU with strict key checks; write counts and prompt lengths."""
    import importlib.util
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    torch.set_default_device('cpu')
    root = Path(root)
    spec = importlib.util.spec_from_file_location('exp123_cpu_runner', root / RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    model, cfg, report = runner.build_model(root / WEIGHTS, grad_ckpt=False)
    model.to('cpu')
    tok = AutoTokenizer.from_pretrained(root / WEIGHTS, local_files_only=True)
    lengths = {}
    for k in (24, 36):
        ids, types, _ = runner.build_prompt(tok, k // 2, 144, runner.token_ids(cfg))
        lengths[str(k)] = dict(total=len(ids), image=sum(types), pairs=k // 2)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU audit initialized CUDA')
    body = dict(schema='exp123-cpu-check-v1', model=report, prompt_tokens=lengths,
                cuda_initialized=False, weights_sha256=sha256(root / WEIGHTS / 'model.safetensors'))
    dest = root / PROTOCOL / 'cpu-check.json'
    dump(dest, body)
    print(json.dumps(body, indent=2))
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare'); sub.add_parser('queue-json'); sub.add_parser('argv'); sub.add_parser('status'); sub.add_parser('cpu-check')
    a = ap.parse_args(argv)
    if a.command == 'prepare':
        prepare()
    elif a.command == 'queue-json':
        print(json.dumps(queue_entries(), indent=2))
    elif a.command == 'argv':
        print(json.dumps(runner_argv(), indent=1))
    elif a.command == 'status':
        status()
    else:
        cpu_check()


if __name__ == '__main__':
    main()
