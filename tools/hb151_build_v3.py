"""HB151 v3.0.2 single EXP143v2 soft-target build; CPU-passive imports.

Changelog: new stage, native probability aggregation, public-only registered
decision, exact generated trainer, bounded IO, smoke/review/identity gates.
v3.0.1: bounded startup publication, explicit recoverable export status, corpus check.
v3.0.2: measured 12 GiB phase caps plus desktop/2 GiB headroom; fresh stage.
No Kaggle/release-kernel operations. Only B's explicit smoke/fit command uses CUDA.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'training/src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
from tools import exp143_v2 as parent
from tools import hb151_ancestry_v2 as lineage
from tools.hb150_io_v1 import atomic_json, _retry, safe_replace, safe_unlink, _failure
from tools.pl_production_watchdog_v1 import PROGRESS, ProgressWriter

VERSION = '3.0.2'
STAGE = ROOT / '.local/scratch/astra-b4/H-B-151/v3.0.2'
PROTOCOL = STAGE / 'manifest.json'
RUN = STAGE / 'S42'
INIT = ROOT / 'storage/weights/exp143-imagenet-init/coatnet384-imagenet.pt'
CORPUS = parent.CORPUS
TAG = 'HB151_PL_COAT_S42'
EXPORT = STAGE / 'export/exp143-v2-e14-e11-e12-bn.pt'
TARGETS = parent.TARGETS
sha256 = parent.sha256
read_json = parent.read_json
need = lineage.need
DECISION = dict(baseline_public=.946, baseline_ledger_ref=57005140,
                alive_min=.950, dead_max=.947, otherwise='one more seed 43',
                evaluation='ONE standalone public-LB submission; no ratio tuning', seed=42,
                local_clean_validation=False, exposure='EXPOSED', clean_oof=False)
SOURCES = [*parent.CODE, 'tools/hb151_build_v3.py', 'tools/hb151_trainer_codegen_v3.py',
           'training/src/hb151_v3.py', 'tools/hb151_ancestry_v2.py',
           'tools/hb150_io_v1.py', 'tools/hb150_watchdog_v1.py',
           'tools/pl_production_watchdog_v1.py', 'tools/hb147_progress_v1.py',
           'tools/job_runtime.py', 'tools/gpu_job.py']


def public_decision(score):
    need(math.isfinite(score) and 0 <= score <= 1, 'invalid AUC')
    return 'alive' if score >= .950 else 'dead' if score <= .947 else 'seed43'


def probability_rows(path, *, metadata=False):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames and all(k in reader.fieldnames for k in [parent.UID, *TARGETS]), 'target columns')
        if not metadata:
            need(reader.fieldnames == [parent.UID, *TARGETS], 'unexpected label columns/order')
        rows, statuses = {}, {}
        for row in reader:
            uid = row[parent.UID]
            need(uid not in statuses, 'duplicate probability UID')
            status = row.get('status', 'ok')
            statuses[uid] = status
            if status != 'ok':
                continue  # whole-study fallback, never a cell repair
            values = [float(row[k]) for k in TARGETS]
            need(all(math.isfinite(x) and 0 <= x <= 1 for x in values), 'invalid probability cell')
            rows[uid] = values
    return rows, statuses


def mix_labels(ids, gold, r7, q7s, l2):
    need(len(ids) == len(set(ids)) and not set(ids) & set(gold), 'duplicate/Gold membership')
    output = {}; counts = dict(r7_q7s=0, r7_l2=0, l2_only=0)
    for uid in ids:
        need(uid in l2, 'missing complete L2 row')
        if uid not in r7:
            output[uid] = list(l2[uid]); counts['l2_only'] += 1
        else:
            use_q = uid in q7s
            secondary = q7s[uid] if use_q else l2[uid]
            output[uid] = [.7*a + .3*b for a, b in zip(r7[uid], secondary, strict=True)]
            counts['r7_q7s' if use_q else 'r7_l2'] += 1
        need(len(output[uid]) == 12 and all(math.isfinite(x) and 0 <= x <= 1 for x in output[uid]), 'bad mix')
    return output, counts


def build_labels():
    import numpy as np
    pins = {}
    def bound(rel, expected=None):
        path, digest = lineage.verify_file(ROOT, rel, expected)
        pins[rel] = digest
        return path
    auth = read_json(bound(lineage.AUTH, lineage.AUTH_SHA))
    base = read_json(bound(lineage.RECIPE, lineage.RECIPE_SHA))
    aligned = read_json(bound(lineage.ALIGNED, lineage.ALIGNED_SHA))
    rosters = []
    for fold in range(5):
        rel = f'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold{fold}/ordered-ids.json'
        rosters.append(read_json(bound(rel, auth['bindings'][rel])))
    groups = {}; members = []
    for entry in aligned['members']:
        member = entry['member']; family = member['family']; fold = member['fold']
        if family not in ('R50', 'C4', 'S'):
            continue
        roster = rosters[fold]
        need(not set(roster['train']) & set(roster['val']), 'fold leakage')
        need(member['train_uid_sha256'] in (lineage.uid_hash(roster['train']), lineage.uid_hash(sorted(roster['train'])))
             and member['heldout_uid_sha256'] == lineage.uid_hash(roster['val']), 'member roster authentication')
        prov_path = (Path(member['path']).parent / 'provenance.json').as_posix()
        provenance = read_json(bound(prov_path, auth['bindings'][prov_path]))
        need(provenance['train_ids_ordered_sha256'] == member['train_uid_sha256'] and
             provenance['eval_ids_ordered_sha256'] == member['heldout_uid_sha256'], 'provenance authentication')
        rows, _ = probability_rows(bound(entry['prediction'], entry['prediction_pin']))
        need(list(rows) == roster['val'], 'prediction heldout order')
        groups.setdefault((family, fold), []).append(np.asarray(list(rows.values()), dtype=np.float32))
        members.append(member['id'])
    need(len(members) == len(set(members)) == 25, '25 distinct authenticated leaf members required')
    q7s = {}
    for fold, roster in enumerate(rosters):
        families = []
        for family, expected_n in [('R50', 2), ('C4', 1), ('S', 2)]:
            arrays = groups[(family, fold)]
            need(len(arrays) == expected_n, 'family seed count')
            families.append(np.stack(arrays).mean(0, dtype=np.float32).astype(np.float64))
        values = sum(families)/3
        need(not set(q7s) & set(roster['val']), 'overlapping OOF folds')
        q7s.update(zip(roster['val'], values.tolist()))
    r7, statuses = probability_rows(bound(lineage.R7+'heldout_raw.csv',
        'b5b4e4b6e9d06183c4dc36940721f825f75bae6992a2f8d8e2ba983e739d29be'), metadata=True)
    receipt = read_json(bound(lineage.R7+'run_receipt.json'))
    bound(lineage.R7+'identity.json')
    need(receipt.get('exit_code') == 0, 'R7 run incomplete')
    need({u for u, status in statuses.items() if status == 'failure'} == set(parent.UNREADABLE_UIDS), 'R7 failure exclusions drift')
    l2, _ = probability_rows(bound(lineage.L2+'.csv', base['label_sha256']))
    bound(lineage.L2+'.provenance.json', base['label_provenance_sha256'])
    ids = base['train_ids']; gold = base['excluded_gold_ids']
    need(len(ids) == 4347 and len(gold) == 58 and len(q7s) == 4343 and set(q7s) <= set(ids), 'population drift')
    need(all(u in r7 for u in set(ids)-set(q7s)), 'four Q7S-missing studies require R7')
    labels, counts = mix_labels(ids, gold, r7, q7s, l2)
    provenance = dict(version=VERSION, exposure='EXPOSED', clean_oof=False, inputs_sha256=pins,
        source='authenticated R7 heldout-role + own Q7S leaf OOF; L2 fallback',
        probability_semantics='native probability-space outputs; no percentile ranks; no fitted post-hoc calibrator',
        calibration_verified=False, calibration_note='No separate calibration artifact was supplied. Native sigmoid outputs are not evidence of empirical calibration.',
        q7s_aggregation='FP32 seed mean within family/fold, then equal-family FP64 mean; no ranks',
        formula='0.7 R7 + 0.3 Q7S; missing Q7S: 0.7 R7 + 0.3 L2; missing R7: L2 only',
        counts=counts, train_rows=len(ids), q7s_rows=len(q7s), missing_q7s=4,
        gold_excluded=len(gold), r7_badzip_exclusions=2, silent_cell_fills=0,
        ancestry='STOP_ANCESTRY remains valid for local clean validation; public-only decision explicitly supersedes that design')
    return base, labels, provenance


def write_bytes(path, payload, *, critical=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    try:
        _retry(lambda: temporary.write_bytes(payload))
        return safe_replace(temporary, path, critical=critical)
    except OSError as exc:
        return _failure(path, exc, critical)
    finally:
        safe_unlink(temporary)


def prepare():
    from tools.hb151_trainer_codegen_v3 import verify
    need(not any(p.exists() for p in (RUN, STAGE/'smoke-attempt', STAGE/'independent-review.json')),
         'CUDA/review already started; stage immutable')
    diff = verify()
    base, labels, provenance = build_labels()
    STAGE.mkdir(parents=True, exist_ok=True)
    # First artifact is the decision, before model fitting is even possible.
    atomic_json(STAGE/'decision.json', DECISION)
    table = io.StringIO(newline=''); writer = csv.writer(table, lineterminator='\n')
    writer.writerow([parent.UID, *TARGETS])
    writer.writerows([u, *labels[u]] for u in base['train_ids'])
    write_bytes(STAGE/'soft-labels.csv', table.getvalue().encode())
    atomic_json(STAGE/'label-provenance.json', provenance)
    write_bytes(STAGE/'trainer.diff', diff.encode())
    p = dict(base)
    p.update(schema='hb151-pl-coat/v3', version=VERSION, status='REGISTERED_NO_TRAINING',
             train_labels=labels, run_dirs=dict(local=RUN.relative_to(ROOT).as_posix()),
             label_sha256=sha256(STAGE/'soft-labels.csv'), label_provenance=provenance,
             label_provenance_sha256=sha256(STAGE/'label-provenance.json'),
             label_file_name='soft-labels.csv', decision=DECISION, exposure='EXPOSED', clean_oof=False,
             parent_manifest_sha256=lineage.RECIPE_SHA,
             source_bindings={s:sha256(ROOT/s) for s in SOURCES})
    p.pop('content_sha256', None); p['content_sha256'] = parent.json_hash(p)
    # Fixed builder scratch can be rebuilt before any CUDA attempt or review.
    atomic_json(PROTOCOL, p)
    return dict(status=p['status'], rows=len(labels), **provenance['counts'], calibration_verified=False)


def load_registration(protocol=PROTOCOL, corpus=None, init_weights=None):
    need(Path(protocol).resolve() == PROTOCOL.resolve(), 'only registered v3 stage')
    p = read_json(protocol); content = p.pop('content_sha256')
    need(parent.json_hash(p) == content, 'registration hash drift'); p['content_sha256'] = content
    base_path, _ = lineage.verify_file(ROOT, lineage.RECIPE, lineage.RECIPE_SHA)
    base = read_json(base_path)
    for key in ('recipe', 'pos_weight', 'train_ids', 'excluded_gold_ids', 'budgets', 'init_sha256',
                'corpus_manifest_sha256', 'corpus_build_sha256'):
        need(p[key] == base[key], 'parent recipe drift: '+key)
    need(p['decision'] == DECISION and read_json(STAGE/'decision.json') == DECISION, 'decision drift')
    need(p['exposure'] == 'EXPOSED' and p['clean_oof'] is False, 'exposure drift')
    need(set(p['source_bindings']) == set(SOURCES), 'source list drift')
    for path, digest in p['source_bindings'].items():
        need(sha256(ROOT/path) == digest, 'source drift: '+path)
    need(sha256(STAGE/'soft-labels.csv') == p['label_sha256'] and
         sha256(STAGE/'label-provenance.json') == p['label_provenance_sha256'], 'label artifact drift')
    rows, _ = probability_rows(STAGE/'soft-labels.csv')
    need(list(rows) == p['train_ids'] and rows == p['train_labels'], 'label roster/value drift')
    need(not set(rows) & set(p['excluded_gold_ids']), 'Gold in labels')
    if corpus is not None:
        need(Path(corpus).resolve() == CORPUS.resolve(), 'wrong corpus path')
        need(sha256(Path(corpus)/'manifest.json') == p['corpus_manifest_sha256'] and
             sha256(Path(corpus)/'build-registration.json') == p['corpus_build_sha256'], 'corpus drift')
    if init_weights is not None:
        need(Path(init_weights).resolve() == INIT.resolve() and sha256(init_weights) == p['init_sha256'], 'init drift')
    from tools.hb151_trainer_codegen_v3 import verify
    verify()
    return p


def execution_settings(p, world, precision, seed, checkpointing=None):
    need((world, precision, seed, checkpointing) == (1, 'bf16', 42, False), 'exact local parent execution required')
    return parent.execution_settings(p, world, precision, seed, checkpointing)


def registered_run(p, lane='local'):
    need(lane == 'local' and p['run_dirs'] == {'local': RUN.relative_to(ROOT).as_posix()}, 'run path drift')
    return RUN


def write_new_json(path, payload):
    # Incomplete epoch replay may reproduce artifacts. Completed fit is immutable.
    need(not (RUN/'summary.json').exists(), 'completed fit immutable')
    atomic_json(path, payload)


def save_torch_atomic(path, payload):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    need(not (RUN/'summary.json').exists(), 'completed fit immutable')
    temporary = path.with_name(path.name+'.partial')
    PROGRESS.phase_start('checkpoint_write')
    try:
        with _retry(lambda: temporary.open('wb')) as stream:
            torch.save(payload, ProgressWriter(stream, PROGRESS))
            stream.flush(); os.fsync(stream.fileno())
        safe_replace(temporary, path)
    except OSError as exc:
        _failure(path, exc, True)
    finally:
        safe_unlink(temporary)


save_torch_new = save_torch_atomic  # restart-safe overwrite of incomplete epoch replay only


def export_checkpoint(source):
    temporary = EXPORT.with_suffix('.partial')
    try:
        EXPORT.parent.mkdir(parents=True, exist_ok=True)
        def copy():
            with Path(source).open('rb') as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, ProgressWriter(dst, PROGRESS), 4 << 20)
                dst.flush(); os.fsync(dst.fileno())
        _retry(copy)
        if safe_replace(temporary, EXPORT, critical=False):
            digest = sha256(EXPORT)
            checksum_ok = write_bytes(EXPORT.with_suffix('.sha256'), (digest+'  '+EXPORT.name+'\n').encode(), critical=False)
            metadata_ok = atomic_json(EXPORT.with_suffix('.json'), dict(sha256=digest, exposure='EXPOSED', clean_oof=False,
                protocol_sha256=sha256(PROTOCOL), serving_name=EXPORT.name), critical=False)
            if checksum_ok and metadata_ok:
                return dict(status='EXPORTED', sha256=digest, path=str(EXPORT))
    except OSError:
        pass  # export is recoverable; canonical checkpoint/summary remain authoritative
    finally:
        safe_unlink(temporary)
    return dict(status='EXPORT_PENDING', recovery='python -B -m tools.hb151_build_v3 export')


def recover_export():
    load_registration(PROTOCOL)
    summary = read_json(RUN/'summary.json')
    need(summary.get('status') == 'COMPLETE_TRAINING_NO_EVALUATION', 'no complete fit to export')
    need(sha256(RUN/'model_best.pt') == summary['model_sha256'], 'canonical checkpoint drift')
    result = export_checkpoint(RUN/'model_best.pt')
    need(result['status'] == 'EXPORTED', 'export failed; canonical checkpoint retained, retry CPU export')
    return result


def require_validation(protocol):
    v = read_json(STAGE/'cpu-validation.json')
    need(v.get('protocol_sha256') == sha256(protocol) and v.get('status') == 'PASS_CPU', 'CPU validation stale/missing')


def cpu_validate():
    import torch
    import hb151_v3 as trainer
    torch.set_num_threads(4)
    start = time.monotonic()
    p = load_registration(PROTOCOL, CORPUS, INIT)
    execution_settings(p, 1, 'bf16', 42, False)
    corpus = parent.check_corpus(CORPUS)
    need(set(corpus['ids']) == set(p['train_ids']) | set(p['excluded_gold_ids']), 'corpus population drift')
    result = trainer.tiny_validation()
    # Real architecture + real corpus paths and soft targets, full K12/384 forward.
    model = trainer.build_model(); trainer.load_imagenet(model, INIT); model.eval()
    ds = trainer.StudyDataset(CORPUS, p['train_ids'][:2], p['train_labels'], 42)
    x, y, _ = trainer.collate([ds[0]])
    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y,
            pos_weight=torch.tensor(p['pos_weight']))
    need(logits.shape == (1, 12) and torch.isfinite(loss).item(), 'real CPU forward/loss')
    bn_ds = trainer.StudyDataset(CORPUS, p['train_ids'][:2], p['train_labels'], 42, augment=False)
    bn_ds.set_epoch(16)
    bx, by, uids = trainer.collate([bn_ds[0], bn_ds[1]])
    from tools.exp112_full_swa import recompute_bn
    bn = recompute_bn(model, [(bx, by, uids)], 'cpu', trainer.amp_context('cpu', 'none'))
    # Persist and reload the real model schema on CPU, then exercise serving strict load.
    payload = dict(model=model.state_dict(), arch=trainer.ARCH, res=384, inference_res=320,
        lab=TARGETS, contract=trainer.CONTRACT, k_eval=94, norm='imagenet', window_policy='legacy',
        epoch=15, tag='S42', amp_train='bf16', amp_eval='fp16', fold=-1,
        label_set=p['recipe']['labels'], provenance={'protocol_sha256':sha256(PROTOCOL)}, seed=42,
        checkpoint_kind='fixed-swa-bn', swa_epochs_1based=[14,11,12], oai=False,
        exposure='EXPOSED', clean_oof=False)
    sample = STAGE/'cpu-checkpoint.pt'
    save_torch_atomic(sample, payload)
    model.load_state_dict(torch.load(sample, map_location='cpu', weights_only=True)['model'], strict=True)
    # This is explicitly a CPU fixture, never published as the trained export.
    result.update(status='PASS_CPU', protocol_sha256=sha256(PROTOCOL), real_forward=True,
                  real_bn_batch2=True, bn_batches=bn['batches'], real_strict_checkpoint_roundtrip=True,
                  corpus_receipts_and_headers=len(corpus['ids']), corpus_full_rehash=False,
                  seconds=time.monotonic()-start, cuda_executed=False)
    atomic_json(STAGE/'cpu-validation.json', result)
    return result


def python_blockers(text, allowed=()):
    blockers = []
    for row in csv.reader(io.StringIO(text)):
        if not row: continue
        need(len(row) == 2, 'unexpected nvidia-smi compute table')
        pid, name = row[0].strip(), row[1].strip()
        exe = name.replace('\\', '/').rsplit('/',1)[-1].lower()
        need(pid.isdigit(), 'invalid compute PID')
        if exe.startswith('python') and int(pid) not in set(allowed):
            blockers.append(int(pid))
    return blockers


def admission_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    need(result.returncode == 0, 'nvidia-smi admission unavailable')
    return result.stdout


def desktop_memory_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    need(result.returncode == 0, 'desktop memory snapshot unavailable')
    rows = list(csv.reader(io.StringIO(result.stdout)))
    need(len(rows) == 1 and len(rows[0]) == 4, 'single GPU required')
    name, total, used, free = [x.strip() for x in rows[0]]
    return dict(device_name=name,total_mib=float(total),used_mib=float(used),free_mib=float(free))


def startup_publication(operation, *, seconds=2.):
    """Bounded local startup handshake only; never waits for GPU availability."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return operation()
        except (FileNotFoundError, BlockingIOError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)


def require_gpu_job(mode, *, child=False):
    return startup_publication(lambda: _require_gpu_job(mode, child=child))


def _require_gpu_job(mode, *, child=False):
    import psutil
    from tools.job_runtime import file_lock, read_json as shared_read, command_digest
    tag = TAG if mode == 'fit' else TAG+'_SMOKE'
    with file_lock(ROOT/'artifacts/logs/.gpu-claim.lock', blocking=False):
        job = shared_read(ROOT/'artifacts/logs'/tag/'job.json')
        ancestors = [psutil.Process()] + psutil.Process().parents()
        identities = {(q.pid, q.create_time()) for q in ancestors}
        wrapper = (job['pid'], job['create_time'])
        direct = job.get('child_identity') or {}
        need(wrapper in identities and (direct.get('pid'), direct.get('create_time')) in identities,
             'must run under registered live gpu_job ancestry')
        need(job['tag'] == tag and job['phase'] == 'running' and
             job['command_sha256'] == command_digest(job['command'], ROOT), 'gpu_job command identity')
        expected = [sys.executable, '-B', '-m', 'tools.hb151_build_v3', mode]
        need(job['command'] in (expected, expected+['--resume']) if mode == 'fit' else job['command'] == expected,
             'unexpected gpu_job command')
        need(not python_blockers(admission_snapshot(), [q.pid for q in ancestors]), 'other Python ML compute is active')
        if child:
            attempt = STAGE/('smoke-attempt' if mode == 'smoke' else 'fit-attempt-'+str(read_json(STAGE/'budget.json')['attempt']))
            command = read_json(attempt/'command.json')
            # Unlike job_runtime.read_json, this preserves FileNotFoundError for
            # the bounded parent-publication handshake before any CUDA work.
            identity = read_json(attempt/'identity.json')
            need(identity['pid'] == os.getpid() and identity['create_time'] == psutil.Process().create_time(), 'watchdog child identity')
            need(command['token'] == os.environ.get('HB147_TOKEN') and
                 Path(os.environ.get('HB147_HEARTBEAT','')).resolve() == (attempt/'heartbeat.json').resolve(), 'watchdog binding')


def check_smoke(s):
    need(s.get('schema') == 'hb151-smoke/v3' and s.get('protocol_sha256') == sha256(PROTOCOL), 'smoke registration')
    need('4070 Ti SUPER' in s.get('device_name','') and 16000 <= s.get('total_memory_mib',0) <= 16400, 'real 4070 measurement required')
    need(s.get('shape') == dict(batch=1, accumulation=16, k=12, res=384, bn_batch=2,
        precision='bf16', grad_checkpointing=False), 'full recipe smoke shape')
    desktop = s.get('desktop_before_cuda', {})
    used = desktop.get('used_mib', float('nan'))
    total = desktop.get('total_mib', float('nan'))
    need(math.isfinite(used) and math.isfinite(total) and 0 <= used < total,
         'measured desktop memory required')
    peaks = []
    for phase in ('train','bn'):
        m = s['measurements'][phase]
        need(0 < m['peak_reserved_bytes'] <= 12*1024**3 and math.isfinite(m['seconds_per_unit']) and m['seconds_per_unit'] > 0, 'measured memory/timing gate')
        peaks.append(m['peak_reserved_bytes'])
    # Both totals are measured; use the smaller one (CUDA may round differently).
    need(used*1024**2 + max(peaks) + 2*1024**3 <= min(total, s['total_memory_mib'])*1024**2,
         'measured desktop headroom gate')
    need(s['elapsed_seconds'] <= 175 and s['train_updates'] >= 2 and s['bn_batches'] >= 2, 'smoke incomplete/time cap')


def require_fit():
    load_registration(PROTOCOL, CORPUS, INIT); require_validation(PROTOCOL)
    s = read_json(STAGE/'cuda-smoke.json'); check_smoke(s)
    receipt = read_json(STAGE/'smoke-attempt/receipt.json')
    need(receipt['exit_code'] == 0 and receipt['reason'] == 'completed' and not receipt['survivors'], 'smoke supervisor failed')
    review = read_json(STAGE/'independent-review.json')
    need(review.get('verdict') == 'PASS' and review.get('reviewer') not in (None, '', 'astra-b4') and
         review.get('independent') is True, 'independent review required')
    for name, path in [('protocol_sha256',PROTOCOL), ('smoke_sha256',STAGE/'cuda-smoke.json'),
                       ('smoke_receipt_sha256',STAGE/'smoke-attempt/receipt.json'),
                       ('cpu_validation_sha256',STAGE/'cpu-validation.json')]:
        need(review.get(name) == sha256(path), 'stale review: '+name)
    # User requested calibrated probabilities. Never silently reclassify native
    # sigmoid means as empirically calibrated. B must resolve this in the review.
    need(review.get('probability_semantics') == 'native-probability-space accepted by owner',
         'native probabilities lack calibration evidence; owner must accept semantics before fit')
    return s


def require_child(mode):
    require_gpu_job(mode, child=True)
    if mode == 'fit': require_fit()


def charged_seconds():
    path = STAGE/'budget.json'
    return read_json(path)['charged_seconds'] if path.exists() else 0.


def cuda_smoke():
    require_child('smoke')
    import torch
    import hb151_v3 as t
    from tools.exp112_full_swa import recompute_bn
    start = time.monotonic(); p = load_registration(PROTOCOL, CORPUS, INIT); require_validation(PROTOCOL)
    desktop = desktop_memory_snapshot()
    torch.set_num_threads(4); torch.cuda.set_device(0)
    need(torch.cuda.is_bf16_supported(), 'bf16 required')
    torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = False
    t._cpu_seed(42); torch.cuda.manual_seed_all(42)
    model = t.build_model(); t.load_imagenet(model, INIT)
    model.backbone.set_grad_checkpointing(False); model.cuda().train()
    optimizer, scheduler, scaler = t.make_optimization(model, p['recipe'], 4352, 'bf16')
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(p['pos_weight'], device='cuda'))
    ds = t.StudyDataset(CORPUS, p['train_ids'][:32], p['train_labels'],42)
    torch.cuda.reset_peak_memory_stats(); times = []
    for update in range(2):
        PROGRESS.phase_start('training_step'); torch.cuda.synchronize(); tick=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for i in range(16):
            x,y,_ = t.collate([ds[update*16+i]])
            with t.amp_context('cuda','bf16'):
                loss=loss_fn(model(x.cuda()).float(),y.cuda())/16
            need(torch.isfinite(loss).item(), 'nonfinite smoke loss'); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.monotonic()-tick)
        PROGRESS.completed('training_step',{'update':update+1})
    measurements={'train':dict(peak_reserved_bytes=torch.cuda.max_memory_reserved(), seconds_per_unit=times[-1])}
    del loss,x,y
    torch.cuda.reset_peak_memory_stats()
    bn_ds=t.StudyDataset(CORPUS,p['train_ids'][:4],p['train_labels'],42,augment=False); bn_ds.set_epoch(16)
    def batches():
        PROGRESS.phase_start('bn_replay')
        for i in range(2):
            yield t.collate([bn_ds[2*i],bn_ds[2*i+1]])
            PROGRESS.completed('bn_replay',{'progress_kind':'bn_replay','cumulative_units':i+1})
    torch.cuda.synchronize(); tick=time.monotonic()
    recompute_bn(model,batches(),'cuda',t.amp_context('cuda','bf16')); torch.cuda.synchronize()
    measurements['bn']=dict(peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds_per_unit=(time.monotonic()-tick)/2)
    properties=torch.cuda.get_device_properties(0)
    s=dict(schema='hb151-smoke/v3',protocol_sha256=sha256(PROTOCOL),device_name=properties.name,
        total_memory_mib=properties.total_memory/1024**2, measurements=measurements,
        shape=dict(batch=1,accumulation=16,k=12,res=384,bn_batch=2,precision='bf16',grad_checkpointing=False),
        train_updates=2,bn_batches=2,elapsed_seconds=time.monotonic()-start,
        desktop_before_cuda=desktop,
        exposure='EXPOSED',clean_oof=False)
    atomic_json(STAGE/'cuda-smoke.json',s); check_smoke(s)


def supervise(mode, resume=False):
    from tools.hb150_watchdog_v1 import supervise as watch
    require_gpu_job(mode)
    load_registration(PROTOCOL, CORPUS, INIT); require_validation(PROTOCOL)
    command=[sys.executable,'-B','-m','tools.hb151_build_v3','_smoke' if mode=='smoke' else '_fit']
    if resume: command.append('--resume')
    if mode=='smoke':
        return watch(command,STAGE/'smoke-attempt',total=165,poll=1)
    s=require_fit()
    previous=read_json(STAGE/'budget.json') if (STAGE/'budget.json').exists() else {'attempt':0,'charged_seconds':0}
    if previous.get('running'):
        # Charge the entire reservation after supervisor loss. No unbounded retry budget.
        raise ValueError('prior supervisor did not close budget; B must reconcile before resume')
    need(not RUN.exists() or resume, 'existing run requires --resume')
    remaining=14*3600-previous['charged_seconds']; need(remaining>600,'fit budget exhausted')
    attempt=previous['attempt']+1; budget=dict(attempt=attempt,charged_seconds=previous['charged_seconds'],running=True)
    atomic_json(STAGE/'budget.json',budget); start=time.monotonic()
    try:
        return watch(command,STAGE/('fit-attempt-'+str(attempt)),total=remaining,
            healthy_rates={'train_step':1/s['measurements']['train']['seconds_per_unit'],
                           'bn_replay':1/s['measurements']['bn']['seconds_per_unit']})
    finally:
        budget.update(running=False,charged_seconds=previous['charged_seconds']+time.monotonic()-start)
        atomic_json(STAGE/'budget.json',budget)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','validate','admit-dry-run','smoke','fit','_smoke','_fit','export'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.action=='prepare': print(json.dumps(prepare())); return 0
    if args.action=='validate': print(json.dumps(cpu_validate())); return 0
    if args.action=='admit-dry-run':
        load_registration(PROTOCOL,CORPUS,INIT); require_validation(PROTOCOL)
        snapshot=admission_snapshot(); write_bytes(STAGE/'nvidia-compute-apps.txt',snapshot.encode())
        blockers=python_blockers(snapshot)
        gates=[]
        try: require_fit()
        except (OSError,ValueError,KeyError) as exc: gates.append(str(exc))
        result=dict(status='CPU_DRY_RUN',python_blockers=blockers,fit_admitted=False,
                    pending=gates,gpu_job_required=True,cuda_executed=False,
                    desktop_memory=desktop_memory_snapshot())
        atomic_json(STAGE/'admission-dry-run.json',result); print(json.dumps(result)); return 0
    if args.action in ('smoke','fit'): return supervise(args.action,args.resume)
    if args.action=='_smoke': cuda_smoke(); return 0
    if args.action=='export':
        print(json.dumps(recover_export())); return 0
    import hb151_v3 as trainer
    trainer.train(argparse.Namespace(protocol=PROTOCOL,corpus=CORPUS,init_weights=INIT,device='cuda',
        precision='bf16',seed=42,grad_checkpointing=False,workers=12,resume=args.resume))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
