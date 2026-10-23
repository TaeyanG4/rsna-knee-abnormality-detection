"""HC06 frozen patch-MIL preparation/guards and bounded pilot-to-full execution.

Run only through gpu_job/gpu_queue after coordinator resource/claim checks.
Importing this module performs no model loading, training, CUDA, or IO.
"""
from __future__ import annotations
import argparse
import copy
import csv
import difflib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp086_dinov2s_v2 as base
from training.adapters import dinov2_patch_mil as adapter

SCHEMA = 'hc06-patch-mil-v1'
TAG = 'HC06_DINOV2_PATCH_MIL_V1'
PROTOCOL = 'artifacts/experiments/HC06_PATCH_MIL_PROTOCOL_V1'
RUNNER = 'training/runners/hc06_patch_mil.py'
PARENT = base.PROTOCOL + '/manifest.json'
REVIEW = '.local/hc06-patch-mil-v1/final_review.json'
AGGREGATION_POLICY = adapter.POLICY
POSITION_POLICY = base.POSITION_POLICY
WEIGHT_SHA = base.WEIGHT_SHA
CODE_NAMES = base.CODE_NAMES
need, read, sha256, write_new = base.need, base.read, base.sha256, base.write_new
uid_hash, read_rows, worker_init = base.uid_hash, base.read_rows, base.worker_init
install_gold_blocker = base.install_gold_blocker
validate_membership, validate_label_contract = base.validate_membership, base.validate_label_contract
loss_weights = base.loss_weights


def build_offline_model(arch, asset):
    return adapter.install(base.build_offline_model(arch, asset))


def load_patch_checkpoint(path):
    """Explicit reconstruction; never use vanilla raptor_common.load_checkpoint."""
    import torch
    if str(ROOT/'training/src') not in sys.path:
        sys.path.insert(0,str(ROOT/'training/src'))
    from raptor_common import build_model
    from training.adapters.dinov2_cpu_pos_v2 import install as install_position
    ck = torch.load(path, map_location='cpu', weights_only=False)
    need(ck.get('aggregation_policy') == AGGREGATION_POLICY, 'patch checkpoint policy missing')
    need(ck['arch'] == base.ARCH and ck['res'] == 224, 'patch checkpoint shape contract')
    model = build_model(ck['arch'], pretrained=False)
    install_position(model.backbone); adapter.install(model)
    model.load_state_dict(ck['model'], strict=True)
    return model, {k: v for k, v in ck.items() if k != 'model'}


def render_runner():
    s = (ROOT/base.RUNNER).read_text(encoding='utf-8')
    old = 'from tools import exp086_dinov2s_v2 as guard'
    need(s.count(old) == 1, 'original guard import changed')
    s = s.replace(old, 'from tools import patch_mil as guard')
    old = '    guard.validate_pretraining(provenance, run_environment, protocol, arm)'
    need(s.count(old) == 1, 'original provenance call changed')
    s = s.replace(old, '    provenance.update(aggregation_policy=guard.AGGREGATION_POLICY)\n'+old)
    old = '"epoch": epoch, "tag": a.tag, "amp_train": a.amp'
    need(s.count(old) == 1, 'checkpoint metadata location changed')
    s = s.replace(old, '"aggregation_policy": guard.AGGREGATION_POLICY, '+old)
    return s


def arm_for(protocol, tag):
    need(tag in (TAG, TAG+'_PILOT'), 'unregistered patch tag')
    return protocol['arm']


def validate_protocol(path, root=ROOT, require_review=False):
    root = Path(root); p = read(path)
    need(p['schema'] == SCHEMA and p['aggregation_policy'] == AGGREGATION_POLICY, 'wrong HC06 protocol')
    need(p['execution']['pilot_seconds'] == 180 and p['execution']['full_seconds'] == 1800
         and p['execution']['total_seconds'] == 1980 and p['execution']['retries'] == 0, 'wrong HC06 budget')
    # Original protocol checks all original source/input/installed-source bindings.
    old, ids = base.validate_protocol(root/PARENT, root, True)
    need(sha256(root/PARENT) == p['parent_sha256'], 'parent protocol changed')
    for k in ('paths','ordered_ids','label_hash','eval_label_hash','split_hash','loss',
              'initial_asset','training_code_sha256','environment','expected_runtime'):
        need(p[k] == old[k], 'frozen parent contract changed: '+k)
    for k in ('expected_args','initial_state_sha256','actual_pos_weight','loss_policy','expected_attempts'):
        need(p['arm'][k] == old['arm'][k], 'frozen parent arm changed: '+k)
    for name, digest in p['hc06_bindings'].items():
        need(sha256(root/name) == digest, 'HC06 binding changed: '+name)
    need((root/RUNNER).read_text(encoding='utf-8') == render_runner(), 'runner diff changed')
    if require_review:
        review = read(root/REVIEW)
        need(review.get('status') == 'READY' and review.get('manifest_sha256') == sha256(path), 'review required for exact manifest')
    return p, ids


def validate_arguments(args, root):
    p, ids = validate_protocol(args.protocol, root, True)
    for key, value in p['arm']['expected_args'].items():
        need(getattr(args, key, None) == value, 'frozen argument differs: '+key)
    for key, value in p['paths'].items():
        need(Path(getattr(args, key)).resolve() == (Path(root)/value).resolve(), 'frozen path differs: '+key)
    need(args.phase in ('pilot','full') and args.tag == TAG+('_PILOT' if args.phase == 'pilot' else ''), 'phase/tag mismatch')
    for key, value in {**base.THREAD_ENV,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}.items():
        need(os.environ.get(key) == value, 'registered environment required: '+key)
    return p, ids


def validate_pretraining(provenance, env, protocol, arm):
    base.validate_pretraining(provenance, env, protocol, arm)
    need(provenance['aggregation_policy'] == AGGREGATION_POLICY, 'patch aggregation not recorded')


def finish_pilot(out, protocol, provenance, timings, updates, skips, initial_seconds, eval_seconds, peak):
    need(updates == 32 and skips == 0 and len(timings) == 32, 'pilot update count')
    steady = sum(timings[8:])/24
    write_new(Path(out)/'receipt.json', dict(schema=SCHEMA, status='PASS', phase='pilot',
        protocol_sha256=provenance['protocol_sha256'], initial_state_sha256=provenance['initial_state_sha256'],
        initial_asset_sha256=WEIGHT_SHA, aggregation_policy=AGGREGATION_POLICY, position_policy=POSITION_POLICY,
        updates=updates, skips=skips, step_seconds=timings, steady_update_seconds=steady,
        projected_full_seconds=1.3*(steady*3448)+300, initial_seconds=initial_seconds,
        eval_seconds=eval_seconds, peak_gpu_allocated_bytes=peak, training_ids_only=True,
        eval_train_studies=4, validation_rows=0, metric_access='disabled',
        checkpoints_saved=False, predictions_saved=False, argv=[sys.executable,*sys.argv]))


def pilot_cost_gate(pilot, elapsed):
    need(pilot.get('schema') == SCHEMA and pilot.get('status') == 'PASS'
         and pilot.get('phase') == 'pilot' and pilot.get('updates') == 32 and pilot.get('skips') == 0
         and pilot.get('training_ids_only') is True and pilot.get('validation_rows') == 0
         and pilot.get('eval_train_studies') == 4 and pilot.get('aggregation_policy') == AGGREGATION_POLICY
         and pilot.get('position_policy') == POSITION_POLICY, 'pilot contract failed')
    times = pilot.get('step_seconds', [])
    need(len(times) == 32 and all(type(t) in (int,float) and math.isfinite(t) and t > 0 for t in times), 'invalid measured intervals')
    steady = sum(times[8:])/24; projected = 1.3*(steady*3448)+300
    need(pilot['steady_update_seconds'] == steady and pilot['projected_full_seconds'] == projected
         and projected <= 1800 and type(elapsed) in (int,float) and 0 < elapsed <= 180, 'pilot exceeds frozen cost')
    need(0 < pilot.get('peak_gpu_allocated_bytes', 0) <= 12*2**30, 'pilot peak exceeds12GiB')


def finish_full(out, protocol, provenance, trace, elapsed):
    out = Path(out)
    base.validate_completion(out, protocol, protocol['arm'])
    need(trace == protocol['expected_cls_training_trace_sha256'], 'actual training UID trace differs from historical CLS')
    with (out/'oof_best.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f); rows = list(reader); fields = reader.fieldnames
    ids = read(ROOT/protocol['ordered_ids']['path'])['val']
    need(fields == protocol['prediction_columns'] and [r[fields[0]] for r in rows] == ids, 'prediction UID/column contract')
    need(all(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1 for r in rows for k in fields[1:]), 'invalid probabilities')
    write_new(out/'trace.json', dict(schema=SCHEMA,training_trace_sha256=trace,epochs=8,updates=3448,skips=0))
    names = ('summary.json','provenance.json','history.json','trace.json','model_best.pt','oof_best.csv')
    write_new(out/'receipt.json', dict(schema=SCHEMA,status='PASS',phase='full',
        protocol_sha256=provenance['protocol_sha256'],aggregation_policy=AGGREGATION_POLICY,
        position_policy=POSITION_POLICY,initial_state_sha256=provenance['initial_state_sha256'],
        initial_asset_sha256=WEIGHT_SHA,updates=3448,skips=0,validation_rows=888,
        training_trace_sha256=trace,elapsed_seconds=elapsed,argv=[sys.executable,*sys.argv],
        outputs={n:dict(sha256=sha256(out/n),bytes=(out/n).stat().st_size) for n in names}))


def phase_command(protocol, path, phase):
    command = [sys.executable,str(ROOT/RUNNER),'--protocol',str(Path(path).resolve()),'--phase',phase,
               '--tag',TAG+('_PILOT' if phase == 'pilot' else '')]
    for key,value in {**protocol['arm']['expected_args'],**protocol['paths']}.items():
        option = '--'+key.replace('_','-')
        if isinstance(value,bool):
            if value: command.append(option)
        else: command.extend([option,str(ROOT/value) if key in protocol['paths'] else str(value)])
    return command


def validate_result(result, protocol, path, phase, command, out):
    need(result.get('schema') == SCHEMA and result.get('status') == 'PASS'
         and result.get('phase') == phase and result.get('protocol_sha256') == sha256(path)
         and result.get('argv') == command and result.get('aggregation_policy') == AGGREGATION_POLICY
         and result.get('position_policy') == POSITION_POLICY and result.get('initial_asset_sha256') == WEIGHT_SHA
         and result.get('initial_state_sha256') == protocol['arm']['initial_state_sha256'], 'phase identity mismatch')
    if phase == 'full':
        base.validate_completion(out, protocol, protocol['arm'])
        need(result.get('updates') == 3448 and result.get('skips') == 0 and result.get('validation_rows') == 888
             and result.get('training_trace_sha256') == protocol['expected_cls_training_trace_sha256'], 'full numerical contract')
        need(set(result.get('outputs',{})) == {'summary.json','provenance.json','history.json','trace.json','model_best.pt','oof_best.csv'}, 'missing output inventory')
        for name, entry in result['outputs'].items():
            need((out/name).stat().st_size == entry['bytes'] and sha256(out/name) == entry['sha256'], 'output inventory mismatch')


def run(path):
    p, _ = validate_protocol(path, require_review=True)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'CUDA hidden in launch environment')
    import psutil
    from tools.process_resource_receipt import supervise
    # Exclusive sentinel prevents a retry even after preflight/partial failure.
    sentinel = Path(path).with_name('execution.json')
    write_new(sentinel, dict(status='STARTING',retries=0,protocol_sha256=sha256(path)))
    started = time.monotonic(); failure = None; phases = []
    before = dict(os.environ)
    try:
        os.environ.update({**base.THREAD_ENV,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                           'PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
        for phase, limit in [('pilot',180),('full',1800)]:
            out = ROOT/'artifacts/experiments'/(TAG+('_PILOT' if phase == 'pilot' else ''))
            need(not out.exists() and not out.with_name(out.name+'_RESOURCE').exists(), 'fresh outputs required; retry forbidden')
            need(psutil.virtual_memory().available >= 8*2**30, 'host available RAM below8GiB')
            remaining = 1980-(time.monotonic()-started)
            need(remaining >= limit, 'insufficient aggregate budget for registered phase')
            command = phase_command(p,path,phase)
            resource = supervise(command,out.with_name(out.name+'_RESOURCE'),limit,ROOT)
            base.resource_gate(resource,command,limit)
            result = read(out/'receipt.json')
            validate_result(result,p,path,phase,command,out)
            if phase == 'pilot': pilot_cost_gate(result,resource['elapsed_seconds'])
            phases.append(dict(phase=phase,receipt_sha256=sha256(out/'receipt.json'),
                               resource_receipt_sha256=sha256(out.with_name(out.name+'_RESOURCE')/'receipt.json')))
        need(time.monotonic()-started <= 1980, 'aggregate cap exceeded including validation')
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        os.environ.clear();os.environ.update(before)
    record = dict(status='INVALID' if failure else 'VALIDATED_SUCCESS',valid_for_scoring=failure is None,
        protocol_sha256=sha256(path),failure=failure,elapsed_seconds=time.monotonic()-started,phases=phases)
    sentinel.write_text(json.dumps(record,indent=2),encoding='utf-8')
    write_new(Path(path).with_name('INVALID.json' if failure else 'VALIDATED_SUCCESS.json'),record)
    print(json.dumps(record));return 1 if failure else 0


def prepare(probe_path):
    out = ROOT/PROTOCOL;need(not out.exists(),'new protocol path required')
    old, ids = base.validate_protocol(ROOT/PARENT,ROOT,True)
    probe = read(probe_path)
    need(probe['status'] == 'PASS' and probe['historical_trace_exact'] and not probe['cuda_initialized']
         and probe['initial_state_sha256'] == old['arm']['initial_state_sha256'], 'CPU probe failed')
    for name,digest in probe['source_sha256'].items():need(sha256(ROOT/name) == digest,'probe source changed')
    runner = render_runner()
    with (ROOT/RUNNER).open('x',encoding='utf-8',newline='\n') as f:f.write(runner)
    p = copy.deepcopy(old)
    p.update(schema=SCHEMA,status='CPU_PREPARED_REQUIRES_EXACT_REVIEW',aggregation_policy=AGGREGATION_POLICY,
             parent_sha256=sha256(ROOT/PARENT),expected_cls_training_trace_sha256=probe['training_trace_sha256'])
    p['parent_cpu_probe'] = copy.deepcopy(old['cpu_probe'])
    p['cpu_probe'] = dict(path=Path(probe_path).resolve().relative_to(ROOT).as_posix(),sha256=sha256(probe_path))
    p['arm'].update(id='PATCH',tag=TAG)
    p['outputs'] = {k:f'artifacts/experiments/{TAG}/{Path(v).name}' for k,v in old['outputs'].items()}
    p['arm']['outputs'] = copy.deepcopy(p['outputs'])
    p['execution'].update(pilot_seconds=180,full_seconds=1800,total_seconds=1980,retries=0,
        projection='1.3*(mean(last24 of32 update seconds)*3448)+300<=1800',peak_gpu_allocated_cap_bytes=12*2**30,
        host_available_floor_bytes=8*2**30)
    p['execution'].pop('v1_elapsed_seconds',None)
    controls = ['artifacts/experiments/EXP086_DINOV2S_G1_V2/'+n for n in
                ('summary.json','provenance.json','trace.json','receipt.json','oof_best.csv')]
    controls += ['artifacts/experiments/EXP084_B_G1/'+n for n in ('summary.json','provenance.json','oof_best.csv')]
    with (ROOT/controls[4]).open(encoding='utf-8',newline='') as f:p['prediction_columns'] = next(csv.reader(f))
    names = [RUNNER,'tools/patch_mil.py','tools/patch_mil_probe.py','training/adapters/dinov2_patch_mil.py',
             'tests/test_patch_mil.py',str(Path(probe_path).resolve().relative_to(ROOT).as_posix()),
             '.local/hc06-patch-mil-v1/control_source_review.json',*controls]
    p['hc06_bindings'] = {name:sha256(ROOT/name) for name in names}
    p['evaluation'] = dict(primary='patch-B raw12macro K24',secondary='patch-CLS raw12macro K24',
        labels='original Dread float32 then rint',bootstrap='500 component-paired draws seed20261003',
        diagnostic='12 targets and scanner/mask groups; rare-positive uncertainty retained',
        selection='fixed epoch8; no Gold, intermediate scoring, ratio tuning, auto expansion or submission')
    p['checkpoint_loading'] = 'tools.patch_mil.load_patch_checkpoint only; vanilla loader reconstructs CLS and is incompatible'
    p['limitations'] += ['Head dropout RNG consumption changes; no identical stochastic model trajectory claim',
                         'Recorded historical UID trace equality checked for CLS only; B has source/provenance evidence',
                         'CPU wiring does not establish CUDA stability, VRAM, throughput, efficacy or serving compatibility']
    out.mkdir(parents=True)
    write_new(out/'manifest.json',p)
    (out/'runner.diff').write_text(''.join(difflib.unified_diff((ROOT/base.RUNNER).read_text(encoding='utf-8').splitlines(True),
        runner.splitlines(True),fromfile=base.RUNNER,tofile=RUNNER)),encoding='utf-8')
    write_new(out/'queue.json',[dict(tag=TAG+'_BOUNDED',command=[sys.executable,str(ROOT/'tools/patch_mil.py'),
        'run','--protocol',str(out/'manifest.json')],after='HC05_UNIQUE2764_SEED43',continue_on_failure=True)])
    write_new(out/'dispatch.json',dict(status='NOT_LAUNCHED_REVIEW_REQUIRED',owner='C',item='H-C-06',
        priority=9,preconditions=['Exact final_review READY for this manifest','H-C-05 completed or explicitly closed invalid',
        'All B/local queued reservations ended; recheck latest queues, claims, processes and GPU',
        'Claim GPU and launch through gpu_queue, never generated runner directly'],
        aggregate_seconds=1980,review_path=REVIEW,
        predecessor_failure='If C05 seed42 failed and seed43 was never created, do not wait on the absent tag: coordinator must record C05 closure and create a new queue version with a verified terminal predecessor before dispatch.'))
    validate_protocol(out/'manifest.json')
    print(json.dumps(dict(status=p['status'],manifest_sha256=sha256(out/'manifest.json'))))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('--probe',type=Path,required=True)
    q=sub.add_parser('run');q.add_argument('--protocol',type=Path,required=True)
    a=ap.parse_args()
    if a.action == 'prepare':prepare(a.probe);return 0
    return run(a.protocol)


if __name__ == '__main__':sys.exit(main())
