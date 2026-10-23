"""H204 fixed C4 cache consumer v1.0.3; reuse measured memory, never re-admit it.

Changelog: consume immutable historical memory evidence as a result, with a
distinct result review, instead of replaying its consumed execution admission.
The v102 scientific worker, supervisor and output/identity checks are unchanged.
No memory command, training, parent inference, submission or automatic retry.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb198_c4_gold58_v102 as old

VERSION = '1.0.3'
SELF = 'tools/hb204_c4_cache_v103.py'
TEST = 'tools/test_hb204_c4_cache_v103.py'
REL = Path('.local/scratch/slot-b/research-resume-1018/h204-prep/cache-v103')
PREP = REL.parent
TAG = 'H-B-204-C4-GOLD58-V103'
HIST = old.REL
REG_SHA = '82c79c6e05accc67cca1b64745f11bc5fc126906e4ae1d9715d9d02f37ec962c'
MEM_SHA = '99a7d7732e6aa026ab70830a766c19c03cba79d4922d93b2254b56bdd5668bc0'
SOURCE_SHA = '60fabfebc1fc838587e761285a68f1fca02567d356082d35bf176dfb2bee75fa'
ROSTER_SHA = '68ba75ccd31351e6d78d11a7ae83112ca5da1b48c7c77152ececd86bae5c0f40'
need, read, sha, ref, fresh = old.need, old.read, old.sha, old.ref, old.fresh


def command(root=ROOT):
    return [old.bridge.PYTHON, '-B', str(root / SELF), 'run']


def queue(root=ROOT):
    return [dict(tag=TAG, command=command(root), continue_on_failure=False)]


def historical(inputs):
    """Read historical output bindings, never call any old admission evaluator."""
    root = inputs.root
    reg = inputs.json(dict(path=(HIST/'registration.json').as_posix(), sha256=REG_SHA))
    need(reg['source_sha256'][old.SELF] == SOURCE_SHA, 'historical worker identity')
    inputs.pins(reg['source_sha256'])
    legacy = inputs.json(reg['legacy_card'])
    plan = inputs.json(reg['memory_plan'])
    card = inputs.json(reg['cache_card'])
    value = inputs.json(dict(path=(HIST/'memory-run/measurement.json').as_posix(), sha256=MEM_SHA))
    need(reg['candidate'] == legacy['candidate'] == value['candidate'] == old.gate.b.KEY and
         reg['kernel_sha256'] == value['kernel_sha256'] == old.KERNEL and
         value['schema'] == 'h198-memory-evidence/v1' and value['status'] == 'PASS_MEASURED' and
         value['registration_sha256'] == REG_SHA and value['source_sha256'] == SOURCE_SHA and
         value['plan_sha256'] == reg['memory_plan']['sha256'] and
         value['card_sha256'] == reg['cache_card']['sha256'] and
         value['renderer_sha256'] == legacy['renderer_sha256'] and value['K'] == 36 and
         value['tta'] == 'plane-aware-coronal-axial-flip' and value['actual_full_shape_both_views'] is True,
         'historical scientific memory identity')
    need(legacy['workload'] == 290 and legacy['expected_shape'] == [5,58,12] and
         len(legacy['models']) == 5 and len(legacy['corpus']['ids']) == 58 and
         len(set(legacy['corpus']['ids'])) == 58 and len(legacy['corpus']['targets']) == 12 and
         old.jr.digest(legacy['corpus']['ids']) == ROSTER_SHA and
         [x['fold'] for x in legacy['models']] == list(range(5)) and
         all(x['seed'] == 42 and x['sha256'] == x['source_sha256'] for x in legacy['models']),
         'fixed C4 five/same58 source-deploy roster')
    need(value['model_sha256'] == [x['sha256'] for x in legacy['models']] and
         plan['model_sha256'] == value['model_sha256'] and
         plan['roster_sha256'] == ROSTER_SHA and plan['targets'] == legacy['corpus']['targets'] and
         plan['representative_index'] == 0 and plan['representative_uid'] == legacy['corpus']['ids'][0] and
         plan['input_shape'] == [36,3,224,224] and plan['serial_models'] == 5 and plan['views_per_model'] == 2 and
         plan['source_sha256'] == reg['source_sha256'] and plan['worker']['sha256'] == SOURCE_SHA and
         card['legacy_card'] == reg['legacy_card'] and card['worker'] == plan['worker'],
         'original plan/card match scientific workload')
    need(len(value['raw_receipts']) == len(value['peak_reserved_bytes']) == 5 and value['physical_gpu'] == 0,
         'five genuine measurement receipts')
    old.memory_ok(value, value['peak_reserved_bytes'])
    for fold, item in enumerate(value['raw_receipts']):
        row = inputs.json(item)
        need(item['path'] == (HIST/'memory-run'/f'model-{fold}.json').as_posix() and
             row['schema'] == 'h198-memory-model/v1' and row['fold'] == fold and
             row['model_sha256'] == value['model_sha256'][fold] and
             row['plan_sha256'] == value['plan_sha256'] and row['source_sha256'] == SOURCE_SHA and
             row['registration_sha256'] == REG_SHA and row['representative_index'] == 0 and
             row['views'] == 2 and row['input_shape'] == [36,3,224,224] and
             row['peak_reserved_bytes'] == value['peak_reserved_bytes'][fold] and
             0 < row['peak_allocated_bytes'] <= row['peak_reserved_bytes'] and
             row['synchronized'] is True and row['model_deleted'] is True and
             math.isfinite(row['seconds']) and row['seconds'] > 0, 'raw measured model identity')
        env = row['environment']
        need(env['os'] == 'win32' and env['physical_gpu'] == 0 and env['logical_device'] == 'cuda:0' and
             '4070 Ti SUPER' in env['gpu'] and env['cudnn_benchmark'] is False and env['cudnn_deterministic'] is True and
             env['precision'] == 'original FP16 forward; FP32 sigmoid and two-probability mean' and
             all(env.get(k) for k in ('python','torch','cuda','cudnn','timm')) and
             all(row['desktop'][k] == value[k] for k in ('desktop_bytes','total_bytes','device_name')),
             'historical software/device/headroom provenance')
    monitor = inputs.json(value['monitor']); worker = inputs.json(value['worker'])
    inputs.file(value['raw_log'])  # Hash only; do not print or parse log contents.
    process = inputs.json(monitor['process'])
    need(monitor['status'] == 'PASS' and monitor['worker_sha256'] == value['worker']['sha256'] and
         all(x['child_exit_code'] == 0 and x['owned_cleanup_complete'] is True for x in (monitor,process)) and
         not process['survivors'] and process['source_sha256'] == SOURCE_SHA,
         'historical successful child and owned cleanup')
    need(worker['state'] == 'MEASUREMENT_COMPLETE' and worker['mode'] == 'memory' and
         worker['workload'] == worker['models'] == 5 and worker['studies'] == 1 and worker['failures'] == 0 and
         worker['npz_sha256'] is None and worker['shape'] is None and worker['kernel_sha256'] == old.KERNEL and
         worker['roster_sha256'] == ROSTER_SHA and worker['targets'] == legacy['corpus']['targets'] and
         all(worker[k] == value[k] for k in ('source_sha256','registration_sha256','plan_sha256','card_sha256',
              'model_sha256','peak_reserved_bytes','raw_receipts')), 'complete measurement, not cached probabilities')
    # Original admission bytes are historical provenance, NOT reusable permission.
    admission = inputs.json(value['admission'])
    ticket = inputs.json(value['launch']); desktop = inputs.json(value['desktop']); job = inputs.json(value['bound_job'])
    need(value['admission']['path'] == (HIST/'memory-admission.json').as_posix() and
         worker['admission_sha256'] == value['admission']['sha256'] and
         admission['candidate'] == old.gate.b.KEY and admission['kernel_sha256'] == old.KERNEL and
         admission['registration'] == dict(path=(HIST/'registration.json').as_posix(),sha256=REG_SHA) and
         ticket['mode'] == 'memory' and ticket['admission'] == value['admission'] and
         ticket['registration_sha256'] == REG_SHA and
         all(desktop[k] == value[k] for k in ('desktop_bytes','total_bytes','device_name')) and
         job['schema'] == 'gpu-job/v2' and job['tag'] == old.TAGS['memory'] and job['phase'] == 'running' and
         job['command'] == old.command('memory',root) and
         job['command_sha256'] == old.jr.command_digest(old.command('memory',root),root) and
         job['cwd'] == str(root) and job['child_identity'] == ticket['parent'], 'historical launch provenance only')
    # No pins(admission['consumed_sha256']), memory_admission, authorize, or gate replay.
    return reg, legacy, value


def source_map(reg, root=ROOT):
    names = set(reg['source_sha256']) | set(old.gate.SOURCE_PATHS) | {
        SELF, TEST, 'tools/exp169_partner_bundle_release.py'}
    return {p: sha(root/p) for p in sorted(names)}


def register(root=ROOT):
    inputs = old.gate.Inputs(root)
    hist, legacy, _ = historical(inputs)
    sources = source_map(hist,root); inputs.pins(sources)
    inputs.file(dict(path=(old.bridge.OUT.relative_to(ROOT)/'candidate-runtime/exp169_partner_bundle_release.py').as_posix(),
                     sha256=legacy['renderer_sha256']))
    inputs.file(dict(path=(old.bridge.build.OUT.relative_to(ROOT)/'kernel/release.ipynb').as_posix(),sha256=old.KERNEL))
    for name in ('INVENTORY.json','H204-PREP-REVIEW.md'):
        inputs.file(ref(root/PREP/name,root))
    value = dict(schema='h204-c4-cache-registration/v1',version=VERSION,workspace=str(root),
        candidate=old.gate.b.KEY,kernel_sha256=old.KERNEL,builder=old.BUILDER,tag=TAG,
        source_sha256=sources,metadata_sha256=dict(inputs.consumed),
        legacy_card=hist['legacy_card'],cache_card=hist['cache_card'],memory_plan=hist['memory_plan'],
        historical_memory=dict(path=(HIST/'memory-run/measurement.json').as_posix(),sha256=MEM_SHA),
        queue=queue(root),workload=290,expected_shape=[5,58,12],fixture_only=root != ROOT,
        execution_authorized=False,qualified_for_submission=False)
    inputs.recheck(); fresh(root/REL/'registration.json',value); fresh(root/REL/'cache-queue.json',queue(root))
    return value


def validate(root=ROOT, *, execution=False):
    inputs = old.gate.Inputs(root); reg = inputs.json(ref(root/REL/'registration.json',root))
    need(reg['schema'] == 'h204-c4-cache-registration/v1' and reg['version'] == VERSION and
         reg['workspace'] == str(root) and reg['builder'] == old.BUILDER and
         reg['candidate'] == old.gate.b.KEY and reg['kernel_sha256'] == old.KERNEL and reg['tag'] == TAG and
         reg['workload'] == 290 and reg['expected_shape'] == [5,58,12] and
         reg['fixture_only'] is (root != ROOT) and reg['execution_authorized'] is False,
         'fixed fresh cache registration identity')
    inputs.pins(reg['metadata_sha256']); inputs.pins(reg['source_sha256'])
    hist, legacy, _ = historical(inputs)
    need(reg['source_sha256'] == source_map(hist,root) and reg['legacy_card'] == hist['legacy_card'] and
         reg['cache_card'] == hist['cache_card'] and reg['memory_plan'] == hist['memory_plan'] and
         reg['queue'] == queue(root) == inputs.json(ref(root/REL/'cache-queue.json',root)), 'exact source/queue/card')
    from tools.gpu_queue import validate_plan
    validate_plan(reg['queue'])
    if execution:
        need(root == ROOT and reg['fixture_only'] is False, 'fixtures cannot execute')
        for model in legacy['models']:
            inputs.file(dict(path=model['source'].replace('\\','/'),sha256=model['sha256']))
        for name, digest in legacy['corpus']['metadata'].items():
            inputs.file(dict(path=(old.bridge.CORPUS.relative_to(ROOT)/name).as_posix(),sha256=digest))
    inputs.recheck()
    return reg,legacy,inputs


def memory_review(inputs):
    report = inputs.root/PREP/'H204-MEMORY-REVIEW.md'
    rr = ref(report,inputs.root)
    actual = old.gate.archive(inputs,ref(str(report)+'.receipt.json',inputs.root),rr)
    text = old.gate.verdict(inputs.file(rr).read_bytes())
    need(actual['agent_id'] != old.BUILDER and
         re.findall(r'^VERDICT:\s*(PASS|FAIL)\s*$',text,re.M) == ['PASS'] and
         MEM_SHA in text and REG_SHA in text, 'actual distinct historical memory result PASS and subjects')
    return rr


def authorize(root=ROOT):
    reg,_,inputs = validate(root)
    memory = memory_review(inputs)
    report = root/REL/'PRELAUNCH-REVIEW.md'
    old.review(report,'CACHE_REUSE',inputs,sha(root/REL/'registration.json'),MEM_SHA)
    tests_ref = ref(root/REL/'CPU-TEST-RECEIPT.json',root); tests = inputs.json(tests_ref)
    need(tests['actual_run'] is True and tests['failed'] == tests['exit_code'] == 0 and
         tests['source_sha256'] == reg['source_sha256'], 'actual current CPU tests')
    inputs.file(tests['log'])
    resource_ref = ref(root/REL/'cache-claim.json',root); resource = inputs.json(resource_ref)
    runtime = runtime_module()
    runtime.claim(resource,'cache')
    need(resource['registration_sha256'] == sha(root/REL/'registration.json'), 'claim exact current registration')
    inputs.recheck()
    return dict(schema='h204-c4-cache-admission/v1',state='READY_FOR_CACHE_ONLY',candidate=old.gate.b.KEY,
        kernel_sha256=old.KERNEL,tag=TAG,registration=ref(root/REL/'registration.json',root),
        memory_result_review=memory,resource_claim=resource_ref,consumed_sha256=dict(inputs.consumed),
        fixture_only=root != ROOT,qualified_for_submission=False)


def admission(root=ROOT):
    saved = read(root/REL/'CACHE-ADMISSION.json')
    need(saved == authorize(root), 'fresh cache admission changed')
    return saved


def runtime_module():
    """Private module namespace; no shared module/source patching or science copy."""
    spec = importlib.util.spec_from_file_location('_h204_c4_v102_runtime',ROOT/old.SELF)
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    runtime.REL = REL; runtime.SELF = SELF; runtime.TAGS = {'cache':TAG}
    runtime.validate = lambda: validate(execution=True)
    runtime.cache_admission = admission
    runtime.measurement_result = lambda: (historical(old.gate.Inputs(ROOT))[2],None)
    return runtime


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('register','validate','prepare','admit-dry-run','admit','run','_cache'))
    a = p.parse_args(argv)
    try:
        if a.command == 'register': register(); state='REGISTERED_NOT_ADMITTED'
        elif a.command in ('validate','prepare'):
            validate(); state='STATIC_VALID_NOT_ADMITTED'
        elif a.command in ('admit','admit-dry-run'):
            value=authorize()
            if a.command == 'admit':
                need(not (ROOT/REL/'cache-run').exists() and not (ROOT/'artifacts/logs'/TAG).exists(),
                     'fresh tag/output required; never auto retry')
                fresh(ROOT/REL/'CACHE-ADMISSION.json',value)
            state='CACHE_ADMISSION_VALID_NO_LAUNCH'
        else:
            # Existing run/worker enforce source bytes, physical ownership,
            # current measured desktop headroom, child identity and no overwrite.
            runtime=runtime_module()
            value=runtime.run('cache') if a.command == 'run' else runtime.worker('cache')
            state=value.get('status',value.get('state'))
        print(json.dumps(dict(status=state,qualified_for_submission=False)))
        return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='HOLD',error=str(exc),execution_authorized=False)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
