"""C4-only local memory/cache integration v1.0.2. Never training or submission.

Additive consumer of strict_variant_gate_c4_v102.verify_receipt. Static commands
never import torch. Fixed production paths; isolated parser fixtures cannot run.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import strict_variant_gate_c4_v102 as gate
from tools import hb198_c4_bridge_v1 as bridge
from tools import hb145_output_io as io
from tools import job_runtime as jr
from tools.hb155_watchdog_v1 import collect_owned, terminate_owned
from tools.c85r7_v150_safety import ThroughputGuard

VERSION = '1.0.2'
REL = Path('.local/scratch/slot-b/batch5-1018/c4-serving')
SELF = 'tools/hb198_c4_gold58_v102.py'
TEST = 'tools/test_hb198_c4_serving_v102.py'
BUILD = Path('.local/scratch/slot-b/batch5-1018/c4-bridge/fix1/build-admission.json')
BUILD_SHA = 'daeaae995dd2932ad033016d4acdd085aea1c7ba7ee313d199e8319b55e9f55b'
KERNEL = 'd86ca0e93e11773ad6119bb9c84ea632130723d899a5f4ef5d5faa4029fc28d9'
BUILDER = '/root/candidate_scout'
TAGS = {'memory': 'H-B-198-C4-MEMORY-V102', 'cache': 'H-B-198-C4-GOLD58'}
PREFIX = 'C4-SERVING-ATTESTATION: '
GIB = 2**30
SOURCES = (SELF, TEST, 'tools/strict_variant_gate_c4_v102.py',
           'tools/native_codex_dispatch_record_v1.py', 'tools/hb145_output_io.py',
           'tools/hb198_c4_bridge_v1.py', 'tools/hb198_c4_gold58_v1.py',
           'tools/gpu_queue.py', 'tools/gpu_job.py', 'tools/job_runtime.py',
           'tools/hb155_watchdog_v1.py', 'tools/hb150_io_v1.py',
           'tools/c85r7_v150_safety.py')
need, sha = gate.need, gate.sha


def read(path):
    return gate.native.parse(Path(path).read_bytes())


def ref(path, root=ROOT):
    return gate.ref(path, root)


def fresh(path, value):
    """Create-only authoritative publication; preserve partial attempts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), 'immutable output already exists: ' + path.name)
    io.atomic_json(path, value)


def diagnostic(path, value):
    try:
        io.atomic_json(path, value)
        return True
    except OSError:
        return False


def fixture_scope(root, anchor):
    root = Path(root).resolve()
    fixture = root != ROOT
    need((not fixture and anchor is None) or
         (fixture and anchor is not None and root.is_relative_to(ROOT / '.local/scratch')),
         'fixed project or isolated explicit TEST-ONLY fixture required')
    return root, fixture


def checked_gate(item, phase, inputs, anchor=None):
    """Production ALWAYS uses verify_receipt; fixtures replay the same evaluator."""
    inputs.file(item)
    if inputs.root == ROOT:
        value = gate.verify_receipt(item, phase=phase, slug=gate.b.SLUG)
    else:
        fixture_scope(inputs.root, anchor)
        saved = inputs.json(item)
        need(saved['fixture_only'] is True and saved['phase'] == phase, 'fixture gate identity')
        value = gate.evaluate(phase, saved['evidence'], saved['review'], root=inputs.root, fixture_frozen=anchor)
        need(value == saved, 'fixture receipt replay differs')
    need(value['kernel_sha256'] == KERNEL and value['phase'] == phase, 'scientific gate identity')
    inputs.pins(value['consumed_sha256'])
    return value


def command(mode, root=ROOT):
    need(mode in TAGS, 'unknown mode')
    return [bridge.PYTHON, '-B', str(root / SELF), 'measure' if mode == 'memory' else 'run']


def queue(mode, root=ROOT):
    return [dict(tag=TAGS[mode], command=command(mode, root), continue_on_failure=False)]


def register(root=ROOT, anchor=None, build_ref=None):
    root, fixture = fixture_scope(root, anchor)
    inputs = gate.Inputs(root)
    item = build_ref or dict(path=BUILD.as_posix(), sha256=BUILD_SHA)
    checked_gate(item, 'BUILD_PASS', inputs, anchor)
    frozen = inputs.json(anchor or dict(path=gate.FROZEN, sha256=gate.FROZEN_SHA))
    legacy_ref = frozen['roles']['legacy_card']; legacy = inputs.json(legacy_ref)
    need(legacy['tag'] == TAGS['cache'] and legacy['workload'] == 290 and
         legacy['expected_shape'] == [5, 58, 12], 'exact original workload')
    sources = {p: sha(root / p) for p in SOURCES}
    sources.update(legacy['sources'])
    inputs.pins(sources)
    if not fixture:
        need(not any((root / 'artifacts/logs' / tag).exists() for tag in TAGS.values()),
             'tag already consumed; new versioned contract review required')
    work = root / REL
    for mode in TAGS:
        fresh(work / (mode + '-queue.json'), queue(mode, root))
    card = dict(schema='h198-cache-integration/v2', candidate=gate.b.KEY, kernel_sha256=KERNEL,
        legacy_card=legacy_ref, gate_sha256=sources['tools/strict_variant_gate_c4_v102.py'],
        gate_module='tools.strict_variant_gate_c4_v102', gate_phase='CACHE_LAUNCH_PASS',
        late_recheck=True, science_unchanged=True, worker=ref(root / SELF, root),
        queue=ref(work / 'cache-queue.json', root))
    fresh(work / 'cache-integration.json', card)
    plan = dict(schema='h198-memory-plan/v1', candidate=gate.b.KEY, kernel_sha256=KERNEL,
        build_admission=item, cache_card=ref(work / 'cache-integration.json', root),
        queue=ref(work / 'memory-queue.json', root), worker=ref(root / SELF, root),
        selection='first exact original same58 roster entry, fixed before results',
        representative_index=0, representative_uid=legacy['corpus']['ids'][0],
        roster_sha256=jr.digest(legacy['corpus']['ids']), targets=legacy['corpus']['targets'],
        model_sha256=[e['sha256'] for e in legacy['models']], renderer_sha256=legacy['renderer_sha256'],
        K=36, input_shape=[36, 3, 224, 224], tta='plane-aware-coronal-axial-flip',
        precision='original FP16 forward; FP32 sigmoid and two-probability mean',
        serial_models=5, views_per_model=2, full_cache_permitted=False,
        physical_gpu=0, logical_device='cuda:0', device='NVIDIA GeForce RTX 4070 Ti SUPER',
        peak_limit_bytes=12*GIB, margin_bytes=2*GIB, source_sha256=sources,
        fixture_only=fixture)
    fresh(work / 'memory-plan.json', plan)
    inputs.recheck()
    value = dict(schema='h198-serving-registration/v1', version=VERSION, candidate=gate.b.KEY,
        kernel_sha256=KERNEL, workspace=str(root), fixture_only=fixture, build_admission=item,
        frozen=anchor or dict(path=gate.FROZEN, sha256=gate.FROZEN_SHA), legacy_card=legacy_ref,
        source_sha256=sources, inherited_sha256=inputs.consumed,
        memory_plan=ref(work / 'memory-plan.json', root), cache_card=ref(work / 'cache-integration.json', root),
        queues={m:ref(work / (m+'-queue.json'), root) for m in TAGS},
        qualified_for_submission=False, execution_authorized=False)
    fresh(work / 'registration.json', value)
    return value


def validate(root=ROOT, anchor=None):
    root, fixture = fixture_scope(root, anchor); work = root / REL
    inputs = gate.Inputs(root); reg = inputs.json(ref(work / 'registration.json', root))
    need(reg['schema'] == 'h198-serving-registration/v1' and reg['version'] == VERSION and
         reg['workspace'] == str(root) and reg['fixture_only'] is fixture and
         reg['candidate'] == gate.b.KEY and reg['kernel_sha256'] == KERNEL, 'registration identity')
    need(set(SOURCES) <= set(reg['source_sha256']), 'missing serving source')
    inputs.pins(reg['inherited_sha256']); inputs.pins(reg['source_sha256'])
    checked_gate(reg['build_admission'], 'BUILD_PASS', inputs, anchor)
    frozen = inputs.json(reg['frozen']); legacy = inputs.json(reg['legacy_card'])
    need(reg['legacy_card'] == frozen['roles']['legacy_card'], 'immutable legacy card')
    card = inputs.json(reg['cache_card']); plan = inputs.json(reg['memory_plan'])
    need(card['schema'] == 'h198-cache-integration/v2' and card['candidate'] == gate.b.KEY and
         card['kernel_sha256'] == KERNEL and card['legacy_card'] == reg['legacy_card'] and
         card['worker'] == ref(root / SELF, root) and card['gate_module'] == 'tools.strict_variant_gate_c4_v102' and
         card['gate_sha256'] == reg['source_sha256']['tools/strict_variant_gate_c4_v102.py'] and
         card['gate_phase'] == 'CACHE_LAUNCH_PASS' and card['science_unchanged'] is True and card['late_recheck'] is True,
         'exact current integration card')
    for mode, item in reg['queues'].items():
        need(inputs.json(item) == queue(mode, root), 'exact queue identity')
    need(set(reg['queues']) == set(TAGS) and card['queue'] == reg['queues']['cache'], 'both fixed queues')
    need(plan['schema'] == 'h198-memory-plan/v1' and plan['candidate'] == gate.b.KEY and
         plan['kernel_sha256'] == KERNEL and plan['fixture_only'] is fixture and
         plan['build_admission'] == reg['build_admission'] and plan['cache_card'] == reg['cache_card'] and
         plan['worker'] == card['worker'] and plan['queue'] == reg['queues']['memory'] and
         plan['source_sha256'] == reg['source_sha256'] and plan['representative_index'] == 0 and
         plan['representative_uid'] == legacy['corpus']['ids'][0] and
         plan['roster_sha256'] == jr.digest(legacy['corpus']['ids']) and plan['targets'] == legacy['corpus']['targets'] and
         plan['model_sha256'] == [e['sha256'] for e in legacy['models']] and
         plan['renderer_sha256'] == legacy['renderer_sha256'] and plan['K'] == 36 and
         plan['input_shape'] == [36,3,224,224] and plan['views_per_model'] == 2 and plan['serial_models'] == 5 and
         plan['full_cache_permitted'] is False and plan['physical_gpu'] == 0 and plan['logical_device'] == 'cuda:0' and
         plan['peak_limit_bytes'] == 12*GIB and plan['margin_bytes'] == 2*GIB,
         'unchanged full-shape first measurement plan')
    need(len(legacy['models']) == 5 and len(set(legacy['corpus']['ids'])) == 58 and
         len(legacy['corpus']['targets']) == 12, 'original five/same58/targets')
    for model in legacy['models']:
        inputs.file(dict(path=model['source'].replace('\\','/'), sha256=model['sha256']))
    inputs.recheck()
    return reg, legacy, inputs


def prepare(root=ROOT, anchor=None):
    reg, _, _ = validate(root, anchor); work = root / REL
    value = dict(schema='h198-serving-prepare/v1', registration=ref(work/'registration.json',root),
        build_review=(REL/'BUILD-REVIEW.md').as_posix(),
        memory_launch_review=(REL/'MEMORY-LAUNCH-REVIEW.md').as_posix(),
        memory_claim=(REL/'memory-claim.json').as_posix(),
        memory_result_review=(REL/'MEMORY-RESULT-REVIEW.md').as_posix(),
        fixture_only=reg['fixture_only'], executor='B', execution_authorized=False)
    fresh(work/'prepare.json',value)
    return value


def attestation(phase, reg_sha, subject_sha, claim_sha=None, reviewer='ACTUAL_DISTINCT_NATIVE_AGENT'):
    return dict(schema='h198-serving-review/v1', phase=phase, candidate=gate.b.KEY,
        kernel_sha256=KERNEL, builder=BUILDER, reviewer=reviewer, verdict='PASS',
        registration_sha256=reg_sha, subject_sha256=subject_sha, resource_claim_sha256=claim_sha)


def review(path, phase, inputs, reg_sha, subject_sha, claim_sha=None):
    report = ref(path, inputs.root); receipt = ref(str(path)+'.receipt.json', inputs.root)
    actual = gate.archive(inputs, receipt, report)
    text = gate.verdict(inputs.file(report).read_bytes())
    need(re.findall(r'^[ \t]*VERDICT[ \t]*:.*$',text,re.M)==['VERDICT: PASS'],'ambiguous review verdict')
    lines = [x for x in text.splitlines() if x.lstrip().startswith('C4-SERVING-ATTESTATION')]
    need(len(lines)==1 and lines[0].startswith(PREFIX), 'one exact serving attestation')
    value = gate.native.parse(lines[0][len(PREFIX):].encode())
    need(actual['agent_id'] != BUILDER and value == attestation(phase,reg_sha,subject_sha,claim_sha,actual['agent_id']),
         'actual independent serving phase/identity/hash review')
    need(re.findall(r'^Reviewer-Lane: (\S+)$',text,re.M)==[actual['lane']], 'actual review lane')
    completed=dt.datetime.fromisoformat(actual['completed']); recorded=dt.datetime.fromisoformat(actual['recorded_at'])
    need(completed.utcoffset() is not None and recorded.utcoffset() is not None and
         completed<=recorded<=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=5), 'native chronology')
    inputs.recheck()
    return actual


def claim(value, mode):
    need(value['schema']=='h198-serving-resource/v1' and value['owner']=='B' and value['state']=='CLAIMED' and
         value['candidate']==gate.b.KEY and value['tag']==TAGS[mode] and value['exclusive'] is True and
         value['physical_gpu']==0 and value['logical_device']=='cuda:0' and
         value['device_name']=='NVIDIA GeForce RTX 4070 Ti SUPER' and
         isinstance(value['reservation'],str) and len(value['reservation'])>=8 and
         isinstance(value['shared_evidence'],str) and value['shared_evidence'], 'fresh real exclusive B claim')


def authorize_memory(root=ROOT, anchor=None):
    reg, _, inputs=validate(root,anchor); work=root/REL
    prep=inputs.json(ref(work/'prepare.json',root)); reg_ref=ref(work/'registration.json',root)
    need(prep==dict(schema='h198-serving-prepare/v1',registration=reg_ref,
        build_review=(REL/'BUILD-REVIEW.md').as_posix(),memory_launch_review=(REL/'MEMORY-LAUNCH-REVIEW.md').as_posix(),
        memory_claim=(REL/'memory-claim.json').as_posix(),memory_result_review=(REL/'MEMORY-RESULT-REVIEW.md').as_posix(),
        fixture_only=reg['fixture_only'],executor='B',execution_authorized=False),'fixed prepared boundary')
    br=review(work/'BUILD-REVIEW.md','BUILD',inputs,reg_ref['sha256'],reg_ref['sha256'])
    resource=inputs.json(ref(work/'memory-claim.json',root));claim(resource,'memory')
    lr=review(work/'MEMORY-LAUNCH-REVIEW.md','MEMORY_LAUNCH',inputs,reg_ref['sha256'],
              sha(work/'prepare.json'),sha(work/'memory-claim.json'))
    need(dt.datetime.fromisoformat(lr['completed'])>=dt.datetime.fromisoformat(br['completed']), 'launch before build review')
    inputs.recheck()
    return dict(schema='h198-memory-admission/v1',state='READY_FOR_MEMORY_ONLY',fixture_only=reg['fixture_only'],
        registration=reg_ref,prepared=ref(work/'prepare.json',root),consumed_sha256=dict(inputs.consumed),
        candidate=gate.b.KEY,kernel_sha256=KERNEL,tag=TAGS['memory'],qualified_for_submission=False)


def admit_memory(dry=True,root=ROOT,anchor=None):
    value=authorize_memory(root,anchor)
    if not dry:fresh(root/REL/'memory-admission.json',value)
    return value


def memory_admission(root=ROOT,anchor=None):
    root,fixture=fixture_scope(root,anchor)
    saved=read(root/REL/'memory-admission.json')
    need(saved['fixture_only'] is fixture,'fixture never executes in production')
    inputs=gate.Inputs(root);inputs.pins(saved['consumed_sha256']);inputs.recheck()
    need(authorize_memory(root,anchor)==saved,'memory admission replay differs')
    return saved


def memory_ok(desktop, peaks):
    total=desktop['total_bytes'];used=desktop['desktop_bytes']
    need('4070 Ti SUPER' in desktop['device_name'] and all(type(x) in (int,float) and math.isfinite(x) for x in [used,total,*peaks]) and
         0<=used<total and 16000*2**20<=total<=16400*2**20 and
         all(0<=x<=12*GIB for x in peaks) and used+max(peaks)+2*GIB<=total,'measured memory/headroom fail')


def measurement_result(root=ROOT,anchor=None,require_review=True):
    reg,legacy,inputs=validate(root,anchor);work=root/REL;run=work/'memory-run'
    value=inputs.json(ref(run/'measurement.json',root))
    need(value['schema']=='h198-memory-evidence/v1' and value['status']=='PASS_MEASURED' and
         value['registration_sha256']==sha(work/'registration.json') and value['plan_sha256']==reg['memory_plan']['sha256'] and
         value['source_sha256']==reg['source_sha256'][SELF] and value['card_sha256']==reg['cache_card']['sha256'] and
         value['kernel_sha256']==KERNEL and value['renderer_sha256']==legacy['renderer_sha256'] and value['K']==36 and
         value['tta']=='plane-aware-coronal-axial-flip' and value['actual_full_shape_both_views'] is True and
         value['model_sha256']==[e['sha256'] for e in legacy['models']] and value['physical_gpu']==0,
         'actual memory evidence identity')
    need(len(value['raw_receipts'])==len(value['peak_reserved_bytes'])==5,'five serial full-shape receipts')
    memory_ok(value,value['peak_reserved_bytes'])
    for f,item in enumerate(value['raw_receipts']):
        row=inputs.json(item)
        need(row['schema']=='h198-memory-model/v1' and row['fold']==f and row['model_sha256']==value['model_sha256'][f] and
             row['plan_sha256']==value['plan_sha256'] and row['source_sha256']==value['source_sha256'] and
             row['representative_index']==0 and row['views']==2 and row['input_shape']==[36,3,224,224] and
             row['peak_reserved_bytes']==value['peak_reserved_bytes'][f] and
             0<row['peak_allocated_bytes']<=row['peak_reserved_bytes'] and row['synchronized'] is True and
             math.isfinite(row['seconds']) and row['seconds']>0 and row['model_deleted'] is True,
             'raw full-shape measured identity/cleanup')
        env=row['environment']
        need(env['os']=='win32' and env['physical_gpu']==0 and env['logical_device']=='cuda:0' and
             '4070 Ti SUPER' in env['gpu'] and env['precision']=='original FP16 forward; FP32 sigmoid and two-probability mean' and
             all(env.get(k) for k in ('python','torch','cuda','cudnn','timm')) and
             row['registration_sha256']==value['registration_sha256'] and
             all(row['desktop'][k]==value[k] for k in ('desktop_bytes','total_bytes','device_name')),
             'real measured hardware/software/desktop provenance')
    mon=inputs.json(value['monitor']);worker_value=inputs.json(value['worker']);inputs.file(value['raw_log'])
    need(mon['status']=='PASS' and mon['owned_cleanup_complete'] is True and mon['child_exit_code']==0 and
         mon['worker_sha256']==value['worker']['sha256'],'measurement actual monitor/worker')
    process=inputs.json(mon['process'])
    need(process['child_exit_code']==0 and process['owned_cleanup_complete'] is True and not process['survivors'] and
         process['source_sha256']==value['source_sha256'],'actual owned measurement process')
    need(worker_value['state']=='MEASUREMENT_COMPLETE' and worker_value['mode']=='memory' and worker_value['failures']==0 and
         worker_value['workload']==5 and worker_value['models']==5 and worker_value['studies']==1 and
         worker_value['npz_sha256'] is None and worker_value['kernel_sha256']==KERNEL and
         all(worker_value[k]==value[k] for k in ('source_sha256','registration_sha256','plan_sha256',
              'card_sha256','model_sha256','peak_reserved_bytes','raw_receipts')),
         'complete actual measurement worker and current pins')
    admission=inputs.json(value['admission']);memory_admission(root,anchor);inputs.pins(admission['consumed_sha256'])
    need(value['admission']==ref(work/'memory-admission.json',root) and
         worker_value['admission_sha256']==value['admission']['sha256'],'actual measured admission binding')
    ticket=inputs.json(value['launch']);desktop=inputs.json(value['desktop']);job=inputs.json(value['bound_job'])
    need(ticket['mode']=='memory' and ticket['admission']==value['admission'] and
         ticket['registration_sha256']==value['registration_sha256'] and
         all(desktop[k]==value[k] for k in ('desktop_bytes','total_bytes','device_name')),'measurement launch/desktop provenance')
    need(job['schema']=='gpu-job/v2' and job['tag']==TAGS['memory'] and job['phase']=='running' and
         job['command']==command('memory',root) and job['command_sha256']==jr.command_digest(command('memory',root),root) and
         job['cwd']==str(root) and job['child_identity']==ticket['parent'], 'measured exact bound job snapshot')
    if require_review:
        review(work/'MEMORY-RESULT-REVIEW.md','MEMORY_RESULT',inputs,sha(work/'registration.json'),sha(run/'measurement.json'))
    inputs.recheck()
    return value,inputs


def cache_admission(root=ROOT,anchor=None):
    reg,legacy,inputs=validate(root,anchor);work=root/REL
    review(work/'BUILD-REVIEW.md','BUILD',inputs,sha(work/'registration.json'),sha(work/'registration.json'))
    memory,mi=measurement_result(root,anchor)
    inputs.pins(mi.consumed)
    saved=checked_gate(ref(work/'CACHE-ADMISSION.json',root),'CACHE_LAUNCH_PASS',inputs,anchor)
    e=inputs.json(saved['evidence']);refs=e['artifacts']
    need(refs['launch_card']==reg['cache_card'] and refs['memory']==ref(work/'memory-run/measurement.json',root) and
         all(e['sources'].get(p)==h for p,h in reg['source_sha256'].items()),'exact worker/card/measurement source closure')
    resource=inputs.json(refs['resource_claim']);claim(resource,'cache')
    inputs.recheck()
    return saved


def prepare_cache(tests):
    reg,_,inputs=validate();work=ROOT/REL
    measurement,mi=measurement_result();inputs.pins(mi.consumed)
    test_ref=ref(tests);test=inputs.json(test_ref)
    original=read(ROOT/read(ROOT/BUILD)['evidence']['path'])
    sources={**original['sources'],**reg['source_sha256']}
    need(test['source_sha256']==sources and test['actual_run'] is True and test['exit_code']==test['failed']==0,
         'actual current test receipt covering exact phase sources')
    resource=inputs.json(ref(work/'cache-claim.json'));claim(resource,'cache')
    artifacts=dict(original['artifacts'],tests=test_ref,build_admission=reg['build_admission'],
        launch_card=reg['cache_card'],resource_claim=ref(work/'cache-claim.json'),
        memory=ref(work/'memory-run/measurement.json'),serving_registration=ref(work/'registration.json'),
        serving_build_report=ref(work/'BUILD-REVIEW.md'),memory_result_report=ref(work/'MEMORY-RESULT-REVIEW.md'))
    for i,(p,h) in enumerate(sorted(mi.consumed.items())):
        artifacts['memory_dependency_'+str(i)]=dict(path=p,sha256=h)
    e=dict(original,phase='CACHE_LAUNCH_PASS',sources=sources,artifacts=artifacts)
    inputs.recheck();fresh(work/'CACHE-EVIDENCE.json',e)
    return e


def snapshot():
    cp=subprocess.run(['nvidia-smi','-i','0','--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=3,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    name,total,used,util,power=[x.strip() for x in cp.stdout.strip().split(',')]
    result=dict(device_name=name,total_bytes=float(total)*2**20,desktop_bytes=float(used)*2**20,util=float(util),power=float(power))
    need(all(math.isfinite(result[x]) for x in ('total_bytes','desktop_bytes','util','power')),'bad telemetry')
    return result


def optional_snapshot():
    try:return snapshot()
    except (OSError,ValueError,subprocess.SubprocessError):return None


def check_job(item,mode,own,parent,cmd,actual_cmd,parent_args):
    need(item['schema']=='gpu-job/v2' and item['tag']==TAGS[mode] and item['phase']=='running' and
         item['command']==cmd and item['command_sha256']==jr.command_digest(cmd,ROOT) and
         item['cwd']==str(ROOT) and item['child_identity']==own and
         dict(pid=item['pid'],create_time=item['create_time'])==parent and actual_cmd==cmd,
         'exact owned gpu_job PID/start/command')
    need(str(ROOT/'tools/gpu_job.py') in parent_args and parent_args[-3:]==['--_run-job',TAGS[mode],item['run_id']],
         'actual gpu_job parent')
    need(jr.liveness(own)==jr.liveness(parent)=='alive','live bound parent/child required')
    return item


def job_identity(mode,own=None):
    import psutil
    need(os.environ.get('KNEE_WORKSPACE_ROOT') in (None,str(ROOT)), 'alternate runtime root forbidden')
    own=own or jr.identity();proc=psutil.Process(own['pid']);parent=jr.identity(proc.ppid())
    args=psutil.Process(parent['pid']).cmdline();last=None
    for _ in range(40):
        try:
            item=jr.read_json(ROOT/'artifacts/logs'/TAGS[mode]/'job.json')
            return check_job(item,mode,own,parent,command(mode),proc.cmdline(),args)
        except (OSError,ValueError,KeyError) as exc:
            last=exc;time.sleep(.05)
    raise ValueError('gpu_job child identity handoff unavailable') from last


class Progress:
    def __init__(self,start,identity,token):
        self.last=self.previous=start;self.identity=identity;self.token=token;self.seq=-1;self.units=0
        self.phase='startup';self.active=0.;self.baseline=0.;self.baseunits=0;self.healthy=None;self.guard=None
    def check(self,now,status,sample,wall):
        if self.phase in ('view1','view2'):self.active+=now-self.previous
        self.previous=now
        if status:
            need(status['pid']==self.identity['pid'] and status['token']==self.token and
                 self.identity['create_time']-1<=status['time']<=wall+2 and
                 type(status['sequence']) is int and status['sequence']>=self.seq and
                 type(status['units']) is int and self.units<=status['units']<=580 and
                 status['phase'] in ('model_load','render','view1','view2','study_complete'),
                 'forged/regressing owned progress')
            if status['sequence']>self.seq:self.last=now
            self.seq,self.units,self.phase=status['sequence'],status['units'],status['phase']
        span=self.active-self.baseline
        if span>=120:
            rate=(self.units-self.baseunits)/span
            if rate>0 and (self.healthy is None or rate>self.healthy):
                self.healthy=rate
                if self.guard is None:
                    self.guard=ThroughputGuard(rate,self.active);self.guard.count=self.units
                else:self.guard.healthy_rate=rate
            self.baseline,self.baseunits=self.active,self.units
        need(now-self.last<900,'900s no real progress')
        if sample and now-self.last>=600:
            need(not(sample['util']>=95 and sample['power']<90 and sample['desktop_bytes']>=sample['total_bytes']-4*GIB),
                 '600s WDDM stall')
        if self.guard:
            why=self.guard.update(self.active,self.units);need(not why,why or '')


def supervise(cmd,run,env,token,*,sample=optional_snapshot,poll=1.,clock=time.monotonic,sleep=time.sleep):
    """C4 private controller; pure tiny-subprocess tests may inject the clock."""
    import psutil
    child=owned_root=None;owned={};survivors=[];code=125;failure=None;start=clock();identity=None
    try:
        with (run/'worker.log').open('xb',buffering=0) as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            owned_root=psutil.Process(child.pid);identity=jr.identity(child.pid);policy=Progress(start,identity,token)
            while True:
                collect_owned(owned_root,owned)
                if child.poll() is not None:code=child.returncode;break
                status=None
                try:status=read(run/'phase.json')
                except (OSError,ValueError):pass
                policy.check(clock(),status,sample(),time.time())
                diagnostic(run/'status.json',dict(pid=os.getpid(),phase=policy.phase,units=policy.units,elapsed=clock()-start))
                sleep(poll)
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc);code=125
    finally:
        try:
            if owned_root is not None:survivors=terminate_owned(owned_root,owned)
        except BaseException as exc:survivors=[dict(cleanup_error=type(exc).__name__)]
        if child is not None:
            try:
                if child.poll() is None:child.kill()
                child.wait(timeout=5)
            except BaseException as exc:survivors.append(dict(cleanup_error=type(exc).__name__))
        if survivors:code=125
        result=dict(schema='h198-serving-process/v1',child_exit_code=code,child=identity,
            owned_cleanup_complete=not survivors,survivors=survivors,failure=failure,
            seconds=clock()-start,source_sha256=sha(ROOT/SELF))
        # Authoritative receipt failure propagates only AFTER owned cleanup.
        fresh(run/'process.json',result)
    return result


def atomic_npz(path,**arrays):
    import numpy as np
    path=Path(path);need(not path.exists(),'immutable NPZ exists');tmp=path.with_suffix('.tmp')
    with tmp.open('xb') as stream:
        np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    io.replace(tmp,path)


def finite_cache(values,ids,targets):
    import numpy as np
    need(len(ids)==len(set(ids))==58 and len(targets)==12 and values.shape==(5,58,12) and
         values.dtype==np.float32 and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),
         'complete five x same58 x12 finite FP32 only')


def worker(mode):
    import psutil
    run=ROOT/REL/(mode+'-run');ticket=read(run/'ticket.json')
    need(ticket['mode']==mode and ticket['parent']==jr.identity(os.getppid()) and
         jr.digest(os.environ.get('C4_SERVING_TOKEN',''))==ticket['token_sha256'],'owned child token/PID/start')
    job_identity(mode,ticket['parent'])
    need(ticket['admission']==ref(ROOT/REL/('memory-admission.json' if mode=='memory' else 'CACHE-ADMISSION.json')),
         'child exact admission bytes')
    (memory_admission if mode=='memory' else cache_admission)()
    reg,card,_=validate();need(reg['fixture_only'] is False and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','one actual CUDA device')
    need(ticket['registration_sha256']==sha(ROOT/REL/'registration.json'),'child registration drift')
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_device(0);need(torch.cuda.device_count()==1,'single visible GPU')
    common,rad,runtime,_=bridge.modules();corpus=common.Corpus(bridge.CORPUS)
    files=bridge.build.unpack(bridge.read(bridge.build.OUT/'kernel/release.ipynb'))[0]
    from tools import exp169_partner_bundle_release as original
    funcs=bridge.functions(files['exp169_partner_bundle_release.py'],['model_meta','plane_flags','_infer_one'],dict(need=need,TARGETS=original.TARGETS))
    scope=dict(center3_windows_from=rad.center3_windows_from,windows_from=common.windows_from,normalise=rad.normalise,normalize_input=runtime.normalize_input)
    ids=card['corpus']['ids'];selected=[0] if mode=='memory' else list(range(58))
    values=np.empty((5,58,12),np.float32) if mode=='cache' else None
    desktop=read(run/'desktop.json');durations=[];peaks=[];raw=[];start=time.monotonic();sequence=0;units=0
    def progress(phase):
        nonlocal sequence
        sequence+=1
        # Model-load progress publication is diagnostic, never a scientific failure.
        diagnostic(run/'phase.json',dict(pid=os.getpid(),token=os.environ['C4_SERVING_TOKEN'],time=time.time(),
            sequence=sequence,units=units,phase=phase))
    import timm
    environment=dict(os=sys.platform,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),timm=timm.__version__,gpu=torch.cuda.get_device_name(0),
        physical_gpu=0,logical_device='cuda:0',cudnn_benchmark=False,cudnn_deterministic=True,
        precision='original FP16 forward; FP32 sigmoid and two-probability mean')
    for f,e in enumerate(card['models']):
        progress('model_load');model=None;model_start=time.monotonic()
        torch.cuda.synchronize(0);torch.cuda.reset_peak_memory_stats(0)
        try:
            ck=torch.load(ROOT/e['source'].replace('\\','/'),map_location='cpu',weights_only=False);funcs['model_meta'](ck,e)
            model=common.build_model(ck['arch'],pretrained=False);model.load_state_dict(ck['model'],strict=True)
            model=model.eval().to('cuda:0');del ck
            for i in selected:
                tick=time.monotonic();u=ids[i];progress('render');volume=corpus.volume(u);bound=card['corpus']['volumes'][i]
                need(bound['uid']==u and hashlib.sha256(volume.tobytes()).hexdigest()==bound['sha256'],'exact cached bytes')
                centers=common.select_centers(corpus.mask(u),64,36,corpus.spec,'within-slot-valid')
                need(list(centers)==bound['centers'],'exact centers')
                x=bridge.candidate_windows(volume,centers,scope)['payload'][2]
                need(list(x.shape)==[36,3,224,224] and x.dtype==np.float32,'original full-shape FP32 normalized input')
                flags=np.asarray(funcs['plane_flags'](centers,corpus.spec),bool)
                mirrored=x.copy();mirrored[flags]=mirrored[flags,:,:,::-1]
                progress('view1');a=funcs['_infer_one'](model,x,'cuda:0');units+=1
                progress('view2');b=funcs['_infer_one'](model,mirrored,'cuda:0');units+=1
                value=(a+b)/2
                bridge.probabilities(value[None],1);need(value.dtype==np.float32,'original FP32 TTA mean')
                torch.cuda.synchronize(0);peak=torch.cuda.max_memory_reserved(0);memory_ok(desktop,[peak])
                duration=time.monotonic()-tick;durations.append(duration)
                if values is not None:values[f,i]=value
                io.journal(run/'journal'/f'{f}-{i:03}.json',dict(identity=gate.b.KEY,fold=f,index=i,
                    seconds=duration,peak_reserved_bytes=peak),run)
                progress('study_complete')
            torch.cuda.synchronize(0);allocated=torch.cuda.max_memory_allocated(0);reserved=torch.cuda.max_memory_reserved(0)
            need(0<allocated<=reserved,'actual positive synchronized peak');memory_ok(desktop,[reserved]);peaks.append(reserved)
        finally:
            if model is not None:del model
            torch.cuda.empty_cache();torch.cuda.synchronize(0)
        if mode=='memory':
            row=dict(schema='h198-memory-model/v1',fold=f,model_sha256=e['sha256'],representative_index=0,
                plan_sha256=reg['memory_plan']['sha256'],source_sha256=reg['source_sha256'][SELF],
                views=2,input_shape=[36,3,224,224],peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,
                synchronized=True,model_deleted=True,seconds=time.monotonic()-model_start,environment=environment,
                desktop=desktop,registration_sha256=sha(ROOT/REL/'registration.json'),clean_oof=False)
            p=run/f'model-{f}.json';fresh(p,row);raw.append(ref(p))
    expected=5 if mode=='memory' else 290
    need(len(durations)==expected,'no partial success')
    if mode=='cache':
        finite_cache(values,ids,card['corpus']['targets'])
        atomic_npz(run/'c4-members.npz',ids=np.asarray(ids),targets=np.asarray(card['corpus']['targets']),folds=np.arange(5),values=values)
    output=dict(schema='h198-serving-worker/v1',mode=mode,state='MEASUREMENT_COMPLETE' if mode=='memory' else 'INFERENCE_COMPLETE',
        candidate=gate.b.KEY,kernel_sha256=KERNEL,models=5,studies=len(selected),workload=expected,failures=0,
        shape=[5,58,12] if mode=='cache' else None,dtype='float32',
        npz_sha256=sha(run/'c4-members.npz') if mode=='cache' else None,
        card_sha256=reg['cache_card']['sha256'],source_sha256=reg['source_sha256'][SELF],
        registration_sha256=sha(ROOT/REL/'registration.json'),plan_sha256=reg['memory_plan']['sha256'],
        admission_sha256=ticket['admission']['sha256'],
        renderer_sha256=card['renderer_sha256'],roster_sha256=jr.digest(ids),targets=card['corpus']['targets'],
        model_sha256=[e['sha256'] for e in card['models']],corpus_manifest_sha256=card['corpus']['metadata'].get('manifest.json'),
        study_seconds=durations,seconds=time.monotonic()-start,peak_reserved_bytes=peaks,raw_receipts=raw,
        environment=environment,exposure='EXPOSED',clean_oof=False,qualified_for_submission=False)
    fresh(run/'worker.json',output)
    return output


def finish(mode,run,process,reg,legacy,desktop,root=ROOT):
    need(process['child_exit_code']==0 and process['owned_cleanup_complete'] is True and not process['survivors'],
         'child failure/unconfirmed cleanup')
    result=read(run/'worker.json');expected=5 if mode=='memory' else 290
    need(result['mode']==mode and result['workload']==expected and result['failures']==0 and
         result['models']==5 and result['studies']==(1 if mode=='memory' else 58) and
         result['roster_sha256']==jr.digest(legacy['corpus']['ids']) and result['targets']==legacy['corpus']['targets'] and
         result['source_sha256']==reg['source_sha256'][SELF] and result['card_sha256']==reg['cache_card']['sha256'] and
         result['registration_sha256']==sha(root/REL/'registration.json') and result['kernel_sha256']==KERNEL and
         result['admission_sha256']==sha(root/REL/('memory-admission.json' if mode=='memory' else 'CACHE-ADMISSION.json')) and
         result['model_sha256']==[e['sha256'] for e in legacy['models']] and
         len(result['study_seconds'])==expected and all(math.isfinite(x) and x>0 for x in result['study_seconds']),
         'exact complete worker output')
    need(len(result['peak_reserved_bytes'])==5 and result['environment']['physical_gpu']==0 and
         result['environment']['logical_device']=='cuda:0' and '4070 Ti SUPER' in result['environment']['gpu'] and
         result['environment']['precision']=='original FP16 forward; FP32 sigmoid and two-probability mean',
         'exact worker device/precision/five peaks')
    memory_ok(desktop,result['peak_reserved_bytes'])
    if mode=='cache':
        import numpy as np
        need(result['state']=='INFERENCE_COMPLETE' and result['npz_sha256']==sha(run/'c4-members.npz'),'immutable complete NPZ')
        with np.load(run/'c4-members.npz',allow_pickle=False) as z:
            need(set(z.files)=={'ids','targets','folds','values'} and z['ids'].tolist()==legacy['corpus']['ids'] and
                 z['targets'].tolist()==legacy['corpus']['targets'] and z['folds'].tolist()==list(range(5)), 'exact cache roster')
            finite_cache(z['values'],z['ids'].tolist(),z['targets'].tolist())
    else:need(result['state']=='MEASUREMENT_COMPLETE' and result['npz_sha256'] is None,'measurement cannot populate cache')
    monitor=dict(schema='h198-serving-monitor/v1',status='PASS',worker_sha256=sha(run/'worker.json'),
        child_exit_code=0,owned_cleanup_complete=True,process=ref(run/'process.json',root),seconds=process['seconds'],
        registration_sha256=sha(root/REL/'registration.json'),source_sha256=reg['source_sha256'][SELF])
    fresh(run/'monitor.json',monitor)
    if mode=='memory':
        evidence=dict(schema='h198-memory-evidence/v1',status='PASS_MEASURED',candidate=gate.b.KEY,kernel_sha256=KERNEL,
            registration_sha256=sha(root/REL/'registration.json'),plan_sha256=reg['memory_plan']['sha256'],
            source_sha256=reg['source_sha256'][SELF],card_sha256=reg['cache_card']['sha256'],
            renderer_sha256=legacy['renderer_sha256'],K=36,tta='plane-aware-coronal-axial-flip',
            model_sha256=result['model_sha256'],physical_gpu=0,device_name=desktop['device_name'],
            actual_full_shape_both_views=True,peak_reserved_bytes=result['peak_reserved_bytes'],
            desktop_bytes=desktop['desktop_bytes'],total_bytes=desktop['total_bytes'],raw_receipts=result['raw_receipts'],
            worker=ref(run/'worker.json',root),monitor=ref(run/'monitor.json',root),raw_log=ref(run/'worker.log',root),
            admission=ref(root/REL/'memory-admission.json',root),launch=ref(run/'ticket.json',root),
            desktop=ref(run/'desktop.json',root),bound_job=ref(run/'bound-job.json',root),
            independent_result_review_required=True,qualified_for_submission=False)
        fresh(run/'measurement.json',evidence)
    return dict(status='COMPLETE_PENDING_INDEPENDENT_RESULT_REVIEW',mode=mode,qualified_for_submission=False)


def run(mode):
    need(os.name=='nt' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Windows physical0/logicalcuda0 required')
    (memory_admission if mode=='memory' else cache_admission)()
    reg,legacy,_=validate();job=job_identity(mode);desktop=snapshot();memory_ok(desktop,[0])
    if mode=='cache':memory_ok(desktop,measurement_result()[0]['peak_reserved_bytes'])
    cp=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=3,creationflags=subprocess.CREATE_NO_WINDOW)
    for line in cp.stdout.splitlines():
        parts=line.split(',',1)
        if len(parts)==2 and 'python' in parts[1].lower():
            need(parts[0].strip() in (str(os.getpid()),str(job['pid'])),'other Python GPU owner')
    work=ROOT/REL/(mode+'-run');work.mkdir(exist_ok=False);(work/'journal').mkdir()
    token=uuid.uuid4().hex
    fresh(work/'desktop.json',desktop)
    fresh(work/'bound-job.json',job)
    fresh(work/'ticket.json',dict(mode=mode,parent=jr.identity(),token_sha256=jr.digest(token),
        registration_sha256=sha(ROOT/REL/'registration.json'),
        admission=ref(ROOT/REL/('memory-admission.json' if mode=='memory' else 'CACHE-ADMISSION.json'))))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',C4_SERVING_TOKEN=token,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    process=supervise([bridge.PYTHON,'-B',str(ROOT/SELF),'_memory' if mode=='memory' else '_cache'],work,env,token)
    try:return finish(mode,work,process,reg,legacy,desktop)
    except BaseException as exc:
        diagnostic(work/'failure.json',dict(state='FAILED',reason=type(exc).__name__+': '+str(exc),automatic_retry=False))
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('register','validate','prepare','memory-admit-dry-run','memory-admit',
        'memory-result-validate','cache-admit-dry-run','prepare-cache','measure','run','_memory','_cache'))
    p.add_argument('--tests',type=Path);args=p.parse_args(argv)
    try:
        if args.command=='register':v=register()
        elif args.command=='validate':validate();v=dict(status='STATIC_VALID_NOT_ADMITTED')
        elif args.command=='prepare':v=prepare()
        elif args.command in ('memory-admit-dry-run','memory-admit'):v=admit_memory(args.command.endswith('dry-run'))
        elif args.command=='memory-result-validate':measurement_result(require_review=False);v=dict(status='RAW_RESULT_VALID_REVIEW_REQUIRED')
        elif args.command=='cache-admit-dry-run':cache_admission();v=dict(status='CACHE_ADMISSION_VALID_NO_LAUNCH')
        elif args.command=='prepare-cache':need(args.tests is not None,'actual test receipt required');v=prepare_cache(args.tests)
        elif args.command=='measure':v=run('memory')
        elif args.command=='run':v=run('cache')
        else:v=worker('memory' if args.command=='_memory' else 'cache')
        print(json.dumps({k:v[k] for k in ('schema','status','state','fixture_only','qualified_for_submission') if k in v}));return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='HOLD',error=str(exc),execution_authorized=False)));return 2


if __name__=='__main__':raise SystemExit(main())
