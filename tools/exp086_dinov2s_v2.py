"""EXP086 single generic DINOv2 transfer. Import/prepare never train or use CUDA.

The no-Gold entrypoint is rendered from frozen EXP084 text. Explicit phase runs
require a root decision and an external owned-process resource receipt.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import difflib
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'exp086-dinov2s-v2'
ARCH = 'vit_small_patch14_dinov2.lvd142m'
PARENT = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
PROTOCOL = 'artifacts/experiments/EXP086_DINOV2S_G1_V2_PROTOCOL'
RUNNER = 'training/runners/exp086_dinov2s_v2.py'
EVIDENCE = 'docs/handoffs/evidence/2026-10/2026-10-04-exp086-dinov2s/v2'
ROOT_DECISION = 'docs/handoffs/evidence/2026-10/2026-10-04-exp086-dinov2s/root-v2-cpu-decision.json'
WEIGHT = Path('C:/Users/Taeyang/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m/snapshots/4610ca143709d58a633b6397a74412c2c3842454/model.safetensors')
WEIGHT_SHA = '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'
CODE_NAMES = ('raptor_train.py', 'raptor_common.py', 'raptor_data.py', 'raptor_runtime.py')
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
TAG = 'EXP086_DINOV2S_G1_V2'
V1_PROTOCOL = 'artifacts/experiments/EXP086_DINOV2S_G1_PROTOCOL/manifest.json'
V1_FAILURE = 'artifacts/experiments/EXP086_DINOV2S_G1_PILOT_RESOURCE/receipt.json'
ADAPTER = 'training/adapters/dinov2_cpu_pos_v2.py'
POSITION_POLICY = 'canonical518-trainable-cpu-fp32-bicubic-aa-v2'
RESOURCE_SHA = '15c0fe61d5d0b350557093ec38a1dc902b73e12c76328c4951934cc88f1da954'


def need(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def contained(root, value):
    p = Path(str(value).removeprefix('workspace://'))
    need(not p.is_absolute() and not p.drive and '..' not in p.parts, 'contained workspace path required')
    result = (Path(root) / p).resolve()
    need(result.is_relative_to(Path(root).resolve()), 'path escapes workspace'); return result


def base_guard():
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    from tools import exp084_group_loss
    return exp084_group_loss


def uid_hash(ids): return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def read_rows(path): return base_guard().read_rows(path)
def worker_init(worker): return base_guard().worker_init(worker)
def install_gold_blocker(root): return base_guard().install_gold_blocker(root)
def validate_membership(*a): return base_guard().validate_membership(*a)
def validate_label_contract(*a): return base_guard().validate_label_contract(*a)
def loss_weights(*a): return base_guard().loss_weights(*a)


def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_offline_model(arch, asset):
    """Explicit local strict load; neither hub lookup nor competition weights."""
    need(arch == ARCH and asset['sha256'] == WEIGHT_SHA, 'generic DINO initial contract differs')
    path = Path(asset['path']); need(sha256(path) == WEIGHT_SHA, 'local initial weight differs')
    need(os.environ.get('HF_HUB_OFFLINE') == '1', 'offline initial load required')
    if str(ROOT / 'training/src') not in sys.path: sys.path.insert(0, str(ROOT / 'training/src'))
    from raptor_common import build_model
    from safetensors.torch import load_file
    model = build_model(arch, pretrained=False)
    sd = load_file(str(path), device='cpu')
    need(set(sd) == set(model.backbone.state_dict()), 'strict backbone key set mismatch')
    model.backbone.load_state_dict(sd, strict=True)
    need(model.backbone.num_features == 384 and tuple(model.backbone.pos_embed.shape) == (1, 1370, 384)
         and tuple(model.backbone.patch_embed.patch_size) == (14, 14) and model.backbone.dynamic_img_size,
         'DINO native position/patch/dynamic-input contract differs')
    from training.adapters.dinov2_cpu_pos_v2 import install
    install(model.backbone)
    return model


def cpu_probe(out, root=ROOT):
    """One changed-op wiring probe, not a repeat of the original full probe."""
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only probe requires hidden CUDA')
    import torch
    import random
    import numpy as np
    from training.adapters import dinov2_cpu_pos_v2 as adapter
    from timm.layers.pos_embed import resample_abs_pos_embed
    from timm.models.vision_transformer import VisionTransformer
    started=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    need(not torch.cuda.is_initialized(), 'CPU probe initialized CUDA')
    old=read(Path(root)/V1_PROTOCOL);original_probe=read(Path(root)/old['cpu_probe']['path'])
    # Changed interpolation operation alone: exact CPU forward and canonical gradients.
    gen=torch.Generator().manual_seed(20261004)
    p=torch.randn((1,1370,384),generator=gen,requires_grad=True)
    q=p.detach().clone().requires_grad_(True)
    target=torch.randn((1,257,384),generator=gen)
    a=adapter.resample_cpu(p,(16,16),(37,37),1)
    b=resample_abs_pos_embed(q,(16,16),old_size=(37,37),num_prefix_tokens=1,interpolation='bicubic',antialias=True)
    (a*target).sum().backward();(b*target).sum().backward()
    need(torch.equal(a,b) and torch.equal(p.grad,q.grad) and bool(torch.isfinite(p.grad).all()) and bool((p.grad!=0).any()),'changed CPU operation/gradient differs')
    operation={'forward_exact':True,'gradient_exact':True,'finite_nonzero_canonical_gradient':True,'position_shape':[1,1370,384],'output_shape':[1,257,384]}
    # One model load and one synthetic forward/backward; initialization evidence reused.
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    original_class_method=VisionTransformer._pos_embed
    model=build_offline_model(ARCH,old['initial_asset']);initial=state_hash(model.state_dict())
    need(initial==old['arm']['initial_state_sha256'],'canonical initial-state changed')
    need(VisionTransformer._pos_embed is original_class_method,'global timm method changed')
    # Installation must not consume RNG, add parameters, or replace the canonical parameter.
    parameter=model.backbone.pos_embed
    rng=torch.get_rng_state().clone();state_keys=list(model.state_dict())
    need(parameter.requires_grad and tuple(parameter.shape)==(1,1370,384),'position parameter detached/resized')
    x=torch.randn((1,1,3,224,224),generator=gen)
    model.backbone.set_grad_checkpointing(True);model.train()
    logits=model(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),torch.full((1,12),.5));loss.backward()
    need(bool(torch.isfinite(logits).all()) and parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and bool((parameter.grad!=0).any()),'canonical gradient wiring failed')
    need(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.requires_grad),'model finite gradients failed')
    need(state_hash(model.state_dict())==initial and list(model.state_dict())==state_keys and model.backbone.pos_embed is parameter,'probe altered model tensors')
    need(not torch.cuda.is_initialized(),'CUDA initialized')
    result={'schema':SCHEMA,'status':'CPU_V2_CHANGED_POSITION_AND_WIRING_PASS','observed_at':datetime.now(timezone.utc).isoformat(),
        'original_cpu_probe':old['cpu_probe'],'asset':old['initial_asset'],'initial_state_sha256':initial,
        'position_policy':POSITION_POLICY,'threads':1,'strict_deterministic':True,'warn_only':False,
        'changed_operation':operation,'single_model_wiring':{'loads':1,'synthetic_shape':[1,1,3,224,224],'forward_backward':1,'finite_all_gradients':True,'position_gradient_finite_nonzero':True,'canonical_parameter_identity_preserved':True,'global_timm_method_unchanged':True,'parameters':sum(v.numel() for v in model.parameters())},
        'CPU_FP16':'Not repeated; v1 diagnostic retained, actual CUDA FP16 still requires V2 pilot',
        'source_bindings':{ADAPTER:sha256(Path(root)/ADAPTER)},'constructor_sha256':hashlib.sha256(inspect.getsource(build_offline_model).encode()).hexdigest(),
        'probe_code_sha256':hashlib.sha256(inspect.getsource(cpu_probe).encode()).hexdigest(),'seconds':time.monotonic()-started,
        'optimizer_updates':0,'GPU':False,'downloads':0,'real_image_reads':0,'limitations':['CPU-only evidence; cross-device copy backward and CUDA deterministic execution unmeasured','Same exposed weak development; not clinical/private/T4 evidence']}
    write_new(out,result);return result


def arm_for(protocol, tag):
    arm = protocol['arm']; need(tag in (TAG, TAG+'_PILOT'), 'unregistered candidate tag'); return arm


def validate_protocol(path, root=ROOT, verify_inputs=False):
    p = read(path)
    need(p['schema'] == SCHEMA and p['arm']['expected_args']['arch'] == ARCH, 'wrong EXP086 protocol')
    for n,h in p['source_bindings'].items(): need(sha256(contained(root,n)) == h, 'frozen binding changed: '+n)
    for n,h in p['training_code_sha256'].items(): need(sha256(Path(root)/'training/src'/n) == h, 'original source changed: '+n)
    if verify_inputs:
        for n,h in p['input_bindings'].items(): need(sha256(contained(root,n)) == h, 'input binding changed: '+n)
        need(sha256(p['initial_asset']['path']) == WEIGHT_SHA, 'generic weights changed')
    need(p['position_policy']==POSITION_POLICY and p['execution']['full_seconds']==5300, 'V2 operation/budget changed')
    for n,h in p['installed_source_bindings'].items(): need(sha256(n)==h,'installed interpolation source changed')
    ids = read(contained(root,p['ordered_ids']['path']))
    need(all(uid_hash(ids[k]) == p['ordered_ids'][k+'_sha256'] for k in ('train','val')), 'ordered UID hash differs')
    need(len(ids['train'])==3455 and len(ids['val'])==888 and not set(ids['train'])&set(ids['val']), 'fixed cohort mismatch')
    return p,ids


def validate_arguments(args, root):
    p,ids = validate_protocol(args.protocol,root,True)
    for k,v in p['arm']['expected_args'].items(): need(getattr(args,k,None)==v, 'frozen argument differs: '+k)
    for k,v in p['paths'].items(): need(Path(getattr(args,k)).resolve()==contained(root,v), 'frozen path differs: '+k)
    need(args.phase in ('pilot','full') and args.tag == TAG+('_PILOT' if args.phase=='pilot' else ''), 'phase/tag mismatch')
    for k,v in {**THREAD_ENV,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}.items(): need(os.environ.get(k)==v, 'registered environment required: '+k)
    return p,ids


def validate_pretraining(provenance, env, protocol, arm):
    base_guard().validate_pretraining({**provenance,'metric_access':'deferred_until_all_three_complete'},env,protocol,arm)
    need(provenance['position_policy']==POSITION_POLICY,'position policy missing')
    need(provenance['metric_access']=='deferred_until_final_candidate', 'metric exposure')
    need(provenance['initial_asset_sha256']==WEIGHT_SHA and provenance['phase'] in ('pilot','full'), 'offline initial provenance missing')


def validate_completion(out, protocol, arm):
    s=read(Path(out)/'summary.json')
    need(s['selected_epoch_1based']==8 and s['optimizer_updates']==3448 and s['skipped_updates']==0
         and s['weak_oof_macro_auc'] is None and s['metric_access']=='deferred_until_final_candidate'
         and s['train']==3455 and s['val']==888 and s['provenance']['initial_state_sha256']==arm['initial_state_sha256'], 'full completion gate failed')
    for n in ('model_best.pt','oof_best.csv','provenance.json','history.json'): need((Path(out)/n).is_file(), 'missing final artifact: '+n)


def finish_pilot(out, protocol, provenance, timings, updates, skips, initial_seconds, eval_seconds, peak):
    need(updates==32 and skips==0 and len(timings)==32, 'pilot requires exact32 successful updates')
    steady=sum(timings[8:])/24
    projection=1.3*(steady*3448)+300
    value={'schema':SCHEMA,'status':'PASS','phase':'pilot','position_policy':POSITION_POLICY,'v1_elapsed_seconds':protocol['v1_failure']['elapsed_seconds'],'protocol_sha256':provenance['protocol_sha256'],
           'initial_state_sha256':provenance['initial_state_sha256'],'initial_asset_sha256':WEIGHT_SHA,
           'updates':updates,'skips':skips,'training_ids_only':True,'eval_train_studies':4,'validation_rows':0,
           'steady_update_seconds':steady,'warmup_steps':8,'step_seconds':timings,'initial_seconds':initial_seconds,
           'eval_seconds':eval_seconds,'projected_full_seconds':projection,'peak_gpu_allocated_bytes':peak,
           'metric_access':'disabled','checkpoints_saved':False,'predictions_saved':False,'argv':[sys.executable,*sys.argv]}
    write_new(Path(out)/'receipt.json',value)


def finish_full(out, protocol, provenance, trace, elapsed):
    out=Path(out);validate_completion(out,protocol,protocol['arm'])
    write_new(out/'trace.json',{'schema':SCHEMA,'training_trace_sha256':trace,'seed':42,'epochs':8,'updates':3448,
        'trace_payload':'UTF8 json.dumps([epoch,ordered_batch_UIDs],separators=(comma,colon)) concatenated in actual optimizer order',
        'input_sampler':'unchanged raptor_data.StudyWindows + UID/epoch deterministic within-slot-valid/intensity streams',
        'train_ids_sha256':provenance['train_ids_ordered_sha256'],'val_ids_sha256':provenance['eval_ids_ordered_sha256']})
    write_new(out/'receipt.json',{'schema':SCHEMA,'status':'PASS','phase':'full','position_policy':POSITION_POLICY,'v1_elapsed_seconds':protocol['v1_failure']['elapsed_seconds'],'protocol_sha256':provenance['protocol_sha256'],
        'initial_state_sha256':provenance['initial_state_sha256'],'initial_asset_sha256':WEIGHT_SHA,'updates':3448,'skips':0,
        'metric_access':'deferred_until_final_candidate','validation_rows':888,'training_trace_sha256':trace,
        'elapsed_seconds':elapsed,'argv':[sys.executable,*sys.argv],
        'outputs':{n:sha256(out/n) for n in ('summary.json','provenance.json','history.json','trace.json','model_best.pt','oof_best.csv')}})


def pilot_cost_gate(pilot, elapsed):
    need(pilot.get('status')=='PASS' and pilot.get('phase')=='pilot' and pilot.get('updates')==32 and pilot.get('skips')==0
         and pilot.get('training_ids_only') is True and pilot.get('validation_rows')==0, 'matching successful T-only pilot required')
    need(pilot.get('position_policy')==POSITION_POLICY and pilot.get('schema')==SCHEMA,'V2 pilot operation missing')
    times=pilot.get('step_seconds',[])
    need(len(times)==32 and all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in times),'32 measured intervals required')
    steady=pilot['steady_update_seconds']; projected=pilot['projected_full_seconds']
    need(steady==sum(times[8:])/24,'pilot speed must match measured intervals')
    need(type(steady) in (int,float) and math.isfinite(steady) and steady>0, 'invalid pilot speed')
    need(projected==1.3*(steady*3448)+300 and projected<=5300 and type(elapsed) in (int,float)
         and 0<elapsed<=180 and projected+elapsed+read(ROOT/V1_FAILURE)['elapsed_seconds']<=5580, 'pilot cost exceeds contract')


def resource_gate(receipt, argv, limit):
    need(receipt.get('timeout_seconds')==limit and receipt.get('access_failures')==[], 'resource timeout/access differs')
    need(receipt.get('status')=='COMPLETE' and receipt.get('child_exit_code')==receipt.get('wrapper_exit_code')==receipt.get('child_exit_code_before_cleanup')==0
         and receipt.get('failure') is None and receipt.get('argv')==argv and receipt.get('retries')==0
         and receipt.get('supervisor_sha256')==RESOURCE_SHA, 'resource execution failed')
    pid=receipt.get('child_identity',{}).get('pid')
    need(type(pid) is int and pid>0 and pid in {x.get('pid') for x in receipt.get('observed_identities',[])}, 'owned worker PID missing')
    need(receipt.get('sample_count',0)>0 and receipt.get('sampled_tree_rss_max_bytes',0)>0
         and receipt.get('access_failure_count')==receipt.get('failed_sample_attempts')==0
         and all(receipt.get('cleanup',{}).get(k)==[] for k in ('actions','errors','survivors')), 'resource telemetry/cleanup failed')
    need(type(receipt.get('elapsed_seconds')) in (int,float) and 0<receipt['elapsed_seconds']<limit, 'phase exceeded absolute cap')


def prepare(probe_path, root=ROOT):
    root=Path(root);out=root/PROTOCOL;need(not out.exists(),'new V2 protocol only')
    install_gold_blocker(root)
    from tools import exp086_dinov2s as v1
    old,_=v1.validate_protocol(root/V1_PROTOCOL,root,True)
    decision=read(root/ROOT_DECISION);need(decision['status']=='AUTHORIZE_CPU_PREPARATION_V2_ONLY','V2 CPU decision missing')
    failure=read(root/V1_FAILURE);need(failure['status']=='FAILED' and failure['child_exit_code']==1 and 83.23<failure['elapsed_seconds']<83.24,'original V1 failure differs')
    probe=read(probe_path)
    need(probe['schema']==SCHEMA and probe['status']=='CPU_V2_CHANGED_POSITION_AND_WIRING_PASS' and probe['position_policy']==POSITION_POLICY
        and probe['initial_state_sha256']==old['arm']['initial_state_sha256'] and probe['threads']==1 and probe['strict_deterministic'] and not probe['warn_only']
        and probe['constructor_sha256']==hashlib.sha256(inspect.getsource(build_offline_model).encode()).hexdigest()
        and probe['probe_code_sha256']==hashlib.sha256(inspect.getsource(cpu_probe).encode()).hexdigest()
        and probe['source_bindings'][ADAPTER]==sha256(root/ADAPTER),'changed CPU probe binding differs')
    p=copy.deepcopy(old);p.update(schema=SCHEMA,status='CPU_PREPARED_REQUIRES_ROOT_PHASE_DECISION',prepared_at=datetime.now(timezone.utc).isoformat(),
        v1_protocol={'path':V1_PROTOCOL,'sha256':sha256(root/V1_PROTOCOL)},v1_failure={'path':V1_FAILURE,'sha256':sha256(root/V1_FAILURE),'elapsed_seconds':failure['elapsed_seconds']},
        original_cpu_probe=old['cpu_probe'],cpu_probe={'path':Path(probe_path).resolve().relative_to(root.resolve()).as_posix(),'sha256':sha256(probe_path)},position_policy=POSITION_POLICY)
    p['arm']['tag']=TAG
    p['outputs']={n:f'artifacts/experiments/{TAG}/{Path(path).name}' for n,path in old['outputs'].items()};p['arm']['outputs']=copy.deepcopy(p['outputs'])
    p['execution'].update(full_seconds=5300,total_seconds=5580,v1_elapsed_seconds=failure['elapsed_seconds'],projection='1.3*(mean(last24 of32 update seconds)*3448)+300<=5300')
    p['limitations'] += ['V1 FAILED preserved; V2 CPU differentiable canonical position resampling changes runtime operation, not initial parameters','CPU/CUDA interpolation rounding may differ; no serving-factory/T4 parity established']
    new_sources=['tools/exp086_dinov2s_v2.py',RUNNER,ADAPTER,'tests/test_exp086_dinov2s_v2.py',V1_PROTOCOL,V1_FAILURE,ROOT_DECISION,p['cpu_probe']['path']]
    p['source_bindings'].update({n:sha256(root/n) for n in new_sources})
    import importlib.util
    timm_base=Path(importlib.util.find_spec('timm').origin).parent
    p['installed_source_bindings']={str(timm_base/n):sha256(timm_base/n) for n in ('layers/pos_embed.py','models/vision_transformer.py')}
    original=(root/'training/src/raptor_train.py').read_text(encoding='utf-8');rendered=render_training_source(original)
    need((root/RUNNER).read_text(encoding='utf-8')==rendered,'V2 runner differs from renderer')
    out.mkdir(parents=True);write_new(out/'manifest.json',p)
    (out/'trainer.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),rendered.splitlines(True),fromfile='frozen/raptor_train.py',tofile=RUNNER)),encoding='utf-8')
    validate_protocol(out/'manifest.json',root,True);return p


def run(protocol, phase, decision, pilot_dir=None, root=ROOT):
    p,_=validate_protocol(protocol,root,True);d=read(decision)
    need(d.get('protocol_sha256')==sha256(protocol) and d.get('allow_phase')==phase and d.get('retries')==0,'root phase decision required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')!='','CUDA hidden in inherited launch environment')
    out=Path(root)/'artifacts/experiments'/(TAG+('_PILOT' if phase=='pilot' else ''));limit=180 if phase=='pilot' else 5300
    need(not out.exists() and not out.with_name(out.name+'_RESOURCE').exists(),'fresh phase paths required; retries0')
    if phase=='full':
        need(pilot_dir is not None,'full requires completed pilot');path=Path(pilot_dir)/'receipt.json';pilot=read(path)
        need(Path(pilot_dir).resolve()==(Path(root)/'artifacts/experiments'/(TAG+'_PILOT')).resolve(),'registered pilot path required')
        resource=read(Path(pilot_dir).with_name(Path(pilot_dir).name+'_RESOURCE')/'receipt.json')
        need(pilot['protocol_sha256']==sha256(protocol) and d.get('pilot_receipt_sha256')==sha256(path),'pilot binding mismatch')
        need(pilot['initial_state_sha256']==p['arm']['initial_state_sha256'] and pilot['initial_asset_sha256']==WEIGHT_SHA,'pilot initial state differs')
        resource_gate(resource,phase_command(p,protocol,'pilot',root),180);pilot_cost_gate(pilot,resource['elapsed_seconds'])
        need(pilot.get('argv')==phase_command(p,protocol,'pilot',root),'pilot argv differs')
    command=phase_command(p,protocol,phase,root)
    from tools.process_resource_receipt import supervise
    before=dict(os.environ)
    try:
        os.environ.update({**THREAD_ENV,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
        receipt=supervise(command,out.with_name(out.name+'_RESOURCE'),limit,root)
    finally:os.environ.clear();os.environ.update(before)
    resource_gate(receipt,command,limit)
    result=read(out/'receipt.json');need(result['status']=='PASS' and result['protocol_sha256']==sha256(protocol),'phase receipt failed')
    need(result.get('argv')==command and result.get('position_policy')==POSITION_POLICY,'phase argv/position differs')
    if phase=='pilot':pilot_cost_gate(result,receipt['elapsed_seconds'])
    else:need(p['v1_failure']['elapsed_seconds']+resource['elapsed_seconds']+receipt['elapsed_seconds']<=5580,'combined V1+V2 cap exceeded')
    return receipt


def phase_command(p, protocol, phase, root=ROOT):
    need(phase in ('pilot','full'),'registered phase required')
    command=[sys.executable,str(Path(root)/RUNNER),'--protocol',str(Path(protocol).resolve()),'--phase',phase,'--tag',TAG+('_PILOT' if phase=='pilot' else '')]
    for k,v in {**p['arm']['expected_args'],**p['paths']}.items():
        option='--'+k.replace('_','-')
        if isinstance(v,bool):
            if v:command.append(option)
        else:command += [option,str(contained(root,v)) if k in p['paths'] else str(v)]
    return command


def render_training_source(original):
    from tools.exp082_r34_loss import replace_once
    s=base_guard().render_training_source(original)
    s=s.replace('EXP084 grouped fixed-final','EXP086 DINOv2 fixed-final')
    s=replace_once(s,'def main():','def main():\n    phase_started = time.monotonic()')
    s=replace_once(s,'    ap.add_argument("--protocol", required=True)',
        '    ap.add_argument("--protocol", required=True)\n    ap.add_argument("--phase", choices=["pilot", "full"], required=True)')
    s=replace_once(s,'from tools import exp084_group_loss as guard','from tools import exp086_dinov2s_v2 as guard')
    s=s.replace('deferred_until_all_three_complete','deferred_until_final_candidate')
    start=s.index('    use_timm = a.pretrained == "timm"');end=s.index('    source_files = ',start)
    s=s[:start]+'''    model = guard.build_offline_model(a.arch, protocol["initial_asset"]).to(dev)
'''+s[end:]
    s=replace_once(s,'    guard.validate_pretraining(provenance, run_environment, protocol, arm)',
        '    provenance.update(phase=a.phase, initial_asset_sha256=guard.WEIGHT_SHA, position_policy=guard.POSITION_POLICY)\n    guard.validate_pretraining(provenance, run_environment, protocol, arm)')
    start=s.index('    if a.grad_ckpt:');end=s.index('    head = ',start)
    s=s[:start]+'''    if a.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
        guard.need(model.backbone.grad_checkpointing is True, "Gradient checkpointing required; no silent fallback")
        log("gradient checkpointing on")
'''+s[end:]
    s=replace_once(s,'    t_start = time.time()',
        '''    t_start = time.time()
    initial_seconds = time.monotonic() - phase_started
    timings = []
    trace = __import__("hashlib").sha256()
    train_members = set(train_ids)''')
    s=replace_once(s,'        for i, (x, y, _) in enumerate(tl):',
        '''        previous_step_finished = time.monotonic()
        for i, (x, y, uids) in enumerate(tl):
            step_started = previous_step_finished
            guard.need(all(u in train_members for u in uids), "Non-T study in training")
            trace.update(json.dumps([ep, uids], separators=(",", ":")).encode())''')
    s=replace_once(s,'                    log(f"FP16 overflow: optimizer/scheduler step skipped ({skipped_updates} total)")',
        '                    raise RuntimeError("Any skipped optimizer update fails EXP086; no automatic precision change")')
    marker='        train_s = time.time() - t0'
    s=replace_once(s,marker,'''            torch.cuda.synchronize()
            previous_step_finished = time.monotonic()
            timings.append(previous_step_finished - step_started)
            if a.phase == "pilot" and optimizer_updates == 32:
                break
        if a.phase == "pilot":
            torch.cuda.synchronize(); eval_started = time.monotonic()
            _, _, pilot_p, pilot_u = evaluate(train_ids[:4])
            torch.cuda.synchronize()
            guard.need(pilot_u == train_ids[:4] and pilot_p.shape == (4, 12), "Pilot T-only evaluation differs")
            guard.finish_pilot(out, protocol, provenance, timings, optimizer_updates, skipped_updates,
                               initial_seconds, time.monotonic() - eval_started, torch.cuda.max_memory_allocated())
            return
'''+marker)
    s=replace_once(s,'    guard.validate_completion(out, protocol, arm)',
        '    guard.finish_full(out, protocol, provenance, trace.hexdigest(), time.monotonic() - phase_started)')
    s=s.replace('Final predictions saved; validation metrics deferred until all three fits complete',
        'Final predictions saved; metrics belong to separate fixed evaluator')
    return s


def main():
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    q=sub.add_parser('cpu-probe');q.add_argument('--out',required=True)
    q=sub.add_parser('prepare');q.add_argument('--probe',required=True)
    q=sub.add_parser('run');q.add_argument('--protocol',required=True);q.add_argument('--phase',choices=['pilot','full'],required=True);q.add_argument('--root-decision',required=True);q.add_argument('--pilot-dir')
    a=ap.parse_args()
    result=cpu_probe(a.out) if a.action=='cpu-probe' else prepare(a.probe) if a.action=='prepare' else run(a.protocol,a.phase,a.root_decision,a.pilot_dir)
    print(json.dumps({'status':result['status']},indent=2))


if __name__=='__main__':main()
