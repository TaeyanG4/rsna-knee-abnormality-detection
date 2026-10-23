"""K2 prospective gate k2-1.0.2; fresh CPU, complete native provenance.

BUILD never imports/hashes canonical or replays an historical gate. FINAL adds
mandatory independently reviewed transport source evidence. No model or send.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import re
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import strict_variant_gate as base
from tools import l1_rco_k2_v102 as b
from tools import l1_rco_k2_worker_v101 as w

VERSION='k2-1.0.2'
PHASES={'BUILD_READY','FINAL_READY'}
NATIVE=b.NATIVE
need=base.need


def snapshot(stage,inputs):
    stage=base.local_path(stage,inputs.root)
    base.need(stage.relative_to(inputs.root).as_posix()==b.STAGE.relative_to(b.ROOT).as_posix(), 'exact K2 stage')
    reg=inputs.json(stage/'registration.json')
    base.need(reg['schema']=='k2-registration/v1.0.2' and reg['candidate_id']==b.CANDIDATE and
        reg['kernel_id']==w.SLUG and reg['version']==w.VERSION and reg['contract']==w.CONTRACT and
        reg['qualified_for_submission'] is False, 'K2 registration contract')
    base.need(reg['provenance_contract']==b.PROVENANCE_CONTRACT and reg['stage_version']==b.STAGE_VERSION and
        reg['builder_version']==b.VERSION and reg['worker_version']==reg['asset_version']==reg['version']==w.VERSION and
        reg['stage']==stage.relative_to(inputs.root).as_posix() and reg['workspace']==str(inputs.root) and reg['fixture_only'] is False and
        reg['worker_source']=='tools/l1_rco_k2_worker_v101.py' and reg['worker_sha256']==b.WORKER_SHA and
        reg['final_only_sources']==list(b.FINAL_ONLY_SOURCES),'stage/runtime/source partition contract')
    base.need(all(reg['runtime'][k]==v for k,v in b.TIMING_POLICY.items()),'registered complete runtime policy')
    source=inputs.json(stage/'source-manifest.json')
    base.need(base.sha(inputs.read(stage/'source-manifest.json'))==reg['source_manifest_sha256'], 'source manifest binding')
    base.need(set(source)==b.required_sources(inputs) and not set(b.FINAL_ONLY_SOURCES)&set(source),'complete BUILD inventory / final-only singleton')
    parent,parent_sources=b.historical(inputs)
    renamed={n.replace('_v102.py','_v101.py').replace('-v102.py','-v101.py') for n in b.NEW}
    for name,h in parent_sources.items():
        if name not in b.FINAL_ONLY_SOURCES and name not in renamed:base.need(source[name]==h,'retained source pin differs')
    base.need(all(reg[k]==parent[k] for k in ('contract','visible_ids','smoke_holdout_ids','n32_ids','cache','parity','ast_identity')),'scientific registration differs')
    base.need(source[b.NATIVE]==b.NATIVE_SHA and source[reg['worker_source']]==b.WORKER_SHA,'native/worker exact source')
    for name,pin in source.items():
        base.need(inputs.digest(name)==pin, 'frozen/new source pin: '+name)
    base.need(set(b.NEW+b.FROZEN)<=set(source), 'required policy source inventory')
    for name,pin in b.PINS.items():
        base.need(source[Path(name).as_posix()]==pin, 'authenticated K1 lineage')
    base.need(reg['assets']==b.ASSETS.relative_to(b.ROOT).as_posix() and reg['asset_manifest_sha256']==b.ASSET_SHA,'exact retained asset identity')
    assets=base.local_path(reg['assets'],inputs.root)
    am=inputs.json(assets/'asset-manifest.json')
    base.need(base.sha(inputs.read(assets/'asset-manifest.json'))==reg['asset_manifest_sha256'], 'asset manifest pin')
    base.need(am['contract']==w.CONTRACT and am['version']==w.VERSION and am['dataset']==b.DATASET and am['private'] is True and
        am['wheels']==b.s6.WHEEL_HASHES and am['deployment_versions']==b.s6.WHEEL_VERSIONS, 'asset settings/wheels')
    paths=sorted(p for p in assets.rglob('*') if p.is_file());inputs.folders[assets]=paths
    base.need({p.relative_to(assets).as_posix() for p in paths}==set(am['files'])|{'asset-manifest.json','dataset-metadata.json'}, 'added/removed asset')
    for name,pin in am['files'].items():
        base.safe_name(name)
        base.need(inputs.digest(assets/name)==pin,'asset tamper: '+name)
    base.need({n for n in am['files'] if n.endswith(('.pt','.safetensors'))}=={'weights/E_prior_s0_f0.pt','weights/anatomy.pt'} and
        am['files']['weights/E_prior_s0_f0.pt']==w.MODEL_SHA and am['files']['weights/anatomy.pt']==w.ANATOMY_SHA, 'one model/anatomy')
    dmeta=inputs.json(assets/'dataset-metadata.json')
    base.need(dmeta['id']==b.DATASET and dmeta['title']==b.DATASET.split('/')[1], 'asset slug')
    origins=inputs.json(b.ASSET_WORK.relative_to(b.ROOT)/'asset-origins.json')
    for name,v in origins.items():
        base.need(am['files'][name]==v['sha256']==inputs.digest(v['source']) and v['relation']=='byte-identical','origin diff')
    raw={n:inputs.read(stage/'kernel'/n) for n in ('release.ipynb','kernel-metadata.json')}
    paths=sorted(p for p in (stage/'kernel').rglob('*') if p.is_file()); inputs.folders[stage/'kernel']=paths
    base.need({p.name for p in paths}==set(raw),'kernel inventory')
    base.need(base.sha(raw['release.ipynb'])==b.NOTEBOOK_SHA and base.sha(raw['kernel-metadata.json'])==b.METADATA_SHA,'byte-identical retained notebook/metadata')
    metadata=base.parse(raw['kernel-metadata.json'])
    expected=dict(id=w.SLUG,title=w.SLUG.split('/')[1],code_file='release.ipynb',language='python',
        kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[b.DATASET,b.CODEC_DATASET],competition_sources=['rsna-knee-abnormality-detection'],kernel_sources=[])
    base.need(metadata==expected,'exact K2 metadata')
    base.need(base.parse(raw['release.ipynb'])==b.notebook(reg['asset_manifest_sha256']), 'notebook differs from registered new path')
    inv=inputs.json(stage/'embedded-file-manifest.json')
    base.need(inv['new_sources']=={n:source[n] for n in b.NEW} and inv['inherited_assets']==origins and
        inv['removed']==[] and inv['changed_inherited']==[] and
        inv['notebook_sha256']==base.sha(raw['release.ipynb']) and
        inv['metadata_sha256']==base.sha(raw['kernel-metadata.json']) and
        inv['complete_diff_sha256']==base.sha(inputs.read(stage/'full-inventory-diff.txt')), 'full inventory proof')
    required=[stage/n for n in ('registration.json','source-manifest.json','embedded-file-manifest.json',
        'full-inventory-diff.txt','kernel/release.ipynb','kernel/kernel-metadata.json')]
    _,manifest_sha=base.manifest(stage,inputs,required)
    kernel_files={n:base.sha(v) for n,v in raw.items()}
    return dict(registration=reg,metadata=metadata,kernel_files=kernel_files,
        kernel_sha256=base.sha(base.encoded(kernel_files)),manifest_sha256=manifest_sha)


def native_dispatch(path, inputs, native_source_sha256):
    """Use the unchanged native verifier, then snapshot its COMPLETE read set.

    Execute only the registered helper bytes after verifying their digest. This
    keeps fixtures on the real verifier without importing an unchecked source
    or trusting a cached module. All native dependencies are standard library.
    """
    source = inputs.read(NATIVE)
    need(base.sha(source) == native_source_sha256, 'native verifier source changed before use')
    helper = types.ModuleType('k2_local_source_verified_native')
    helper.__file__ = str(inputs.root / NATIVE)
    exec(compile(source, helper.__file__, 'exec'), helper.__dict__)
    sidecar = base.local_path(path, inputs.root)
    raw = inputs.read(sidecar)
    record = helper.parse(raw)
    verified = helper.verify(sidecar, inputs.root)
    need(record == verified, 'native receipt changed during verification')
    archive = inputs.root / '.local/astra-improve/lanes/runs' / record['lane'] / record['run_id']
    completion_path = helper.local(record['completion_source'], inputs.root)
    completion_raw = inputs.read(completion_path)
    evidence = helper.parse(completion_raw)
    prompt_path = helper.local(evidence['prompt'], inputs.root)
    report_path = helper.local(evidence['report'], inputs.root)
    prompt, report = inputs.read(prompt_path), inputs.read(report_path)
    # Bind precisely the bytes read by native.verify; detect races before and
    # after snapshotting instead of merely trusting the helper's return value.
    need(inputs.read(archive / 'receipt.json') == raw and
         inputs.read(archive / 'completion.json') == completion_raw and
         base.sha(completion_raw) == record['completion_evidence_sha256'] and
         inputs.read(archive / 'prompt.txt') == prompt and base.sha(prompt) == record['prompt_sha256'] and
         inputs.read(archive / 'report.md') == report and base.sha(report) == record['report_sha256'] and
         base.sha(inputs.read(archive / 'run.log')) == record['log_sha256'], 'native snapshot differs')
    need(record['lane'] == evidence['lane'] and record['agent_id'] == evidence['agent_id'] and
         record['completed'] == evidence['observed_at'] and
         helper.local(record['out'], inputs.root) == report_path and
         sidecar == Path(str(report_path) + '.receipt.json'), 'native snapshot identity differs')
    log = helper.parse(inputs.read(archive / 'run.log'))
    need(all(log[key] == record[key] for key in ('host', 'model', 'agent_id', 'lane', 'note')) and
         log['recorder'] == 'native_codex_dispatch_record_v1' and log['version'] == helper.VERSION,
         'native log identity differs')
    completed = helper.dt.datetime.fromisoformat(record['completed'])
    recorded = helper.dt.datetime.fromisoformat(record['recorded_at'])
    need(completed.utcoffset() is not None and recorded.utcoffset() is not None and
         completed <= recorded <= helper.dt.datetime.now(helper.dt.timezone.utc) + helper.dt.timedelta(seconds=5),
         'native completion/recording chronology')
    # No arbitrary maximum evidence age: exact current registration/source
    # bindings and completion order, not wall-age, determine stale evidence.
    inputs.recheck()
    return record, report, report_path


def native_source(inputs):
    base.need(inputs.digest(b.NATIVE)==b.NATIVE_SHA,'exact approved native source')
    return b.NATIVE_SHA


def identity_lines(report,lines):
    text=report.decode('utf-8-sig') if isinstance(report,bytes) else report
    for line in lines:base.need(text.splitlines().count(line)==1,'missing/ambiguous attestation: '+line)
    for line in lines:
        if line.startswith('Kernel-SHA256: '):base.need(text.splitlines()[0]==line,'builder kernel hash must be first line')
    return text


def independent_review(path,pin_path,builder,inputs,label='K2-Pin-SHA256',phase='BUILD_READY'):
    path=base.local_path(path,inputs.root);text=base.verdict(inputs.read(path))
    record,report,actual=native_dispatch(str(path)+'.receipt.json',inputs,native_source(inputs))
    base.need(record['lane']!=builder['lane'] and record['agent_id']!=builder['agent_id'] and
        actual==path and report==inputs.read(path),'actual independent reviewer agent/lane')
    from datetime import datetime
    base.need(datetime.fromisoformat(record['completed'])>=datetime.fromisoformat(builder['completed']),'review before builder')
    identity_lines(text,['Reviewer-Lane: '+record['lane'],label+': '+inputs.digest(pin_path),
        'Candidate-ID: '+b.CANDIDATE,'Review-Phase: '+phase,'Kernel-ID: '+w.SLUG])
    return record


def prepare(builder_receipt,review,output,root=ROOT):
    inputs=base.Inputs(Path(root));m=snapshot(inputs.root/b.STAGE.relative_to(b.ROOT),inputs)
    builder,report,_=native_dispatch(builder_receipt,inputs,native_source(inputs))
    sm=inputs.root/b.STAGE.relative_to(b.ROOT)/'source-manifest.json'
    identity_lines(report,['Kernel-SHA256: '+m['kernel_sha256'],'K2-Implementation-Sources-SHA256: '+inputs.digest(sm)])
    reviewer=independent_review(review,sm,builder,inputs,'K2-Implementation-Sources-SHA256','IMPLEMENTATION')
    rec=dict(schema='k2-cpu-prepare/v1.0.2',provenance_contract=b.PROVENANCE_CONTRACT,
        stage=b.STAGE.relative_to(b.ROOT).as_posix(),candidate_id=b.CANDIDATE,fixture_only=False,
        builder_receipt=base.local_path(builder_receipt,inputs.root).relative_to(inputs.root).as_posix(),
        review=base.local_path(review,inputs.root).relative_to(inputs.root).as_posix(),
        builder_native=builder,reviewer_native=reviewer,
        outputs={part:(b.WORK.relative_to(b.ROOT)/'cpu-v1'/part).as_posix() for part in (*b.CPU_PARTS,'cached-reference-n32')},
        input_sha256=inputs.hashes(),executor='B',execution_authorized=False)
    inputs.recheck();b.exclusive_json(output,rec);return rec


def checked_prepare(path,inputs):
    rec=inputs.json(path)
    base.need(rec['schema']=='k2-cpu-prepare/v1.0.2' and rec['provenance_contract']==b.PROVENANCE_CONTRACT and
        rec['candidate_id']==b.CANDIDATE and rec['fixture_only'] is False and rec['executor']=='B','CPU prepare identity')
    for name,h in rec['input_sha256'].items():base.need(inputs.digest(name)==h,'CPU prepare drift')
    m=snapshot(rec['stage'],inputs)
    builder,report,_=native_dispatch(rec['builder_receipt'],inputs,native_source(inputs))
    sm=base.local_path(rec['stage'],inputs.root)/'source-manifest.json'
    identity_lines(report,['Kernel-SHA256: '+m['kernel_sha256'],'K2-Implementation-Sources-SHA256: '+inputs.digest(sm)])
    reviewer=independent_review(rec['review'],sm,builder,inputs,'K2-Implementation-Sources-SHA256','IMPLEMENTATION')
    base.need(builder==rec['builder_native'] and reviewer==rec['reviewer_native'],'CPU actual implementation review changed')
    expected={part:(b.WORK.relative_to(b.ROOT)/'cpu-v1'/part).as_posix() for part in (*b.CPU_PARTS,'cached-reference-n32')}
    base.need(rec['outputs']==expected,'exact fresh CPU output paths')
    inputs.recheck();return rec


def admit(prepared,output=None,dry_run=False,root=ROOT):
    from datetime import datetime,timezone
    inputs=base.Inputs(Path(root));rec=checked_prepare(prepared,inputs)
    base.need(all(not base.local_path(p,inputs.root).exists() for p in rec['outputs'].values()),'fresh CPU outputs required')
    result=dict(schema='k2-cpu-admission/v1.0.2',status='CPU_READY',prepared=dict(path=base.local_path(prepared,inputs.root).relative_to(inputs.root).as_posix(),sha256=inputs.digest(prepared)),
        candidate_id=b.CANDIDATE,provenance_contract=b.PROVENANCE_CONTRACT,executor='B',fixture_only=False,
        created_at=datetime.now(timezone.utc).isoformat(),input_sha256=inputs.hashes(),qualified_for_submission=False)
    inputs.recheck()
    if not dry_run:b.exclusive_json(output,result)
    return result


def checked_cpu_admission(path,inputs):
    rec=inputs.json(path)
    base.need(rec['schema']=='k2-cpu-admission/v1.0.2' and rec['status']=='CPU_READY' and rec['candidate_id']==b.CANDIDATE and
        rec['provenance_contract']==b.PROVENANCE_CONTRACT and rec['fixture_only'] is False and rec['executor']=='B' and
        rec['qualified_for_submission'] is False,'CPU admission schema')
    for name,h in rec['input_sha256'].items():base.need(inputs.digest(name)==h,'CPU admission drift')
    base.need(inputs.digest(rec['prepared']['path'])==rec['prepared']['sha256'],'CPU prepare changed')
    prepared=checked_prepare(rec['prepared']['path'],inputs);inputs.recheck();return rec,prepared


def replay_cpu_admission(path,part,destination):
    inputs=base.Inputs(ROOT);_,prepared=checked_cpu_admission(path,inputs)
    base.need(part in prepared['outputs'] and base.local_path(destination,inputs.root)==base.local_path(prepared['outputs'][part],inputs.root),
        'CPU part/output differs from reviewed preparation')
    base.need(not Path(destination).exists(),'CPU part output already exists')
    inputs.recheck()


def cpu_collection(rec,inputs,stage):
    from datetime import datetime,timezone,timedelta
    stage=base.local_path(stage,inputs.root);reg=inputs.json(stage/'registration.json')
    base.need(rec['schema']=='k2-cpu-collection/v1.0.2' and rec['provenance_contract']==b.PROVENANCE_CONTRACT and
        rec['status']=='PASS' and rec['qualified_for_submission'] is False and rec['raw_cached_images_exact'] is True and
        rec['registration_sha256']==inputs.digest(stage/'registration.json') and
        rec['manifest_sha256']==reg['asset_manifest_sha256'] and
        rec['builder_source_sha256']==inputs.digest('tools/l1_rco_k2_v102.py'),'fresh CPU collection identity')
    base.need(set(rec['parts'])==set(b.CPU_PARTS),'all five fresh CPU pieces required')
    parts={};run_ids=set();directories=set();admissions={}
    for name,p in rec['parts'].items():
        path=base.local_path(p['path'],inputs.root);parent=path.parent
        base.need(inputs.digest(path)==p['sha256'],'CPU receipt hash changed')
        part=inputs.json(path);parts[name]=part
        base.need(part['schema']=='k2-cpu-smoke/v1.0.2' and part['part']==name and part['status']=='PASS' and
            part['provenance_contract']==b.PROVENANCE_CONTRACT and part['qualified_for_submission'] is False and
            part['registration_sha256']==rec['registration_sha256'] and part['manifest_sha256']==reg['asset_manifest_sha256'] and
            part['builder_source_sha256']==rec['builder_source_sha256'] and part['worker_sha256']==b.WORKER_SHA and
            part['runtime_version']==w.VERSION and part['driver_sha256']==b.CPU_DLL_SHA and
            part['driver_path']==b.CPU_DLL_PATH,'fresh CPU part source/registration/asset/runtime/environment')
        base.need(part['environment']['device']=='cpu' and part['environment']['cuda_initialized'] is False and
            part['environment']['tf32'] is False and part['environment']['autocast'] is False and
            part['cuda_visible_devices']=='-1','CPU/no initialized CUDA required')
        base.need(re.fullmatch('[0-9a-f]{32}',part['run_id']) and part['run_id'] not in run_ids and
            type(part['pid']) is int and part['pid']>0 and parent not in directories,'distinct actual CPU runs')
        run_ids.add(part['run_id']);directories.add(parent)
        admission=part['cpu_admission'];base.need(inputs.digest(admission['path'])==admission['sha256'],'CPU run admission hash')
        # Reuse only within this one immutable Inputs snapshot. Each distinct
        # admission is fully replayed once; every consumed byte is still freshly
        # rehashed at collection completion and each external gate invocation.
        key=(str(base.local_path(admission['path'],inputs.root)),admission['sha256'])
        if key not in admissions:admissions[key]=checked_cpu_admission(admission['path'],inputs)
        ad,prepared=admissions[key]
        base.need(parent==base.local_path(prepared['outputs'][name],inputs.root) and
            part['output_directory']==parent.relative_to(inputs.root).as_posix(),'fresh prepared output identity')
        start=datetime.fromisoformat(part['started_at']);end=datetime.fromisoformat(part['completed_at'])
        created=datetime.fromisoformat(ad['created_at'])
        base.need(all(t.utcoffset() is not None for t in (start,end,created)) and
            created<=start<end<=datetime.now(timezone.utc)+timedelta(seconds=5),'fresh CPU chronology')
        att=inputs.json(parent/'execution-attestation.json')
        base.need(att['schema']=='k2-cpu-execution/v1.0.2' and att['recorded_by']=='B' and att['fixture_only'] is False and
            att['part']==name and att['run_id']==part['run_id'] and att['pid']==part['pid'] and att['exit_code']==0 and
            att['result']==dict(path=path.relative_to(inputs.root).as_posix(),sha256=inputs.digest(path)) and
            att['admission']==admission and att['observed_terminal_status']=='completed','B actual CPU execution attestation')
        observed=datetime.fromisoformat(att['observed_at'])
        base.need(observed.utcoffset() is not None and end<=observed<=datetime.now(timezone.utc)+timedelta(seconds=5),'CPU terminal observation chronology')
        if name=='model':base.need(part['initialization_plus_deploy_exact'] is True,'strict model initialization')
        if name=='spawn':base.need(part['workers']==[0,3] and part['studies']==2 and part['exact'] is True,'actual spawn workers3 parity')
        if name.startswith('raw-'):
            ids=reg['visible_ids'] if name=='raw-visible' else reg['smoke_holdout_ids']
            worker=inputs.json(parent/'worker-receipt.json');output=parent/'submission.csv'
            base.need(inputs.digest(parent/'worker-receipt.json')==part['worker_receipt_sha256'] and
                inputs.digest(output)==part['prediction_sha256']==worker['output_sha256'],'raw worker/output hashes')
            b.prediction_table(output,ids)
            base.need(worker['schema']=='k2-worker/v1' and worker['status']=='COMPLETE' and worker['version']==w.VERSION and
                worker['kernel_id']==w.SLUG and worker['contract']==w.CONTRACT and worker['manifest_sha256']==reg['asset_manifest_sha256'] and
                worker['roster_sha256']==w.digest(ids) and worker['study_count']==len(ids) and worker['prediction_count']==len(ids) and
                worker['fallbacks']['prediction']==0 and worker['fast_path'] is False and worker['placeholder'] is False and
                worker['environment']['device']=='cpu' and worker['environment']['cuda_initialized'] is False,'raw CPU runtime/contract/roster/no fallback')
            studies=worker['studies']
            base.need([s['index'] for s in studies]==list(range(len(ids))) and
                [s['uid_sha256'] for s in studies]==[hashlib.sha256(uid.encode()).hexdigest() for uid in ids] and
                all(s['windows']>0 and s['error'] is None and s['prediction_fallback']==0 for s in studies),'raw ordered study execution')
        if name=='cached-reference':
            base.need(inputs.digest(parent/'reference.csv')==part['reference_sha256'] and part['studies']==2,'cached reference output')
            values=b.prediction_table(parent/'reference.csv',reg['smoke_holdout_ids'])
            saved=inputs.root/'.local/scratch/l1-rco/k1v2/fit/holdout.csv';inputs.digest(saved)
            measured=b.parity(values,b.prediction_table(saved,reg['smoke_holdout_ids'],True),'k1_bf16')
            base.need(measured==part['k1_bf16'] and measured['status']=='PASS','fresh cached/saved-bf16 semantic parity')
    rawpath=base.local_path(rec['parts']['raw-holdout']['path'],inputs.root).parent
    refpath=base.local_path(rec['parts']['cached-reference']['path'],inputs.root).parent
    raw=inputs.json(rawpath/'worker-receipt.json');cached=parts['cached-reference']
    base.need([{k:s[k] for k in ('images_sha256','metadata_sha256','windows')} for s in raw['studies']]==cached['tensors'],'raw/cache tensor-window identity')
    comparison=b.parity(b.prediction_table(rawpath/'submission.csv',reg['smoke_holdout_ids']),b.prediction_table(refpath/'reference.csv',reg['smoke_holdout_ids']),'same_runtime')
    base.need(comparison==rec['same_runtime'] and comparison['status']=='PASS','same-runtime probability parity')
    inputs.recheck();return rec


def cpu_check(path,inputs,stage):
    return cpu_collection(inputs.json(path),inputs,stage)


def pin(stage,cpu,builder_receipt,output,root=ROOT):
    inputs=base.Inputs(Path(root));stage=base.local_path(stage,inputs.root)
    m=snapshot(stage,inputs);cpu_check(cpu,inputs,stage)
    builder,report,_=native_dispatch(builder_receipt,inputs,native_source(inputs))
    identity_lines(report,['Kernel-SHA256: '+m['kernel_sha256'],'K2-Registration-SHA256: '+inputs.digest(stage/'registration.json')])
    rec=dict(schema='k2-build-pin/v1.0.2',provenance_contract=b.PROVENANCE_CONTRACT,status='PINNED_NOT_ADMITTED',gate_version=VERSION,
        kernel_id=w.SLUG,stage=stage.relative_to(inputs.root).as_posix(),kernel_sha256=m['kernel_sha256'],manifest_sha256=m['manifest_sha256'],
        builder_lane=builder['lane'],builder_native=builder,builder_receipt=base.local_path(builder_receipt,inputs.root).relative_to(inputs.root).as_posix(),
        cpu=base.local_path(cpu,inputs.root).relative_to(inputs.root).as_posix(),input_sha256=inputs.hashes(),qualified_for_submission=False)
    inputs.recheck();b.exclusive_json(output,rec);return rec


def checked_pin(path,inputs):
    rec=inputs.json(path)
    base.need(rec['schema']=='k2-build-pin/v1.0.2' and rec['provenance_contract']==b.PROVENANCE_CONTRACT and rec['status']=='PINNED_NOT_ADMITTED' and
        rec['kernel_id']==w.SLUG and rec['gate_version']==VERSION,'new BUILD pin identity')
    for name,h in rec['input_sha256'].items():base.need(inputs.digest(name)==h,'BUILD pin source/evidence changed')
    stage=base.local_path(rec['stage'],inputs.root);m=snapshot(stage,inputs);cpu_check(rec['cpu'],inputs,stage)
    builder,report,_=native_dispatch(rec['builder_receipt'],inputs,native_source(inputs))
    identity_lines(report,['Kernel-SHA256: '+m['kernel_sha256'],'K2-Registration-SHA256: '+inputs.digest(stage/'registration.json')])
    base.need(builder==rec['builder_native'] and builder['lane']==rec['builder_lane'] and
        rec['kernel_sha256']==m['kernel_sha256'] and rec['manifest_sha256']==m['manifest_sha256'],'BUILD material/actual builder changed')
    inputs.recheck();return rec


def bind(pin_path,review,output,root=ROOT):
    inputs=base.Inputs(Path(root));pin=checked_pin(pin_path,inputs)
    reviewer=independent_review(review,pin_path,pin['builder_native'],inputs)
    rec=dict(schema='k2-gate/v1.0.2',provenance_contract=b.PROVENANCE_CONTRACT,status='PASS',errors=[],gate_version=VERSION,phase='BUILD_READY',
        qualified_for_submission=False,fixture_only=False,kernel_id=w.SLUG,stage=pin['stage'],kernel_sha256=pin['kernel_sha256'],
        pin=base.local_path(pin_path,inputs.root).relative_to(inputs.root).as_posix(),review=base.local_path(review,inputs.root).relative_to(inputs.root).as_posix(),
        reviewer_lane=reviewer['lane'],reviewer_native=reviewer,input_sha256=inputs.hashes())
    inputs.recheck();b.exclusive_json(output,rec);return rec


def verify_receipt(path,inputs,phase='FINAL_READY'):
    rec=inputs.json(path)
    base.need(phase in PHASES and rec['schema']=='k2-gate/v1.0.2' and rec['provenance_contract']==b.PROVENANCE_CONTRACT and rec['status']=='PASS' and
        rec['errors']==[] and rec['gate_version']==VERSION and rec['phase']==phase and rec['kernel_id']==w.SLUG and
        rec['qualified_for_submission'] is (phase=='FINAL_READY') and rec['fixture_only'] is False,'K2 phase not admitted')
    for name,h in rec['input_sha256'].items():base.need(inputs.digest(name)==h,'gate dependency changed')
    pin=checked_pin(rec['pin'],inputs);reviewer=independent_review(rec['review'],rec['pin'],pin['builder_native'],inputs)
    base.need(reviewer==rec['reviewer_native'] and reviewer['lane']==rec['reviewer_lane'] and rec['stage']==pin['stage'] and
        rec['kernel_sha256']==pin['kernel_sha256'],'BUILD gate replay')
    if phase=='FINAL_READY':
        from tools import l1_rco_k2_admission_v102 as admission
        admission.replay_final(rec,inputs,pin)
    inputs.recheck();return rec


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('prepare')
    for n in ('builder-receipt','review','output'):q.add_argument('--'+n,type=Path,required=True)
    q=s.add_parser('admit');q.add_argument('--prepared',type=Path,required=True);q.add_argument('--output',type=Path);q.add_argument('--dry-run',action='store_true')
    q=s.add_parser('pin')
    for n in ('stage','cpu','builder-receipt','output'):q.add_argument('--'+n,type=Path,required=True)
    q=s.add_parser('bind')
    for n in ('pin','review','output'):q.add_argument('--'+n,type=Path,required=True)
    for mode in ('check-build','check-final'):
        q=s.add_parser(mode);q.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.command=='prepare':r=prepare(args.builder_receipt,args.review,args.output)
        elif args.command=='admit':r=admit(args.prepared,args.output,args.dry_run)
        elif args.command=='pin':r=pin(args.stage,args.cpu,args.builder_receipt,args.output)
        elif args.command=='bind':r=bind(args.pin,args.review,args.output)
        else:r=verify_receipt(args.receipt,base.Inputs(ROOT),'BUILD_READY' if args.command=='check-build' else 'FINAL_READY')
        print(w.encoded({k:r[k] for k in ('status','kernel_id','qualified_for_submission') if k in r}).decode());return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(w.encoded(dict(status='BLOCKED',error=str(exc),qualified_for_submission=False)).decode());return 2


if __name__=='__main__':sys.exit(main())
