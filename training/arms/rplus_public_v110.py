"""H-B-113 v1.1.0: isolated seed44 public-R all-roster L2 member; no fold claim.

Changelog: seed44 identity and source closure only; frozen v1.0.0 recipe reused.
No original contract or serving source is modified.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
import numpy as np
from training.arms import lifecycle as life, coatnet as c, coatnet_lifecycle as cl, rplus
from tools.exp099_kfold import pos_weight
from training.arms import rplus_public as legacy

VERSION = '1.1.0'
ROOT = Path(__file__).resolve().parents[2]
BASE = 'artifacts/experiments/H_B_109_RPLUS_PUBLIC/v1.1.0'
PARENT = '.local/rplus-v1/public-R-download.bin'
PUBLIC_SHA = '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
SCOPE = 'PRODUCTION_PUBLIC_R_ANCESTRY'
FULL = cl.FULL
TARGET = legacy.TARGET
DESCRIPTOR = legacy.DESCRIPTOR
PREVIOUS = legacy.BASE+'/member-spec.json'
PREVIOUS_SHA = 'f2774d3564a1a220540f8ace25d76c9173f56ce99336c021a1f0b6f664441a7a'
DECISION_INPUTS = (PREVIOUS,
    '.local/astra-improve/builds/rplus-decision-1440-out.md',
    'artifacts/experiments/H_B_109_RPLUS/v1/PRODUCTION.md',
    'artifacts/experiments/H_B_109_RPLUS/v1.1.0/PRODUCTION.md')
OUTPUTS = cl.OUTPUTS
SOURCES = tuple(dict.fromkeys((*rplus.SOURCES, *cl.SOURCES,
    'training/arms/rplus_public.py', 'training/arms/rplus_public_fit.py',
    'training/runners/rplus_public.py', 'tools/rplus_public_cpu_card.py',
    'training/arms/rplus_public_v110.py', 'training/arms/rplus_public_fit_v110.py',
    'training/runners/rplus_public_v110.py', 'tools/rplus_public_cpu_card_v110.py')))
need, read, write, uid_hash = life._need, cl.read, cl.write, cl.uid_hash


def scientific(p):
    weights = pos_weight({u:np.asarray(p['train_labels'][u],np.float32) for u in p['train_ids']},p['train_ids']).tolist()
    return fixed_config(uid_hash(p['train_ids']),weights)


def fixed_config(train_sha256,weights):
    return dict(legacy.fixed_config(train_sha256, weights), seed=44)


@dataclass(frozen=True)
class Spec(life.ArmSpec):
    def __post_init__(self):
        d=life.parse_json(self.encoded)
        life._keys(d,{'schema','family','run_kind','arm','attempt','seed','fold','config','inputs',
                     'target_environment','outputs','population_source','target_source'},'public member')
        need((d['schema'],d['family'],d['run_kind'],d['arm'],d['attempt'],d['seed'],d['fold']) ==
             ('rplus-public-spec/v1','coatnet-rplus-public','member','PUBLIC-R-L2','local4070-s44-v110',44,-1),'member identity differs')
        need(d['population_source']==FULL and d['target_source']==TARGET,'member sources differ')
        life._keys(d['inputs'],{'sources','data','rosters','controls','evidence'},'input groups')
        need(d['inputs']['controls']=={},'public member cannot carry paired controls')
        need(set(d['inputs']['sources'])==set(SOURCES),'source inventory differs')
        need(set(d['inputs']['data'])=={DESCRIPTOR,c.CORPUS+'/manifest.json'},'data inventory differs')
        need(d['inputs']['rosters']=={FULL:cl.POPULATION_PINS[FULL]},'full population identity differs')
        need(set(d['inputs']['evidence'])=={TARGET,*DECISION_INPUTS},'evidence inventory differs')
        need(d['outputs']==OUTPUTS,'outputs differ')
        need(d['inputs']['evidence'][PREVIOUS]==PREVIOUS_SHA,'seed43 registration differs')
        need(sum(len(g) for g in d['inputs'].values())==len(self.inventory),'duplicate input path')
        life._inventory(self.inventory)
        need(not set(self.inventory)&life.STAGE_METADATA,'reserved stage path')
        life.validate_environment(d['target_environment'])
        need(d['target_environment']['schema']==life.MEMBER_ENVIRONMENT_SCHEMA,'member environment required')
        # Complete numeric/scientific identity is recomputed from authenticated roster at validation.
        cfg=d['config']
        need(cfg==fixed_config(cfg['train_sha256'],cfg['pos_weight']),'fixed scientific recipe differs')
        life.validate_bindings(self.bindings)
        object.__setattr__(self,'encoded',life.canonical_bytes(d))


def population(spec,root):
    p=read(life._path(root,FULL))
    ids,labels=p['train_ids'],p['train_labels']
    need(len(ids)==4347 and len(set(ids))==4347 and set(labels)==set(ids),'full roster differs')
    need(len(set(p['excluded_gold_ids']))==58 and not set(ids)&set(p['excluded_gold_ids']),'Gold58 exclusion differs')
    values=np.asarray([labels[u] for u in ids],dtype=np.float32)
    need(values.shape==(4347,12) and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),'invalid labels')
    expected=scientific(p)
    prior=legacy.Spec((Path(root)/PREVIOUS).read_bytes()).document
    need(dict(expected,seed=43)==prior['config'],'seed-only recipe drift')
    need(spec.document['config']==expected,'fixed scientific recipe differs')
    need(expected['pos_weight']==p['pos_weight'],'full roster class weights differ from registered recipe')
    return dict(p,pos_weight=expected['pos_weight'])


def validate_inputs(spec,root):
    life._verify_files(root,spec.inventory)
    p=population(spec,root)
    d=spec.document
    need(d['inputs']['data'][c.CORPUS+'/manifest.json']==p['corpus_manifest_sha256'],'corpus binding differs')
    need(read(Path(root)/DESCRIPTOR)==dict(path=PARENT,sha256=PUBLIC_SHA,scope=SCOPE,mount='external-read-only-no-copy'),'parent descriptor differs')
    profile,declaration=life.environment_from_declaration(Path(root)/TARGET,expected_sha256=d['inputs']['evidence'][TARGET])
    need(profile==d['target_environment'],'target profile differs')
    life.check_target_observation(spec,declaration)
    return p


def register(root):
    root=Path(root)
    need(life.file_sha256(root/PARENT)==PUBLIC_SHA,'public parent bytes differ')
    groups=dict(sources={f:life.file_sha256(root/f) for f in SOURCES},
        data={f:life.file_sha256(root/f) for f in (DESCRIPTOR,c.CORPUS+'/manifest.json')},
        rosters={FULL:life.file_sha256(root/FULL)},controls={},
        evidence={f:life.file_sha256(root/f) for f in (TARGET,*DECISION_INPUTS)})
    profile,_=life.environment_from_declaration(root/TARGET,expected_sha256=groups['evidence'][TARGET])
    spec=Spec.from_document(dict(schema='rplus-public-spec/v1',family='coatnet-rplus-public',run_kind='member',
        arm='PUBLIC-R-L2',attempt='local4070-s44-v110',seed=44,fold=-1,population_source=FULL,target_source=TARGET,
        config=scientific(read(root/FULL)),inputs=groups,target_environment=profile,outputs=OUTPUTS))
    validate_inputs(spec,root)
    return spec


def validate_stage(spec,stage):
    need(life._files(stage)==set(spec.inventory)|life.STAGE_METADATA,'stage inventory differs')
    validate_inputs(spec,stage)
    need((Path(stage)/'arm-spec.json').read_bytes()==spec.encoded,'staged spec differs')
    expected=dict(schema='rplus-public-stage/v1',spec_sha256=spec.sha256,inputs=spec.inventory,status='STAGED_NO_EXECUTION')
    need(read(Path(stage)/'stage-receipt.json')==expected,'stage receipt differs')
    return expected


def prepare(spec,root,destination):
    if Path(destination).exists(): return validate_stage(spec,destination)
    validate_inputs(spec,root)
    # Fixed sibling staging area; retain any incomplete preparation for inspection.
    work=Path(str(destination)+'.partial')
    work.mkdir(parents=True,exist_ok=False)
    for name in spec.inventory:
        target=life._path(work,name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(life._path(root,name),target)
    life.create_new(work/'arm-spec.json',spec.encoded)
    write(work/'stage-receipt.json',dict(schema='rplus-public-stage/v1',spec_sha256=spec.sha256,inputs=spec.inventory,status='STAGED_NO_EXECUTION'))
    result=validate_stage(spec,work)
    life._publish(work,destination)
    return result


def require_pre_cuda_review(spec,review,evidence_root):
    """Independent human/agent review attestation; checked BEFORE CUDA observation.

    This is distinct from the final PASS requiring both CPU and target smokes.
    Authenticity of reviewer identity is a workflow trust boundary, not a signature.
    """
    need(review is not None and evidence_root is not None,'independent pre-CUDA review required')
    life._keys(review,{'schema','spec_sha256','scope','reviewer','independent','verdict','blockers',
                      'report','cpu_note','h112_precedence'},'pre-CUDA review')
    need(review['schema']=='rplus-public-pre-cuda/v1' and review['scope']==SCOPE and review['spec_sha256']==spec.sha256,'review binding differs')
    need(review['independent'] is True and review['verdict']=='PASS' and review['blockers']==[],'independent review must pass')
    life._keys(review['reviewer'],{'name','host'},'reviewer')
    for v in review['reviewer'].values(): life._text(v,'reviewer')
    need(review['reviewer']['name'] not in ('H-B-113-builder','executed-coatnet-smoke-producer'),'builder cannot self-review')
    for field in ('report','cpu_note'):
        life._keys(review[field],{'path','sha256'},field)
        life._verify_files(evidence_root,{review[field]['path']:review[field]['sha256']})
    cpu_path=life._path(evidence_root,review['cpu_note']['path'])
    note=read(cpu_path)
    life.verify_evidence(spec,note,cpu_path.parent)
    cpu=note['smokes']['cpu']
    need(cpu is not None and cpu['exit_code']==0
         and all(cpu['results'][k] for k in life.SMOKE_FLAGS)
         and all(cpu['results'][k]>0 for k in life.SMOKE_COUNTS), 'real CPU training evidence required')
    need(review['h112_precedence'] in ('completed-before-H113','not-admitted-B-confirmed'),'H112 precedence unresolved')
    return life.content_sha256(review)


def admit(spec,stage,note,evidence_root,review=None,review_root=None):
    validate_stage(spec,stage)
    digest=life.verify_evidence(spec,note,evidence_root)
    blockers=list(note['blockers'])
    if review is None: blockers.append('Independent pre-CUDA review and B H112 precedence confirmation required')
    else: require_pre_cuda_review(spec,review,review_root)
    return dict(status='HOLD' if blockers else 'EVIDENCE_READY_TARGET_CHECK_REQUIRED',launch_authorized=False,
                spec_sha256=spec.sha256,verification_sha256=digest,blockers=blockers)
