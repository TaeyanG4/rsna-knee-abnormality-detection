"""R+ v1.2.1: isolated fold-1, labels-only continuation contracts.

Parent binding copies external outputs into a portable, create-only input root.
Engineering bindings are permanently barred from full training and science.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import shutil
from training.arms import coatnet_fold1 as parent, lifecycle as life
from training.arms.rplus_base import load_labels, LABELS, LABEL_SHA

VERSION = '1.2.1'
ROOT = Path(__file__).resolve().parents[2]
BASE = 'artifacts/experiments/H_B_109_RPLUS/v1.2.1'
PARENT = 'parent/outputs/model_best.pt'
SOURCES = tuple(dict.fromkeys((*parent.SOURCES,
    'training/arms/rplus_f1.py', 'training/arms/rplus_f1_fit.py',
    'training/arms/rplus_data.py', 'training/arms/rplus_base.py',
    'training/arms/rplus.py', 'training/arms/exp190f.py',
    'training/arms/exp190f_data.py', 'training/runners/rplus_f1.py',
    * __import__('training.arms.rplus', fromlist=['SOURCES']).SOURCES)))
need, read, write, uid_hash = parent.need, parent.read, parent.write, parent.uid_hash


def continuation_ids(p):
    # Label-independent deterministic budget match. Preserve source order.
    import hashlib
    selected = set(sorted(p['train_ids'], key=lambda u: hashlib.sha256(
        ('RPLUS-F1-43:'+u).encode()).hexdigest())[:3455])
    return [u for u in p['train_ids'] if u in selected]


def scientific(basis, p, arm, checkpoint_sha, state_sha):
    cfg = copy.deepcopy(basis['config'])
    ids = continuation_ids(p)
    cfg['recipe'].update(epochs=3, swa_1based=[1,2,3], bb_lr=4.01e-6,
                         head_lr=1.95e-4, labels='dread' if arm=='C-BASE-f1' else 'registered-replacement-soft')
    cfg.update(epochs=3, train_count=3455, train_sha256=uid_hash(ids),
        updates_per_epoch=216, total_updates=648, initial_state_sha256=state_sha,
        continuation=dict(optimizer='AdamW', optimizer_state='fresh', scheduler='constant',
            epoch_offset=16, checkpoint=PARENT, checkpoint_sha256=checkpoint_sha),
        bn=dict(policy='all-training-once', count=3455, epoch=19),
        supervision=dict(kind='Dread-only' if arm=='C-BASE-f1' else 'L2-only',
            loss='fixed-12-BCE-study-mean', sampling=dict(per_epoch=3455,rsna=3455,oai=0)),
        budget_selection='lowest-sha256-RPLUS-F1-43-UID-3455-preserve-parent-order')
    return cfg


def check_parent(spec, root, engineering):
    p = parent.validate_inputs(spec, root)
    d = spec.document
    need(d['seed']==43 and d['fold']==1 and d['config']['variant']=='control'
         and d['config']['checkpointing'] is False, 'CONTROL43 fold1 parent required')
    summary, prov = [read(Path(root)/'parent/outputs'/n) for n in ('summary.json','provenance.json')]
    need(prov['spec_sha256']==spec.sha256 and prov['scientific']==d['config'], 'parent provenance differs')
    if engineering:
        need(summary['status']=='ENGINEERING_SMOKE_COMPLETE' and summary['mode']=='smoke'
             and summary['updates']==1 and summary['cpu_cuda_initialized'] is False, 'CPU engineering parent required')
    else:
        need(summary['status']=='COMPLETE_TRAINING_AND_READOUT' and summary['mode']=='full'
             and summary['updates']==3488 and summary['skips']==0 and prov['verification_sha256'] is not None,
             'completed full fold1 parent required')
        need(summary['bn']['studies']==3479 and summary['bn']['unique_studies']==3479, 'parent BN differs')
        life.check_target_observation(spec, summary['observation'])
        for view in ('plain','mirror','mean'):
            life.load_prediction_csv(Path(root)/f'parent/outputs/predictions-{view}.csv',p['val_ids'],d['config']['targets'])
    return p, summary


def bind(root, parent_spec, outputs, destination, engineering=False):
    root, outputs, destination = Path(root), Path(outputs), Path(destination)
    spec = parent.Spec(Path(parent_spec).read_bytes())
    parent.validate_inputs(spec,root)
    work = life._working_directory(destination)
    inventory = dict(spec.inventory)
    inventory.update({k:life.file_sha256(root/k) for k in SOURCES})
    inventory[LABELS] = LABEL_SHA
    life._verify_files(root,inventory)
    for name in inventory:
        target = work/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    life.create_new(work/'parent/spec.json',spec.encoded)
    for name in parent.OUTPUTS:
        target=work/'parent/outputs'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(outputs/name,target)
    check_parent(spec,work,engineering)
    write(work/'parent/binding.json',dict(engineering=engineering,parent_spec_sha256=spec.sha256))
    life._publish(work,destination)
    return dict(status='PARENT_BOUND',engineering=engineering,parent_spec_sha256=spec.sha256)


def build_document(root, arm, attempt):
    root=Path(root)
    basis=parent.Spec((root/'parent/spec.json').read_bytes())
    binding=read(root/'parent/binding.json')
    need(type(binding['engineering']) is bool and binding['parent_spec_sha256']==basis.sha256,'binding differs')
    p, summary=check_parent(basis,root,binding['engineering'])
    d=basis.document
    d.update(schema='rplus-f1-spec/v1',family='coatnet-rplus-f1',arm=arm,attempt=attempt,
        parent_basis=basis.document,engineering=binding['engineering'])
    d['config']=scientific(basis.document,p,arm,life.file_sha256(root/PARENT),summary['final_state_sha256'])
    d['inputs']['sources']={k:life.file_sha256(root/k) for k in SOURCES}
    d['inputs']['data'].update({k:life.file_sha256(root/k) for k in
        ['parent/spec.json','parent/binding.json',*[f'parent/outputs/{n}' for n in parent.OUTPUTS],LABELS]})
    return d


@dataclass(frozen=True)
class Spec(life.ArmSpec):
    def __post_init__(self):
        d=life.parse_json(self.encoded)
        need(d['schema']=='rplus-f1-spec/v1' and d['family']=='coatnet-rplus-f1','schema differs')
        need(d['arm'] in ('C-L2-f1','C-BASE-f1') and type(d['engineering']) is bool,'arm/scope differs')
        parent.Spec.from_document(d['parent_basis'])
        for name in ('arm','attempt'): need('/' not in life.safe_member(d[name]),'identity differs')
        life._inventory(self.inventory);life.validate_bindings(self.bindings)
        object.__setattr__(self,'encoded',life.canonical_bytes(d))


def population(spec,root):
    d=spec.document
    p=parent.population(parent.Spec.from_document(d['parent_basis']),root)
    ids=continuation_ids(p)
    p.update(train_ids=ids,rsna_train_ids=ids,oai_train_ids=[],oai_exclusion={'scope':'NO_OAI'},
             train_labels={u:p['train_labels'][u] for u in ids},train_masks={u:[1]*12 for u in ids})
    if d['arm']=='C-BASE-f1':
        need(life.file_sha256(Path(root)/LABELS)==LABEL_SHA,'Dread bytes differ')
        p['train_labels']=load_labels(Path(root)/LABELS,ids,d['config']['targets'])
    return p


def validate_inputs(spec,root):
    life._verify_files(root,spec.inventory)
    d=spec.document
    need(d==build_document(root,d['arm'],d['attempt']),'fixed fold1 contract differs')
    return population(spec,root)


def register(root,*,arm,attempt):
    need(arm in ('C-L2-f1','C-BASE-f1'),'unknown fold1 arm')
    spec=Spec.from_document(build_document(root,arm,attempt))
    validate_inputs(spec,root)
    return spec


def validate_stage(spec,stage):
    need(life._files(stage)==set(spec.inventory)|life.STAGE_METADATA,'stage inventory differs')
    validate_inputs(spec,stage)
    need((Path(stage)/'arm-spec.json').read_bytes()==spec.encoded,'staged spec differs')
    expected=dict(schema='rplus-f1-stage/v1',spec_sha256=spec.sha256,inputs=spec.inventory,status='STAGED_NO_EXECUTION')
    need(read(Path(stage)/'stage-receipt.json')==expected,'stage receipt differs')
    return expected


def prepare(spec,root,destination):
    if Path(destination).exists(): return validate_stage(spec,destination)
    validate_inputs(spec,root)
    work=life._working_directory(destination)
    for name in spec.inventory:
        target=work/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(root)/name,target)
    life.create_new(work/'arm-spec.json',spec.encoded)
    write(work/'stage-receipt.json',dict(schema='rplus-f1-stage/v1',spec_sha256=spec.sha256,inputs=spec.inventory,status='STAGED_NO_EXECUTION'))
    result=validate_stage(spec,work);life._publish(work,destination)
    return result


def admit(spec,stage,note,evidence_root,observation=None):
    validate_stage(spec,stage)
    if spec.document['engineering']:
        need(note['verdict']!='PASS' and observation is None,'engineering binding cannot admit full training')
    digest=life.verify_evidence(spec,note,evidence_root)
    if observation is not None:
        life.require_verification_pass(note,**spec.verification_args)
        life.check_target_observation(spec,observation)
    return dict(status='EVIDENCE_READY_TARGET_CHECK_REQUIRED' if note['verdict']=='PASS' else 'HOLD',
        launch_authorized=False,spec_sha256=spec.sha256,verification_sha256=digest,blockers=note['blockers'])
