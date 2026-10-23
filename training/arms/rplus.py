"""R+ v1.0.0: immutable CONTROL43 continuations; existing contracts unchanged."""
from dataclasses import dataclass
from pathlib import Path
import copy
import shutil
from training.arms import exp190f as o, lifecycle as life

VERSION = '1.0.0'
ROOT, control = o.ROOT, o.control
BASE = 'artifacts/experiments/H_B_109_RPLUS/v1'
PARENT = o.CONTROL + 'model_best.pt'
OAI = o.OAI
SOURCES = (*o.SOURCES, 'training/arms/rplus.py', 'training/arms/rplus_fit.py',
           'training/arms/rplus_data.py', 'training/runners/rplus.py')
need, read, write, uid_hash = o.need, o.read, o.write, o.uid_hash


def scientific(basis, preset):
    d = copy.deepcopy(basis['config'])
    mixed = preset['arm'] == 'C-OAI'
    d['recipe'].update(epochs=3, swa_1based=[1,2,3], bb_lr=4.01e-6, head_lr=1.95e-4)
    d.update(epochs=3, total_updates=648, initial_state_sha256=preset['parent_state_sha256'],
             continuation=dict(optimizer='AdamW', optimizer_state='fresh', scheduler='constant',
                 epoch_offset=16, checkpoint=PARENT, checkpoint_sha256=preset['parent_sha256']),
             bn=dict(policy='all-mixed-training-once' if mixed else 'all-training-once',
                     count=4959 if mixed else 3455, epoch=19))
    if not mixed:
        d['supervision'] = dict(kind='L2-only', loss='fixed-12-BCE-study-mean',
                               sampling=dict(per_epoch=3455, rsna=3455, oai=0))
    return d


@dataclass(frozen=True)
class Spec(life.ArmSpec):
    def __post_init__(self):
        d = life.parse_json(self.encoded)
        need(d['schema']=='rplus-spec/v1' and d['family']=='coatnet-rplus', 'R+ schema differs')
        basis = o.Spec.from_document(d['basis'])
        b = basis.document
        need(set(d)==set(b)|{'basis','preset','preset_path'}, 'R+ fields differ')
        need(d['arm']==b['arm'], 'basis arm differs')
        need(d['arm'] in ('C-L2','C-OAI'), 'unknown continuation')
        for key in ('seed','fold','attempt','run_kind','target_environment','population_source','target_source','outputs'):
            need(d[key]==b[key], 'basis identity differs: '+key)
        need(d['preset_path']==BASE+'/'+d['arm']+'.json', 'preset path differs')
        need(d['config']==scientific(b,d['preset']), 'fixed continuation recipe differs')
        need(d['preset']['arm']==d['arm'] and d['preset']['version']==VERSION, 'preset identity differs')
        need(set(d['preset'])=={'schema','version','arm','controls','parent_sha256','parent_state_sha256'}
             and d['preset']['schema']=='rplus-preset/v1', 'preset fields differ')
        need(d['preset']['parent_sha256']=='f015c5ac90dc514f5612ff6a61c6c24ab4f86ae01a4dd0d7a9896a5cc0a983e1'
             and d['preset']['parent_state_sha256']=='b1e2b82f9c4482274c4695e86630f125b44296f93364449594ee71171e41b138',
             'CONTROL43 parent differs')
        need(set(d['inputs']['sources'])==set(SOURCES), 'R+ sources differ')
        expected = copy.deepcopy(b['inputs'])
        expected['sources'] = {k:d['inputs']['sources'][k] for k in SOURCES}
        need(all(expected['sources'][k]==v for k,v in b['inputs']['sources'].items()), 'basis sources differ')
        expected['data'].update({PARENT:d['preset']['parent_sha256'],
            d['preset_path']:life.content_sha256(d['preset'])})
        need(d['inputs']==expected, 'continuation input inventory differs')
        life._inventory(self.inventory)
        life.validate_bindings(self.bindings)
        object.__setattr__(self,'encoded',life.canonical_bytes(d))


def population(spec, root):
    d = spec.document
    p = o.population(o.Spec.from_document(d['basis']), root)
    if d['arm']=='C-L2':
        p['train_ids'] = p['rsna_train_ids']
        p['oai_train_ids'] = []
        p['train_labels'] = {u:p['train_labels'][u] for u in p['train_ids']}
        p['train_masks'] = {u:[1]*12 for u in p['train_ids']}
    return p


def validate_inputs(spec, root):
    life._verify_files(root,spec.inventory)
    d = spec.document
    o.validate_inputs(o.Spec.from_document(d['basis']), root)
    need(read(Path(root)/d['preset_path'])==d['preset'], 'preset bytes differ')
    summary = read(Path(root)/o.CONTROL/'summary.json')
    need(summary['final_state_sha256']==d['preset']['parent_state_sha256'], 'parent final state differs')
    need(d['preset']['controls']==d['inputs']['controls'], 'parent controls differ')
    return population(spec,root)


def register(root, *, controls, arm, attempt):
    need(arm in ('C-L2','C-OAI'), 'unknown continuation')
    basis = o.register(root,controls=controls,arm=arm,attempt=attempt)
    d = basis.document
    preset_path = BASE+'/'+arm+'.json'
    preset = read(Path(root)/preset_path)
    d.update(schema='rplus-spec/v1',family='coatnet-rplus',basis=basis.document,
             preset=preset,preset_path=preset_path,config=scientific(basis.document,preset))
    d['inputs']['sources'] = {k:life.file_sha256(Path(root)/k) for k in SOURCES}
    d['inputs']['data'].update({PARENT:life.file_sha256(Path(root)/PARENT),
                              preset_path:life.file_sha256(Path(root)/preset_path)})
    spec = Spec.from_document(d)
    validate_inputs(spec,root)
    return spec


def validate_stage(spec, stage):
    need(life._files(stage)==set(spec.inventory)|life.STAGE_METADATA,'stage inventory differs')
    validate_inputs(spec,stage)
    need((Path(stage)/'arm-spec.json').read_bytes()==spec.encoded,'staged spec differs')
    expected=dict(schema='rplus-stage/v1',spec_sha256=spec.sha256,inputs=spec.inventory,status='STAGED_NO_EXECUTION')
    need(read(Path(stage)/'stage-receipt.json')==expected,'stage receipt differs')
    return expected


def prepare(spec,root,destination):
    if Path(destination).exists():
        return validate_stage(spec,destination)
    validate_inputs(spec,root)
    work=life._working_directory(destination)
    for name in spec.inventory:
        target=life._path(work,name)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(life._path(root,name),target)
    life.create_new(work/'arm-spec.json',spec.encoded)
    write(work/'stage-receipt.json',dict(schema='rplus-stage/v1',spec_sha256=spec.sha256,
        inputs=spec.inventory,status='STAGED_NO_EXECUTION'))
    result=validate_stage(spec,work)
    life._publish(work,destination)
    return result


def admit(spec,stage,note,evidence_root,observation=None):
    validate_stage(spec,stage)
    digest=life.verify_evidence(spec,note,evidence_root)
    if observation is not None:
        life.require_verification_pass(note,**spec.verification_args)
        life.check_target_observation(spec,observation)
    return dict(status='EVIDENCE_READY_TARGET_CHECK_REQUIRED' if note['verdict']=='PASS' else 'HOLD',
        launch_authorized=False,spec_sha256=spec.sha256,verification_sha256=digest,blockers=note['blockers'])
