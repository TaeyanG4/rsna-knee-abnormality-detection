"""R+ v1.1.0: labels-only Dread continuation paired with frozen C-L2 v1.

The embedded C-L2 contract fixes every non-label scientific setting. Existing
v1 sources and registrations are not rewritten by this additive amendment.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import shutil
import numpy as np
import pandas as pd
from training.arms import rplus as original, lifecycle as life

VERSION = '1.1.0'
ROOT, control, o, OAI = original.ROOT, original.control, original.o, original.OAI
BASE = 'artifacts/experiments/H_B_109_RPLUS/v1.1.0'
PARENT = original.PARENT
LABELS = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
LABEL_SHA = '8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1'
SOURCES = (*original.SOURCES, 'training/arms/rplus_base.py',
           'training/arms/rplus_base_fit.py', 'training/runners/rplus_base.py')
need, read, write, uid_hash = original.need, original.read, original.write, original.uid_hash


def preset_for(l2):
    p = copy.deepcopy(l2['preset'])
    p.update(schema='rplus-base-preset/v1', version=VERSION, arm='C-BASE',
             labels=LABELS, labels_sha256=LABEL_SHA)
    return p


def scientific(l2):
    cfg = copy.deepcopy(l2['config'])
    cfg['recipe']['labels'] = 'dread'
    cfg['supervision']['kind'] = 'Dread-only'
    return cfg


def document(l2, sources):
    d = copy.deepcopy(l2)
    p = preset_for(l2)
    path = BASE+'/C-BASE.json'
    d.update(schema='rplus-base-spec/v1', family='coatnet-rplus', arm='C-BASE',
             l2_basis=l2, preset=p, preset_path=path, config=scientific(l2))
    d['inputs']['sources'] = sources
    d['inputs']['data'].update({LABELS:LABEL_SHA, path:life.content_sha256(p)})
    return d


@dataclass(frozen=True)
class Spec(life.ArmSpec):
    def __post_init__(self):
        d = life.parse_json(self.encoded)
        l2 = original.Spec.from_document(d['l2_basis']).document
        need(l2['arm']=='C-L2', 'C-BASE requires C-L2 basis')
        sources = d['inputs']['sources']
        need(set(sources)==set(SOURCES), 'C-BASE sources differ')
        need(all(sources[k]==v for k,v in l2['inputs']['sources'].items()), 'C-L2 sources differ')
        need(d==document(l2,sources), 'labels-only continuation contract differs')
        life._inventory(self.inventory)
        life.validate_bindings(self.bindings)
        object.__setattr__(self,'encoded',life.canonical_bytes(d))


def load_labels(path, ids, targets):
    """Same pinned source and float32 target selection as S6+ S-B; no rows logged."""
    frame = pd.read_csv(path, usecols=['StudyInstanceUID',*targets],
                        dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
    need(not frame.index.has_duplicates and set(ids)<=set(frame.index), 'baseline label coverage differs')
    values = frame.loc[ids,targets].to_numpy('float32')
    need(values.shape==(len(ids),12) and np.isfinite(values).all()
         and ((values>=0)&(values<=1)).all(), 'baseline label values invalid')
    return dict(zip(ids,values.tolist()))


def population(spec, root):
    d = spec.document
    p = original.population(original.Spec.from_document(d['l2_basis']),root)
    need(life.file_sha256(Path(root)/LABELS)==LABEL_SHA, 'baseline label bytes differ')
    # Deliberately retain C-L2 pos_weight and held-out labels. Recomputing either
    # would confound the labels-only training comparison.
    p['train_labels'] = load_labels(Path(root)/LABELS,p['train_ids'],d['config']['targets'])
    return p


def validate_inputs(spec, root):
    life._verify_files(root,spec.inventory)
    d = spec.document
    original.validate_inputs(original.Spec.from_document(d['l2_basis']),root)
    need(read(Path(root)/d['preset_path'])==d['preset'], 'preset bytes differ')
    return population(spec,root)


def register(root, *, controls, arm, attempt):
    need(arm=='C-BASE', 'new adapter registers C-BASE only')
    l2 = original.register(root,controls=controls,arm='C-L2',attempt=attempt)
    sources = {k:life.file_sha256(Path(root)/k) for k in SOURCES}
    spec = Spec.from_document(document(l2.document,sources))
    validate_inputs(spec,root)
    return spec


def validate_stage(spec, stage):
    need(life._files(stage)==set(spec.inventory)|life.STAGE_METADATA,'stage inventory differs')
    validate_inputs(spec,stage)
    need((Path(stage)/'arm-spec.json').read_bytes()==spec.encoded,'staged spec differs')
    expected=dict(schema='rplus-base-stage/v1',spec_sha256=spec.sha256,inputs=spec.inventory,status='STAGED_NO_EXECUTION')
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
    write(work/'stage-receipt.json',dict(schema='rplus-base-stage/v1',spec_sha256=spec.sha256,
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
