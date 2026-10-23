"""EXP190F lifecycle v1.0.0: CONTROL43 plus corrected, panel-excluded OAI.

Changelog: isolated experiment contract; no changes to panel/release sources.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import re
import shutil
import csv
import io

from training.arms import coatnet_lifecycle as control
from training.arms import lifecycle as life
from training.arms import oai_exclusions as exclusions
from tools import exp155_oai_labels_v2 as mapping

VERSION = '1.0.0'
ROOT = Path(__file__).resolve().parents[2]
BASE = 'artifacts/experiments/EXP190F_COATNET_OAI_F0/lifecycle-v1'
PRESET = BASE + '/preset.json'
COHORT = BASE + '/cohort.json'
PANEL = 'artifacts/experiments/OAI_MOAKS_PANEL_V1/panel.csv'
OAI = 'storage/cache/derived/EXP155_OAI_RAPTOR96_384_C140_S02_v2'
CACHE_PINS = {OAI+'/manifest.json': 'da3509c79039e101573dc58cea7aa9e5e5f762b66478d36fd1b52df1cdbeed21',
              OAI+'/build-registration.json': 'b95022ef07f7f637ed3319b42e15a2679986016bfd25f6aa818bc366bfedecec'}
CONTROL = '.local/refactor/qual-4070-control-fold0-s43-8594f460d5d24d09a9f0f3f6796ab082/train/outputs/'
EXTRA_SOURCES = ('training/arms/exp190f.py', 'training/arms/exp190f_data.py',
    'training/arms/exp190f_fit.py', 'training/runners/exp190f.py',
    'training/arms/oai_exclusions.py', 'training/src/exp155_own0949_oai_v11.py',
    'tools/exp155_oai_labels_v2.py', 'tools/extlabels.py', 'tools/exp155_oai_data.py',
    'tools/exp143_v2.py')
SOURCES = tuple(dict.fromkeys((*control.SOURCES, *EXTRA_SOURCES)))
need, read, write = life._need, control.read, control.write
uid_hash = control.uid_hash


def build_cohort(root):
    """Read authenticated numeric clinical tables; never emit label rows to stdout."""
    root = Path(root)
    life._verify_files(root, CACHE_PINS)
    manifest = read(root/OAI/'manifest.json')
    registration = read(root/OAI/'build-registration.json')
    life._verify_files(root, registration['code_sha256'])
    need(manifest['build_registration_sha256'] == CACHE_PINS[OAI+'/build-registration.json'], 'cache build differs')
    policy = exclusions.bind(root/PANEL)
    allowed = set(exclusions.filter_ids(manifest['ids'], policy))
    inputs = mapping.clinical_inputs(mapping.DEFAULT_RECEIPT)
    primary, fallback = mapping.select_sources(inputs)
    labels, masks, ids = {}, {}, []
    for uid in manifest['ids']:
        need(re.fullmatch(r'OAI_[0-9]{7}_[LR]_V00', uid) is not None, 'nonbaseline OAI study')
        if uid not in allowed:
            continue
        m = primary[uid].scores if uid in primary else None
        f = fallback[uid].scores if uid in fallback else None
        values = {t: mapping.oa_v2(m, t) for t in ('PF OA', 'Lateral OA')}
        values['Synovitis'], _ = mapping.synovitis_v2(m, f)
        if not any(v is not None for v in values.values()):
            continue
        ids.append(uid)
        labels[uid] = [values.get(t) if values.get(t) is not None else 0 for t in control.LAB]
        masks[uid] = [int(values.get(t) is not None) for t in control.LAB]
    need(len(ids) == 1504 and policy['participant_count'] == 557, 'OAI admission counts differ')
    result = dict(schema='exp190f-cohort/v1', ids=ids, labels=labels, masks=masks, exclusion=policy,
        clinical_receipt_sha256=life.file_sha256(mapping.DEFAULT_RECEIPT),
        clinical_inputs_sha256={n: h for n, (_, h) in inputs.items()},
        mapping_sha256=life.file_sha256(root/'tools/exp155_oai_labels_v2.py'))
    write(root/COHORT, result)
    return dict(status='COHORT_BUILT', studies=len(ids), withheld_participants=557,
                sha256=life.file_sha256(root/COHORT))


def projection(document):
    d = copy.deepcopy(document)
    d['schema'], d['family'] = 'coatnet-spec/v1', 'coatnet'
    d['config'].pop('supervision')
    d['config']['bn'] = dict(policy='all-training-once', count=3455, epoch=16)
    d['inputs']['sources'] = {k: d['inputs']['sources'][k] for k in control.SOURCES}
    d['inputs']['data'] = {k: d['inputs']['data'][k] for k in
                          (control.c.INIT, control.c.CORPUS+'/manifest.json', control.c.METADATA)}
    return control.Spec.from_document(d)


def supervision(cohort):
    return dict(kind='EXP155-masked-OAI-v2', cohort_sha256=life.content_sha256(cohort),
        targets=['Lateral OA', 'PF OA', 'Synovitis'], loss='fixed-12-masked-BCE-study-mean',
        class_weights='CONTROL43-RSNA-only', oai_count=1504, rsna_count=3455,
        sampling=dict(per_epoch=3455, rsna=2407, oai=1048,
                      policy='step7-fixed-step-replacement', selection_seeds=[42,43,44]),
        panel_participants=557, bn_count=4959)


@dataclass(frozen=True)
class Spec(life.ArmSpec):
    def __post_init__(self):
        d = life.parse_json(self.encoded)
        need(d['schema']=='exp190f-spec/v1' and d['family']=='coatnet-oai', 'EXP190F schema differs')
        need(d['run_kind']=='experiment' and d['seed']==43 and d['fold']==0, 'EXP190F requires seed43 fold0 experiment')
        need(d['config']['variant']=='control' and not d['config']['checkpointing'], 'CONTROL43 factory differs')
        need(set(d['inputs'])=={'sources','data','rosters','controls'}, 'input groups differ')
        need(set(d['inputs']['sources'])==set(SOURCES), 'EXP190F sources differ')
        need(set(d['inputs']['data'])=={control.c.INIT, control.c.CORPUS+'/manifest.json', control.c.METADATA,
                                     PRESET, COHORT, PANEL, *CACHE_PINS}, 'EXP190F data differs')
        need(d['config']['bn']==dict(policy='all-mixed-training-once',count=4959,epoch=16), 'mixed BN differs')
        projection(d)
        merged = {}
        for group in d['inputs'].values():
            life._inventory(group)
            need(not set(merged)&set(group), 'duplicate input')
            merged.update(group)
        life._inventory({**merged, **dict.fromkeys(life.STAGE_METADATA,'0'*64)})
        need(not set(merged)&life.STAGE_METADATA, 'reserved stage member')
        life.validate_bindings(self.bindings)
        object.__setattr__(self,'encoded',life.canonical_bytes(d))


def population(spec, root):
    root = Path(root)
    p = control.population(projection(spec.document), root)
    cohort = read(root/COHORT)
    policy = cohort['exclusion']
    # Verify staged bytes directly: the exclusion utility's default root is local.
    need(policy['path']==PANEL and life.file_sha256(root/PANEL)==policy['sha256'], 'panel bytes differ')
    need(len(set(policy['participants']))==557, 'panel participant count differs')
    rows = csv.DictReader(io.StringIO((root/PANEL).read_text(encoding='utf-8-sig')))
    need(rows.fieldnames==['study_key'], 'panel columns differ')
    need(sorted({exclusions.participant(r['study_key']) for r in rows})==policy['participants'], 'panel membership differs')
    ids = cohort['ids']
    need(len(ids)==len(set(ids))==1504 and set(ids)==set(cohort['labels'])==set(cohort['masks']), 'OAI membership differs')
    excluded = set(policy['participants'])
    need(not any(exclusions.participant(u) in excluded for u in ids), 'OAI panel participant leaked')
    manifest = read(root/OAI/'manifest.json')
    need(set(ids)<=set(manifest['ids']), 'OAI cache membership differs')
    need(not set(ids)&set(p['train_ids']+p['val_ids']+p['excluded_gold_ids']), 'cohort overlap')
    for uid in ids:
        ys, ms = cohort['labels'][uid], cohort['masks'][uid]
        need(len(ys)==len(ms)==12 and any(ms), 'empty OAI supervision')
        need(all(m in (0,1) and y in (0,1) and (m or y==0) and (not m or i in (5,6,8))
                 for i,(y,m) in enumerate(zip(ys,ms))), 'OAI mask/label differs')
    need(spec.document['config']['supervision']==supervision(cohort), 'supervision config differs')
    p = copy.deepcopy(p)
    p.update(rsna_train_ids=p['train_ids'], oai_train_ids=ids,
             train_ids=p['train_ids']+ids,
             train_masks={**{u:[1]*12 for u in p['train_ids']}, **cohort['masks']},
             train_labels={**p['train_labels'], **cohort['labels']}, oai_exclusion=policy)
    return p


def validate_inputs(spec, root):
    life._verify_files(root, spec.inventory)
    control.validate_inputs(projection(spec.document), root)
    preset = read(Path(root)/PRESET)
    need(preset['schema']=='exp190f-preset/v1' and preset['seed']==43 and preset['fold']==0
         and preset['supervision']==spec.document['config']['supervision'], 'preset differs')
    need(preset['controls']==spec.document['inputs']['controls'], 'CONTROL43 pins differ')
    need(set(preset['controls'])=={CONTROL+'summary.json',CONTROL+'provenance.json'}, 'wrong paired control')
    need(preset['cohort_sha256']==spec.inventory[COHORT], 'cohort pin differs')
    need(all(spec.inventory[k]==v for k,v in CACHE_PINS.items()), 'corrected cache pins differ')
    return population(spec, root)


def register(root, *, controls, arm, attempt):
    need(controls is not None, 'genuine paired --controls required')
    base = control.register(root,target=None,target_sha256=None,variant='control',population_name='fold0',
        seed=43,arm=arm,attempt=attempt,checkpointing=False,controls=controls)
    d = base.document
    d.update(schema='exp190f-spec/v1',family='coatnet-oai')
    d['inputs']['sources'] = {k:life.file_sha256(Path(root)/k) for k in SOURCES}
    for k in (PRESET,COHORT,PANEL,*CACHE_PINS):
        d['inputs']['data'][k] = life.file_sha256(Path(root)/k)
    d['config']['supervision'] = supervision(read(Path(root)/COHORT))
    d['config']['bn'] = dict(policy='all-mixed-training-once',count=4959,epoch=16)
    spec = Spec.from_document(d)
    validate_inputs(spec,root)
    return spec


def validate_stage(spec, stage):
    need(life._files(stage)==set(spec.inventory)|life.STAGE_METADATA, 'stage inventory differs')
    validate_inputs(spec,stage)
    need((Path(stage)/'arm-spec.json').read_bytes()==spec.encoded, 'staged spec differs')
    expected=dict(schema='exp190f-stage/v1',spec_sha256=spec.sha256,inputs=spec.inventory,status='STAGED_NO_EXECUTION')
    need(read(Path(stage)/'stage-receipt.json')==expected, 'stage receipt differs')
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
    write(work/'stage-receipt.json',dict(schema='exp190f-stage/v1',spec_sha256=spec.sha256,
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
