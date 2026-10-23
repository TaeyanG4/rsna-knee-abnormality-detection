"""CoAtNet lifecycle v1.1.0: immutable members, paired experiments and staging.

Separate from the lifecycle imported by the running MIL queue. Reuses its
strict evidence/receipt primitives without changing globals or old stages.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import shutil

from training.arms import lifecycle as life
from training.arms import coatnet as c
from training.src.raptor_common import LAB

VERSION = '1.1.0'
FULL = 'artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json'
FOLD = 'artifacts/experiments/H_B_109_RPLUS/v1.2.1/parent-population.json'
POPULATION_PINS = {FOLD: '0593263547f540217b08ebe18599f065a47bce4aff335a2a50f499463dc2ad26', FULL: 'b2aaca64289249595e532b74d361175ed78b1170109eee302ec29068e0e7af11'}
INITIAL = {42: '8064b3a56c602e5e52abd3a10b77f235300dd7008f56fc4d7d47fca75e5d3c43',
           43: '1a6c176c585f601375c5dc0006d958b2d27388c915c98eb08049e2a283990eb3'}
SOURCES = tuple(dict.fromkeys((*c.SOURCES, 'training/arms/lifecycle.py',
    'tools/__init__.py', 'tools/exp084_group_loss.py', 'tools/exp099_kfold.py', 'tools/exp101_kfold_arch.py',
    'training/arms/coatnet_fold1.py', 'training/arms/coatnet_fold1_fit.py',
    'training/arms/coatnet_lifecycle.py', 'training/arms/coatnet_fit.py',
    'training/runners/coatnet_fold1.py')))
OUTPUTS = ['model_best.pt', 'predictions-plain.csv', 'predictions-mirror.csv',
           'predictions-mean.csv', 'summary.json', 'provenance.json', 'history.json']
# Explicit reproduction of the legacy CUDA backend policy, not MIL's strict policy.
BACKEND = dict(cudnn_benchmark=True, cudnn_deterministic=False,
               cudnn_allow_tf32=True, matmul_allow_tf32=False,
               deterministic_algorithms=False, float32_matmul_precision='highest')
need = life._need


def read(path):
    return life.parse_json(Path(path).read_bytes())


def write(path, value):
    life.create_new(path, life.canonical_bytes(value))


def uid_hash(ids):
    import hashlib
    return hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest()


def scientific(p, population, variant, seed, checkpointing):
    """Only scientific keys enter equality; no historical budgets/run paths."""
    recipe_keys = ('epochs', 'swa_1based', 'batch', 'effective_batch', 'bn_batch',
                   'bb_lr', 'head_lr', 'wd', 'pct_start', 'clip', 'labels')
    recipe = {k: p['recipe'][k] for k in recipe_keys}
    ids = p['train_ids']
    val = p['val_ids'] if population == 'fold1' else []
    return dict(variant=variant, factory=asdict(c.PRESETS[variant]), population=population,
        recipe=recipe, seed=seed, checkpointing=checkpointing, precision='bf16', eval_precision='fp16',
        train_count=len(ids), train_sha256=uid_hash(ids), val_count=len(val), val_sha256=uid_hash(val),
        pos_weight=p['pos_weight'], epochs=16, updates_per_epoch=math.ceil(len(ids)/16),
        total_updates=16*math.ceil(len(ids)/16), bn={'policy': 'all-training-once', 'count': len(ids), 'epoch': 16},
        readout='held-out-fold1' if val else 'first-training-study-diagnostic',
        backend_policy=BACKEND, targets=list(LAB), initial_state_sha256=INITIAL[seed])


@dataclass(frozen=True)
class Spec(life.ArmSpec):
    def __post_init__(self):
        d = life.parse_json(self.encoded)
        life._keys(d, {'schema','family','run_kind','arm','attempt','seed','fold','config','inputs',
                      'target_environment','outputs','population_source','target_source'}, 'CoAtNet spec')
        need(d['schema'] == 'coatnet-fold1-spec/v1' and d['family'] == 'coatnet', 'CoAtNet schema differs')
        need(d['run_kind'] in ('member','experiment'), 'unknown run kind')
        for k in ('arm','attempt'):
            need('/' not in life.safe_member(d[k]), 'identity must be one component')
        need(type(d['seed']) is int and d['seed'] in (42,43), 'registered seed required')
        need(type(d['fold']) is int and d['fold'] in (1,-1), 'registered fold required')
        groups = {'sources','data','rosters','controls'} | ({'evidence'} if d['run_kind']=='member' else set())
        life._keys(d['inputs'], groups, 'CoAtNet input groups')
        combined = {}
        for k,v in d['inputs'].items():
            if k == 'controls' and d['run_kind'] == 'member':
                need(v == {}, 'member cannot carry paired controls')
                continue
            life._inventory(v)
            need(not set(combined)&set(v), 'duplicate input')
            combined.update(v)
        life._inventory({**combined, **dict.fromkeys(life.STAGE_METADATA, '0'*64)})
        need(not set(combined)&life.STAGE_METADATA, 'reserved stage member')
        need(set(d['inputs']['sources']) == set(SOURCES), 'source inventory differs')
        need(set(d['inputs']['data']) == {c.INIT,c.CORPUS+'/manifest.json',c.METADATA}, 'data inventory differs')
        need(d['inputs']['data'][c.METADATA] == c.PINS[c.METADATA], 'coverage metadata differs')
        need(d['outputs'] == OUTPUTS, 'required outputs differ')
        need(d['config']['variant'] in c.PRESETS, 'variant differs')
        need(d['config']['population'] in ('fold1','full'), 'population differs')
        need(d['fold']==(1 if d['config']['population']=='fold1' else -1), 'fold/population differs')
        need(type(d['config']['checkpointing']) is bool, 'checkpointing must be boolean')
        need(d['population_source'] == (FOLD if d['fold']==1 else FULL), 'population source differs')
        need(d['population_source'] in d['inputs']['rosters'], 'population must be bound')
        need(d['inputs']['rosters']=={d['population_source']:POPULATION_PINS[d['population_source']]}, 'population identity differs')
        if d['run_kind']=='member':
            need(set(d['inputs']['evidence'])=={d['target_source']}, 'target source differs')
        else:
            need(d['target_source'] is None, 'experiment target comes from paired control')
        life.validate_environment(d['target_environment'])
        need(d['target_environment']['schema'] == (life.MEMBER_ENVIRONMENT_SCHEMA if d['run_kind']=='member'
             else life.ENVIRONMENT_SCHEMA), 'run kind target schema differs')
        life.validate_bindings(self.bindings)
        object.__setattr__(self, 'encoded', life.canonical_bytes(d))


def population(spec, root):
    d = spec.document
    p = read(life._path(root, d['population_source']))
    ids, labels = p['train_ids'], p['train_labels']
    excluded = p['excluded_gold_ids']
    need(len(ids)==(3479 if d['fold']==1 else 4347) and len(set(ids))==len(ids), 'population count differs')
    need(len(set(excluded))==58 and not set(ids)&set(excluded), 'Gold exclusion differs')
    need(set(labels)==set(ids), 'label membership differs')
    import numpy as np
    values = np.array([labels[u] for u in ids], dtype=np.float32)
    need(values.shape==(len(ids),12) and np.isfinite(values).all() and
         ((values>=0)&(values<=1)).all(), 'label values invalid')
    val = p['val_ids'] if d['fold']==1 else []
    need(not set(val)&(set(ids)|set(excluded)) and len(val)==(864 if d['fold']==1 else 0), 'validation membership differs')
    expected = scientific(p, d['config']['population'], d['config']['variant'], d['seed'], d['config']['checkpointing'])
    need(d['config']==expected, 'scientific config differs from population recipe')
    return p


def validate_inputs(spec, root):
    life._verify_files(root, spec.inventory)
    d = spec.document
    p = population(spec, root)
    need(d['inputs']['data'].get(c.INIT)==p['init_sha256'] and
         d['inputs']['data'].get(c.CORPUS+'/manifest.json')==p['corpus_manifest_sha256'], 'corpus/init binding differs')
    if spec.run_kind=='member':
        digest = d['inputs']['evidence'][d['target_source']]
        profile, declaration = life.environment_from_declaration(life._path(root,d['target_source']), expected_sha256=digest)
        need(profile==d['target_environment'], 'target declaration differs')
        life.check_target_observation(spec, declaration)
    else:
        controls = d['inputs']['controls']
        paths = {Path(k).name:k for k in controls}
        need(len(controls)==2 and set(paths)=={'summary.json','provenance.json'}, 'paired control records missing')
        profile = life.environment_from_control(*[life._path(root,paths[k]) for k in ('summary.json','provenance.json')],
            expected_hashes={k:controls[v] for k,v in paths.items()})
        need(profile==d['target_environment'], 'paired target differs')
        summary, provenance = [read(life._path(root,paths[k])) for k in ('summary.json','provenance.json')]
        reference = dict(d['config'], variant='control', factory=asdict(c.PRESETS['control']))
        need(summary.get('status')=='COMPLETE_TRAINING_AND_READOUT' and summary.get('mode')=='full' and
             summary.get('updates')==reference['total_updates'] and summary.get('skips')==0,
             'paired control must be a completed full fit')
        need(provenance.get('scientific')==reference, 'paired control scientific identity differs')
        need(provenance.get('population_sha256')==d['inputs']['rosters'][d['population_source']], 'paired population differs')
        life.check_target_observation(spec, summary['observation'])
    return p


def register(root, *, target, target_sha256, variant, population_name, seed, arm, attempt,
             checkpointing=False, controls=None):
    root=Path(root)
    need(population_name in ('fold1','full') and variant in c.PRESETS and seed in INITIAL, 'unsupported registration')
    need(type(checkpointing) is bool, 'checkpointing must be boolean')
    if controls is not None:
        need(type(controls) is dict and len(controls)==2, 'paired control records missing')
        controls={life.safe_member(k.replace('\\','/')):v for k,v in controls.items()}
        need(len(controls)==2, 'duplicate paired control path')
    source=FOLD if population_name=='fold1' else FULL
    p=read(root/source)
    config=scientific(p,population_name,variant,seed,checkpointing)
    sources={f:c.sha256(root/f) for f in SOURCES}
    data={f:c.sha256(root/f) for f in (c.INIT,c.CORPUS+'/manifest.json',c.METADATA)}
    groups=dict(sources=sources,data=data,rosters={source:c.sha256(root/source)},controls=controls or {})
    if controls:
        life._verify_files(root,controls)
        paths={Path(f).name:f for f in controls}
        need(len(controls)==2 and set(paths)=={'summary.json','provenance.json'}, 'paired control records missing')
        profile=life.environment_from_control(*[root/paths[k] for k in ('summary.json','provenance.json')],
            expected_hashes={k:controls[v] for k,v in paths.items()})
        target=None
    else:
        need(type(target) is str and bool(target), 'member target declaration required')
        target=life.safe_member(target.replace('\\','/'))
        profile,_=life.environment_from_declaration(root/target,expected_sha256=target_sha256)
        groups['evidence']={target:target_sha256}
    spec=Spec.from_document(dict(schema='coatnet-fold1-spec/v1',family='coatnet',run_kind='experiment' if controls else 'member',
        arm=arm,attempt=attempt,seed=seed,fold=1 if population_name=='fold1' else -1,
        population_source=source,target_source=target,config=config,inputs=groups,
        target_environment=profile,outputs=OUTPUTS))
    validate_inputs(spec,root)
    return spec


def validate_stage(spec, stage):
    need(life._files(stage)==set(spec.inventory)|life.STAGE_METADATA, 'stage inventory differs')
    validate_inputs(spec,stage)
    need((Path(stage)/'arm-spec.json').read_bytes()==spec.encoded, 'staged spec differs')
    expected=dict(schema='coatnet-stage/v1',spec_sha256=spec.sha256,inputs=spec.inventory,status='STAGED_NO_EXECUTION')
    need(read(Path(stage)/'stage-receipt.json')==expected, 'stage receipt differs')
    return expected


def prepare(spec, root, destination):
    if Path(destination).exists():
        return validate_stage(spec,destination)
    validate_inputs(spec,root)
    work=life._working_directory(destination)
    for name in spec.inventory:
        target=life._path(work,name)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(life._path(root,name),target)
    life.create_new(work/'arm-spec.json',spec.encoded)
    write(work/'stage-receipt.json',dict(schema='coatnet-stage/v1',spec_sha256=spec.sha256,
        inputs=spec.inventory,status='STAGED_NO_EXECUTION'))
    result=validate_stage(spec,work)
    life._publish(work,destination)
    return result


def admit(spec, stage, note, evidence_root, observation=None):
    validate_stage(spec,stage)
    digest=life.verify_evidence(spec,note,evidence_root)
    if observation is not None:
        life.require_verification_pass(note,**spec.verification_args)
        life.check_target_observation(spec,observation)
    return dict(status='EVIDENCE_READY_TARGET_CHECK_REQUIRED' if note['verdict']=='PASS' else 'HOLD',
        launch_authorized=False,spec_sha256=spec.sha256,verification_sha256=digest,blockers=note['blockers'])
