"""vNext-R7-PL 1.1.1: production adapters and fail-closed publication.

Changelog 1.1.1: shared H-B-146 readout adapter and receipt decision binding.

Changelog 1.1.0: diagnostic-only sensitivity, authenticated heldout targets,
production cohort/runner/export adapters. Frozen v1.0.0 stages are retained.

No training, download or submission commands. Production requires separately
sealed GO, checkpoint/training bindings and two-T4 qualification. Frozen native
workers are adapted only inside dedicated children; original stages stay intact.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import release_metadata_v122 as metadata
from tools.release_deadline_v1 import Deadline, Supervisor

VERSION = '1.1.1'
STAGE = ROOT / 'artifacts/experiments/VNEXT_R7_PL/v1.1.1'
SCRATCH = ROOT / '.local/scratch/astra-b1/pl-prod-prep'
BASE = ROOT / 'artifacts/experiments/HB116_VNEXT/v1.2.0/release'
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
MIL_ARCH = 'convnext_small.fb_in22k_ft_in1k'
COAT_ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
OWN2_SHA = '1b925ea8d1469e55e95796159241cd5c34f2cff3e1e6dc71e1dadc61a2355937'
R7_SHA = 'c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6'
SOURCES = ['tools/vnext_r7_pl_v1.py', 'tools/vnext_r7_pl_worker_v1.py',
           'tools/hb116_vnext_release.py', 'tools/hb116_native_runtime.py',
           'tools/hb127_vnextr7_release.py', 'tools/release_deadline_v1.py',
           'tools/release_metadata_v122.py', 'tools/hb126_codecs.py',
           'tools/exp169_release_c85r7codec_deadline_v122.py',
           'tools/exp169_release_c85r7codec.py', 'tools/exp169_r7_port.py',
           'tools/hb105_serving_v101.py', 'tools/exp152_own0949_release.py',
           'tools/vnext_r7_pl_targets_v110.py', 'tools/vnext_r7_pl_train_v110.py',
           'tools/vnext_r7_pl_go_v111.py', 'tools/hb146_readout_v111.py',
           'tools/multi_lens_readout_l4_v2.py', 'tools/multi_lens_readout.py',
           'tools/exp156_k_sweep.py', 'tools/exp125_radr50_k16_local.py',
           'tools/exp118_radr50_local_s43.py',
           'training/arms/vnext_r7_pl_coat_v110.py',
           'tools/hb146_pl_labels_v111.py', 'tools/hb146_teacher_v111.py',
           'tools/hb146_screen_v111.py', 'tools/hb145_r7_heldout_v1.py',
           'tools/hb145_r7_pipeline_v111.py', 'tools/hb145_r7_monitor_v111.py',
           'training/arms/hb146_pl_v1.py', 'training/arms/mil_screen_hooks_v1.py',
           'training/arms/mil_screen_v1.py', 'training/arms/mil.py',
           'training/arms/mil_small.py', 'training/arms/lifecycle.py',
           'training/src/raptor_common.py', 'training/src/raptor_runtime.py',
           'training/src/raptor_exp185.py', 'training/src/exp143_v2.py',
           'training/src/exp143_own0949.py', 'tools/exp143_v2.py',
           'tools/exp112_full_swa.py', 'tools/exp113_radr50_local.py',
           'tools/exp143_own0949.py', 'tools/exp101_kfold_arch.py',
           'tools/exp099_kfold.py', 'tools/exp084_group_loss.py', 'tools/exp107_kfold_radr50.py',
           'training/src/raptor_dicom.py', 'training/src/raptor_corpus.py',
           'training/configs/hb122-control-s42-f0-v1.json']


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def dump(path, value, *, immutable=False):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    return metadata.atomic_write(path, data, immutable=immutable)


def pin(entry):
    path = Path(entry['path'])
    need(path.is_file() and sha(path) == entry['sha256'], 'asset pin mismatch: ' + str(path))
    if 'bytes' in entry:
        need(path.stat().st_size == entry['bytes'], 'asset size mismatch')
    return path


def member_roster(mode='fivefold', seeds=(42, 43)):
    need(mode in ('fivefold', 'all-data') and list(seeds) in ([42], [42, 43]), 'MIL mode/seeds')
    folds = range(5) if mode == 'fivefold' else ['all']
    entries = [dict(id=f'PL-S-s{s}-f{f}', branch='own', kind='MIL', family='S',
                    arch=MIL_ARCH, seed=s, fold=f) for s in seeds for f in folds]
    return entries + [dict(id='PL-CoAt', branch='own', kind='CoAt', arch=COAT_ARCH, seed=42, fold='all'),
        dict(id='EXP155', branch='own', kind='OWN2', sha256=OWN2_SHA),
        dict(id='R2-s43', branch='R', kind='R2', seed=43, sha256='1ddbc89c0f233d7f37c4811250d05f7446e2aeb4af8b522f45d9d1106b07b359'),
        dict(id='R2-s44', branch='R', kind='R2', seed=44, sha256='2b69a0552666d74b0b3871f3cb199303777621a3c090e322372f8fed0617a4a4'),
        dict(id='S_R7', branch='R7', kind='R7', sha256=R7_SHA)]


def source_payloads():
    from tools import hb127_vnextr7_release as base
    from tools import exp169_release_c85r7codec_deadline_v122 as insurance
    files = base.source_payloads()
    for name, data in insurance.source_payloads().items():
        key = 'tools/' + name
        need(key not in files or files[key] == data, 'source conflict: '+key)
        files[key] = data
    for name in SOURCES:
        files[name] = (ROOT/name).read_bytes()
    return files


def training_configs(arm):
    """Recipe index; executable adapters materialize member-specific pinned specs."""
    need(arm in ('global', 'fill'), 'winning arm')
    target = dict(producer='H-B-145', producer_version='1.1.2', producer_exposure='EXPOSED',
        adapter='tools/vnext_r7_pl_targets_v110.py', role='heldout', output='heldout_raw.csv',
        output_kind='pre_stacker', model_fold='UID hash fold', teacher_seal=None,
        targets_sha256=None, ordered_train_ids_sha256=None, frozen_l2_pos_weight_sha256=None,
        calibration='slope=1; training-only logit offset matches L2 mean; clip=1e-6',
        formula='.5 L2 + .5 scaled teacher' if arm == 'global' else 'L2; frozen silent cells := scaled teacher',
        silence_mask_sha256=None if arm == 'fill' else 'not_used',
        clean_oof=False, exposure='EXPOSED_EXPLORATORY',
        limitations=['pooled report/state priors', 'fastMRI+ shared auxiliary',
                     'SKM-TEA crop anchor', '4344/4343 membership unresolved'],
        forbidden_recompute=['silence mask', 'pos_weight from PL', 'validation calibration'])
    common = dict(schema='pl-production-plan-v1', version=VERSION, arm=arm, status='CONDITIONAL_HOLD',
        go_receipt=None, target_contract=target, workers=0, cpu_threads=4,
        pair_policy='control/PL comparison entirely on one GPU; no cross-GPU arms',
        validation='training Gold58 excluded; no test or validation labels for selection')
    plans = []
    for mode in ('fivefold', 'all-data'):
        for seed in (42, 43):
            for fold in (range(5) if mode == 'fivefold' else ['all']):
                plans.append(dict(common, id=f'{arm}-mil-s{seed}-f{fold}', family='MIL',
                    seed=seed, fold=fold, mode=mode, arch=MIL_ARCH,
                    trainer_components=['training/arms/hb146_pl_v1.py', 'training/arms/mil_screen_hooks_v1.py'],
                    runner_contract='tools/vnext_r7_pl_train_v110.py; shared H-B-146 hooks; pinned member spec',
                    corpus='storage/cache/derived/RAPTOR64_336_C140_S02',
                    init_path='artifacts/experiments/EXP171_CONVNEXT_S_L2/initial-state.pt',
                    init_sha256='bf08e71a6c568009ff1e78e0ee1fb4f7e501883ce84000996480aa2c9e53a348',
                    recipe=dict(epochs=20, train_k=12, batch=8, bb_lr=.0005, head_lr=.001,
                        wd=.02, pct_start=.15, clip=3., amp='fp16', checkpoint='fixed-final epoch19',
                        loss='BCE frozen training-L2 pos_weight', drop_last=True),
                    serving=dict(k=36, mirror=True, checkpoint_k_eval=24, res=224,
                                 pool='finding_spatial_tokens_v1', policy='native mixed center3'),
                    estimate_4070_hours=2/3 if mode == 'fivefold' else 5/6,
                    vast_3090='eligible; optional seed43, rerun paired control there if comparing'))
    plans.append(dict(common, id=arm+'-coat-s42-all', family='CoAt', seed=42, fold='all',
        mode='all-data', arch=COAT_ARCH, trainer_components=['tools/exp143_v2.py', 'training/src/exp143_v2.py'],
        runner_contract='tools/vnext_r7_pl_train_v110.py + training/arms/vnext_r7_pl_coat_v110.py',
        efficacy_transfer='MIL-to-CoAt transfer unvalidated by H-B-146 screen',
        corpus='storage/cache/derived/EXP143_RAPTOR96_384_C140_S02',
        recipe=dict(epochs=16, train_k=12, batch=1, effective_batch=16, bn_batch=2,
            bb_lr=4.01e-5, head_lr=1.95e-3, wd=.0588, pct_start=.15, clip=3.,
            swa_1based=[14, 11, 12], bn='training-only fixed-SWA recalibration',
            res=384, norm='imagenet', window_policy='legacy', oai=False,
            loss='BCE frozen training-L2 pos_weight', checkpoint='fixed-SWA-BN', grad_checkpointing=True),
        serving=dict(res=320, k=94, mirror=True), estimate_4070_hours=4.,
        vast_3090='eligible; full fit + SWA + BN on one GPU; planning 4-6h, unmeasured'))
    return plans


def register(out=STAGE, arm='global', mode='fivefold', seeds=(42, 43)):
    out = Path(out)
    need(out.resolve() != BASE.resolve(), 'frozen stage')
    reg = dict(name='vNext-R7-PL', version=VERSION, status='PREPARED_NOT_QUALIFIED',
        arm=arm, mode=mode, seeds=list(seeds), clean_oof=False, exposure='EXPOSED_EXPLORATORY',
        members=member_roster(mode, seeds), weights=dict(primary=.30, sensitivity=.40,
            own=[17, 17, 12], own_order=['PL-CoAt', 'EXP155', 'PL-MIL'], public=[1, 1]),
        formula='P=mean(rank(R2),rank(R7)); O=weighted(rank(PL-CoAt),rank(OWN2),rank(mean(PL-MIL))); final=(1-w)*rank(P)+w*rank(O)',
        baseline_registration_sha256=sha(BASE/'registration.json'),
        source_pins={name: sha(ROOT/name) for name in SOURCES},
        payload_pins={name: hashlib.sha256(data).hexdigest() for name, data in source_payloads().items()},
        upstream_source_pins=read(BASE/'registration.json')['source_pins'],
        deadline_seconds=31200, limit_seconds=32400, reserve_seconds=1200,
        runtime=dict(N=1300, native_seconds_per_study=8.3, r7_seconds_per_study=11.6,
            separate_r2_upper_seconds_per_study=2.8, conservative_seconds=29510,
            native_includes_r2_seconds=25870, setup_and_margin_budget_seconds=1690,
            qualification='planning only; two-T4 visible/full and fault-N32 required'))
    need(arm in ('global', 'fill'), 'winning arm')
    dump(out/'registration.json', reg, immutable=True)
    metadata.write_once(out/'registration.sha256', (digest(reg)+'\n').encode())
    return {'status': 'REGISTERED_CONDITIONAL', 'members': len(reg['members']), 'seal': digest(reg)}


def registration(stage=STAGE):
    stage = Path(stage)
    reg = read(stage/'registration.json')
    need(digest(reg) == (stage/'registration.sha256').read_text().strip(), 'registration seal')
    need(reg['weights'] == dict(primary=.30, sensitivity=.40, own=[17, 17, 12],
        own_order=['PL-CoAt', 'EXP155', 'PL-MIL'], public=[1, 1]), 'sealed weights')
    for name, expected in reg['source_pins'].items():
        need(sha(ROOT/name) == expected, 'source drift: '+name)
    payload = source_payloads() if BASE.exists() else {name: (ROOT/name).read_bytes() for name in reg['payload_pins']}
    need({name: hashlib.sha256(data).hexdigest() for name, data in payload.items()} == reg['payload_pins'], 'source closure drift')
    return reg


def prepare(stage=STAGE):
    stage = Path(stage)
    reg = registration(stage)
    from tools.hb116_vnext_release import package
    archive, pins = package(source_payloads())
    need(pins == reg['payload_pins'], 'package pins')
    metadata.write_once(stage/'runtime.zip', archive)
    for arm in ('global', 'fill'):
        dump(stage/f'training-{arm}.json', training_configs(arm), immutable=True)
    template = dict(registration_sha256=digest(reg), arm=reg['arm'], go=None, teacher=None,
        qualification=None, members=[dict(e, path=None, bytes=None, sha256=e.get('sha256'),
            training_receipt=None) for e in reg['members']],
        runtime=dict(legacy=None, public_source=None, own_manifest=None,
            competition=None, bundle=None, code=None, anatomy=None, source=None, wheels=None))
    dump(stage/'bindings-template.json', template, immutable=True)
    # First cell starts the one absolute budget before all setup/codec/import work.
    cells = [dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=[
        'import os, time\n',
        "os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')\n",
        "os.environ.setdefault('KNEE_RELEASE_START_MONOTONIC_V1', repr(time.monotonic()))\n",
        "os.environ.setdefault('KNEE_RELEASE_DEADLINE_SECONDS_V1', '31200')\n"]),
        dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=[
        'import hashlib, json, pathlib, sys, zipfile\n',
        "stage = pathlib.Path(os.environ['PL_STAGE']).resolve()\n",
        "runtime = pathlib.Path(os.environ['PL_RUNTIME']).resolve()\n",
        "assert not runtime.exists(), 'fresh runtime directory required'\n",
        "reg = json.loads((stage / 'registration.json').read_text())\n",
        "seal = hashlib.sha256(json.dumps(reg, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()\n",
        "assert seal == (stage / 'registration.sha256').read_text().strip()\n",
        "assert seal == os.environ['PL_REGISTRATION_SHA256']\n",
        "with zipfile.ZipFile(stage / 'runtime.zip') as archive:\n",
        "    assert len(archive.namelist()) == len(reg['payload_pins'])\n",
        "    assert {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()} == reg['payload_pins']\n",
        "    assert all((runtime / n).resolve().is_relative_to(runtime) for n in archive.namelist())\n",
        '    archive.extractall(runtime)\n',
        "sys.path[:0] = [str(runtime), str(runtime / 'tools')]\n",
        'from tools.release_deadline_v1 import Deadline\n', 'deadline = Deadline.from_env()\n',
        'from tools.vnext_r7_pl_v1 import serve\n',
        "serve(stage, os.environ['PL_BINDINGS'], os.environ['PL_OUTPUT'], deadline=deadline)\n"])]
    dump(stage/'release-template.ipynb', dict(nbformat=4, nbformat_minor=5, metadata={}, cells=cells), immutable=True)
    return dict(status='PREPARED', configs_per_arm=len(training_configs('global')), production_run=False)


def validate(stage=STAGE):
    reg = registration(stage)
    for arm in ('global', 'fill'):
        need(read(Path(stage)/f'training-{arm}.json') == training_configs(arm), 'training plan drift')
    need(reg['members'] == member_roster(reg['mode'], reg['seeds']), 'roster drift')
    import zipfile
    with zipfile.ZipFile(Path(stage)/'runtime.zip') as archive:
        need({name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()} == reg['payload_pins'], 'runtime archive drift')
    return dict(status='PASS_PREPARED', training_runner='v1.1.1 pinned production adapters available',
                qualified_for_submission=False)


def validate_bindings(reg, bindings):
    """Only explicit, pinned metadata inputs; never reads target/label tables."""
    need(bindings['registration_sha256'] == digest(reg) and bindings['arm'] == reg['arm'], 'binding registration/arm')
    from tools import vnext_r7_pl_go_v111 as scientific_go
    go = scientific_go.validate(bindings['go'], reg['arm'])
    teacher = read(pin(bindings['teacher']))
    need(teacher.get('producer') == 'H-B-145' and teacher.get('producer_version') == '1.1.2'
         and teacher.get('producer_exposure') == 'EXPOSED' and teacher.get('status') == 'PASS'
         and teacher.get('role') == 'heldout' and teacher.get('output_kind') == 'pre_stacker'
         and teacher.get('model_fold') == 'int(sha1(uid)[:8],16)%4'
         and teacher.get('complete') is True and teacher.get('failures') == 0
         and teacher.get('clean_oof') is False and teacher.get('exposure') == 'EXPOSED_EXPLORATORY'
         and teacher.get('producer_identity') and teacher.get('csv_sha256'), 'heldout teacher seal required')
    entries = bindings['members']
    need([e['id'] for e in entries] == [e['id'] for e in reg['members']], 'checkpoint roster')
    for template, entry in zip(reg['members'], entries):
        need(all(entry.get(k) == v for k, v in template.items()), 'member recipe changed')
        if entry['kind'] != 'R7':
            pin(entry)
        if entry['kind'] in ('MIL', 'CoAt'):
            receipt = read(pin(entry['training_receipt']))
            need(receipt.get('status') == 'COMPLETE' and receipt.get('checkpoint_sha256') == entry['sha256']
                 and receipt.get('arm') == reg['arm'] and receipt.get('seed') == entry['seed']
                 and receipt.get('fold') == entry['fold'] and receipt.get('teacher_sha256') == bindings['teacher']['sha256']
                 and receipt.get('registration_sha256') == digest(reg) and receipt.get('clean_oof') is False
                 and receipt.get('exposure') == 'EXPOSED_EXPLORATORY'
                 and receipt.get('target_sha256') and receipt.get('train_ids_sha256')
                 and receipt.get('l2_pos_weight_sha256') and receipt.get('spec_sha256'), 'PL training binding')
            receipt_go = scientific_go.admission(receipt.get('go'), reg['arm'], receipt['spec_sha256'])
            need(receipt_go == go, 'training/serving GO source differs')
            if reg['arm'] == 'fill':
                need(receipt.get('silence_mask_sha256'), 'fill mask binding')
    return entries


def bind(stage, inputs, output):
    """Hash explicit paths in the filled template; never load checkpoint tensors."""
    reg = registration(stage)
    bound = read(inputs)
    def seal(entry):
        path = Path(entry['path']).resolve()
        result = dict(entry, path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        need(not entry.get('sha256') or entry['sha256'] == result['sha256'], 'binding pin changed')
        return result
    for name in ('go', 'teacher', 'qualification'):
        if bound.get(name):
            bound[name] = seal(bound[name])
    for index, entry in enumerate(bound['members']):
        if entry['kind'] != 'R7':
            bound['members'][index] = entry = seal(entry)
        if entry['kind'] in ('MIL', 'CoAt'):
            entry['training_receipt'] = seal(entry['training_receipt'])
    validate_bindings(reg, bound)
    dump(output, bound, immutable=True)
    return dict(status='BOUND', registration_sha256=digest(reg), production_run=False)


def admit(stage=STAGE, bindings=None):
    reg = registration(stage)
    if bindings is None:
        return dict(status='HOLD', reasons=['H-B-146 GO absent', 'trained checkpoint bindings absent',
            'heldout target seal absent', 'two-T4 runtime/fault-N32 qualification absent'])
    bound = read(bindings)
    validate_bindings(reg, bound)
    q = read(pin(bound['qualification']))
    need(q.get('registration_sha256') == digest(reg) and q.get('bindings_members_sha256') == digest(bound['members']), 'qualification identity')
    required = ['complete_ordered_csv', 'finite', 'zero_healthy_fallbacks', 'descendants_exited',
                'gpu_memory_released', 'fault_n32_pass', 'native_parity', 'codecs_best_effort']
    need(q.get('device') == '2xT4' and all(q.get(k) is True for k in required)
         and 0 < q.get('upper_seconds', 1e9) <= 31200
         and 0 <= q.get('publication_reserve_seconds', 1e9) <= 1200
         and q.get('total_seconds', 1e9) <= 32400, 'target qualification required')
    return dict(status='ADMITTED', qualified_for_submission=False)


def blend(reg, raw, weight='primary'):
    import numpy as np
    from tools.hb116_vnext_release import mean_survivors as mean, ranks
    need(weight in ('primary', 'sensitivity'), 'unregistered weight')
    n = len(next(iter(raw.values())))
    get = lambda name: np.asarray(raw.get(name, np.full((n, 12), np.nan)))
    mil = mean([get(e['id']) for e in reg['members'] if e['kind'] == 'MIL'], dtype='float32')
    own = mean([ranks(get('PL-CoAt')), ranks(get('EXP155')), ranks(mil)], reg['weights']['own'])
    r2 = mean([get('R2-s43'), get('R2-s44')], dtype='float32')
    public = mean([ranks(r2), ranks(get('S_R7'))])
    w = reg['weights'][weight]
    final = mean([ranks(public), ranks(own)], [1-w, w])
    final[~np.isfinite(final).all(1)] = .5
    return final.astype(np.float32)


class Journal:
    """A complete row is visible only after an atomic replace; failures are sticky."""
    def __init__(self, directory, ids, identity):
        self.directory, self.ids, self.identity = Path(directory), list(ids), identity
        self.members = list(identity['members'])
        for name, value in [('identity', identity), ('assignment', dict(ids=self.ids, members=self.members))]:
            dump(self.directory/(name+'.json'), value, immutable=True)
        self.failures = set()

    def record(self, uid, member, values=None, error=None, seconds=None):
        import numpy as np
        need(uid in self.ids and member in self.members, 'journal assignment')
        key = digest([uid, member])
        path = self.directory/'rows'/(key+'.json')
        if path.exists():
            try:
                previous = metadata.metadata(path)
            except metadata.MetadataUnavailable:
                previous = {}
            if previous.get('status') != 'OK':
                self.failures.add(key)
        good = error is None and key not in self.failures
        try:
            a = np.asarray(values, dtype=np.float32) if good else None
        except (TypeError, ValueError, OverflowError):
            good, a = False, None
        good = good and a.shape == (12,) and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
        if not good:
            self.failures.add(key)
        dump(path, dict(uid=uid, member=member, identity=self.identity,
            status='OK' if good else 'FAILED', values=a.tolist() if good else None))

    def finish(self):
        dump(self.directory/'receipt.json', dict(identity=self.identity, ids=self.ids,
            members=self.members, status='COMPLETE', failures=sorted(self.failures)))


def collect(directory, ids, identity):
    import numpy as np
    directory, members = Path(directory), list(identity['members'])
    raw = {k: np.full((len(ids), 12), np.nan, np.float32) for k in members}
    reason = None
    try:
        actual = metadata.shaped(metadata.metadata(directory/'identity.json'), identity)
        need(actual == identity, 'journal identity authentication')
        assignment = metadata.shaped(metadata.metadata(directory/'assignment.json'), dict(ids=ids, members=members))
        need(assignment == dict(ids=ids, members=members), 'assignment authentication')
        receipt = metadata.shaped(metadata.metadata(directory/'receipt.json'),
            dict(identity=identity, ids=ids, members=members, status='', failures=[]))
        need(receipt['identity'] == identity and receipt['ids'] == ids and receipt['members'] == members,
             'receipt authentication')
        if receipt['status'] != 'COMPLETE':
            raise metadata.MetadataUnavailable('receipt incomplete')
        if not all(isinstance(key, str) for key in receipt['failures']):
            raise metadata.MetadataUnavailable('receipt failure shape')
        failed = set(receipt['failures'])
        need(failed <= {digest([u, m]) for u in ids for m in members}, 'failure assignment authentication')
    except metadata.MetadataUnavailable as exc:
        return raw, len(ids)*len(members), str(exc)
    for i, uid in enumerate(ids):
        for member in members:
            key = digest([uid, member])
            try:
                row = metadata.shaped(metadata.metadata(directory/'rows'/(key+'.json')),
                                      dict(uid='', member='', identity=identity, status=''))
                need(row['uid'] == uid and row['member'] == member and row['identity'] == identity,
                     'row authentication')
                if key in failed or row['status'] != 'OK':
                    continue
                try:
                    a = np.asarray(row.get('values'), dtype=np.float32)
                except (ValueError, TypeError, OverflowError):
                    continue
                if a.shape == (12,) and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all():
                    raw[member][i] = a
            except (metadata.MetadataUnavailable, TypeError):
                continue
    return raw, sum(int((~np.isfinite(v).all(1)).sum()) for v in raw.values()), reason


def publish(reg, ids, shards, output, weight='primary', *, filename='submission.csv', diagnostic=False):
    # Enforce the routing contract at the writer as well as the GPU entry point.
    # No alternate filename or direct caller may turn .40 into a production run.
    need(weight in ('primary', 'sensitivity'), 'unregistered weight')
    need(diagnostic or weight == 'primary', 'sensitivity is diagnostic only')
    need(filename == ('diagnostic.csv' if diagnostic else 'submission.csv'),
         'diagnostic/publication filename contract')
    import numpy as np
    need(ids and len(ids) == len(set(ids)), 'publication cohort')
    raw = {e['id']: np.full((len(ids), 12), np.nan, np.float32) for e in reg['members']}
    positions = {uid: i for i, uid in enumerate(ids)}
    assigned = set()
    for shard in shards:
        values, _, _ = collect(shard['path'], shard['ids'], shard['identity'])
        for member, array in values.items():
            need(member in raw, 'unknown member')
            for i, uid in enumerate(shard['ids']):
                need(uid in positions and (uid, member) not in assigned, 'overlapping/foreign assignment')
                assigned.add((uid, member))
                raw[member][positions[uid]] = array[i]
    final = blend(reg, raw, weight)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow([UID, *TARGETS])
    writer.writerows([[uid, *[format(float(x), '.9g') for x in row]] for uid, row in zip(ids, final)])
    output = Path(output)
    metadata.atomic_write(output/filename, stream.getvalue().encode())
    result = dict(status='DIAGNOSTIC_ONLY' if diagnostic else 'PUBLISHED',
        diagnostic=diagnostic, qualified_for_submission=False,
        rows=len(ids), registration_sha256=digest(reg), weight=weight,
        fallbacks=sum(int((~np.isfinite(v).all(1)).sum()) for v in raw.values()),
        submission_sha256=sha(output/filename))
    try:
        dump(output/'publication.json', result)
    except OSError:
        pass  # telemetry may fail after the CSV has committed
    return result


def identity(reg, branch, entries, run_id):
    return dict(version=VERSION, registration=digest(reg), run_id=run_id, branch=branch,
                members={e['id']: e.get('sha256') or 'PLACEHOLDER' for e in entries})


def runtime_preflight(reg, runtime):
    """Authenticate external executable assets before child failures can fall back."""
    from tools import hb116_vnext_release as h, hb127_vnextr7_release as base
    h.pin(runtime['public_source'], {'sha256': h.c.p.PINS['public-serving.py']})
    h.pin(runtime['own_manifest'], {'sha256': h.OWN_MANIFEST_SHA})
    for name, expected in reg['payload_pins'].items():
        if name.startswith('legacy/'):
            pin(dict(path=str(Path(runtime['legacy'])/name.removeprefix('legacy/')), sha256=expected))
    base.r7.verify_bundle(Path(runtime['bundle']), base.MANIFEST_SHA, Path(runtime['source']))
    base.r7.code_record(Path(runtime['code']), Path(runtime['anatomy']), Path(runtime['source']))


def serve(stage, binding_path, output, *, deadline=None, weight='primary', diagnostic=False):
    """GPU entry point for B only. Always publish after supervised cleanup."""
    need(weight in ('primary', 'sensitivity'), 'unregistered weight')
    need(diagnostic or weight == 'primary', 'sensitivity is diagnostic only')
    deadline = deadline or Deadline.from_env()
    reg = registration(stage)
    need(deadline.seconds <= reg['deadline_seconds'], 'deadline exceeds sealed compute budget')
    if not diagnostic:
        admit(stage, binding_path)
    bindings = read(binding_path)
    entries = validate_bindings(reg, bindings)
    from tools import hb127_vnextr7_release as base
    runtime = bindings['runtime']
    runtime_preflight(reg, runtime)
    output = Path(output).resolve()
    need(not output.exists(), 'fresh output required')
    output.mkdir(parents=True)
    with (Path(runtime['competition'])/'sample_submission.csv').open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames == [UID, *TARGETS], 'sample schema')
        ids = [row[UID] for row in reader]
    with (Path(runtime['competition'])/'test_series.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    run_id = digest([digest(reg), time.time_ns(), str(output)])
    # Offline hook installer is bounded by the SAME supervisor, never a parent hang.
    shards, jobs, exits = [], [], []
    for part, selected in [('NATIVE', [e for e in entries if e['branch'] != 'R7']),
                           ('R7', [e for e in entries if e['branch'] == 'R7'])]:
        for shard in range(2 if part == 'NATIVE' else 1):
            subset = ids[shard::2] if part == 'NATIVE' else ids
            if not subset:
                continue
            directory = output/part/str(shard)
            branch_specs = []
            for branch in dict.fromkeys(e['branch'] for e in selected):
                members = [e for e in selected if e['branch'] == branch]
                spec = dict(path=str(directory/branch), ids=subset,
                            identity=identity(reg, branch, members, run_id))
                shards.append(spec)
                branch_specs.append(spec)
            job = dict(runtime, ids=subset, entries=selected, branches=branch_specs, codec=None,
                rows=[r for r in rows if r[UID] in set(subset)], images=str(Path(runtime['competition'])/'test_series'),
                family=part, output=str(directory), device='cuda:0', gpu_mask=str(shard) if part == 'NATIVE' else '0,1', work=str(output),
                registration=reg, runtime_root=str(ROOT))
            jobs.append((directory/'job.json', job))
    # Assignment lives in memory before disk writes; failed job writes still fall back.
    with Supervisor(deadline) as supervisor:
        codec_job = output/'codec-job.json'
        try:
            dump(codec_job, dict(runtime_root=str(ROOT), output=str(output)))
            with (output/'codec.log').open('w') as log:
                supervisor.run([sys.executable, '-B', str(ROOT/'tools/vnext_r7_pl_worker_v1.py'),
                    '--codec-job', str(codec_job)], label='codec', stdout=log, stderr=log)
        except OSError:
            pass
        try:
            codec = metadata.metadata(output/'codec.json')
        except metadata.MetadataUnavailable:
            codec = dict(status='INACTIVE')
        if codec.get('status') == 'ACTIVE' and not all(isinstance(codec.get(k), str) for k in ('hook_dir', 'entry')):
            codec = dict(status='INACTIVE_CORRUPT_METADATA')
        for family in ('NATIVE', 'R7'):
            children = []
            with context_logs() as logs:
                for job_path, job in jobs:
                    if job['family'] != family:
                        continue
                    try:
                        job['codec'] = codec
                        dump(job_path, job)
                        log = logs.open(job_path.parent/'child.log')
                        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                            OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                            CUDA_VISIBLE_DEVICES=job['gpu_mask'])
                        env = base.codecs.child_env(env, codec, branch=family, telemetry_dir=output/'codec-telemetry')
                        children.append(supervisor.start([sys.executable, '-B',
                            str(ROOT/'tools/vnext_r7_pl_worker_v1.py'), '--job', str(job_path)],
                            label=str(job_path.parent), env=env, stdout=log, stderr=log))
                    except OSError:
                        continue
                for child in children:
                    supervisor.wait(child)
                    exits.append(dict(label=child.label, code=child.returncode, cleanup_error=child.cleanup_error))
    result = publish(reg, ids, shards, output, weight, filename='diagnostic.csv', diagnostic=True) if diagnostic else publish(reg, ids, shards, output, weight)
    if diagnostic:
        result.update(status='DIAGNOSTIC_ONLY', qualified_for_submission=False)
    result.update(seconds=time.monotonic()-deadline.started, exits=exits)
    try:
        dump(output/'execution.json', result)
    except OSError:
        pass
    return result


@contextmanager
def context_logs():
    from contextlib import ExitStack
    with ExitStack() as stack:
        class Logs:
            def open(self, path):
                return stack.enter_context(Path(path).open('w', encoding='utf-8'))
        yield Logs()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'prepare', 'validate', 'admit', 'bind'])
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--arm', choices=['global', 'fill'], default='global')
    parser.add_argument('--mode', choices=['fivefold', 'all-data'], default='fivefold')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--bindings', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'register':
        result = register(args.stage, args.arm, args.mode, args.seeds)
    elif args.command == 'admit':
        result = admit(args.stage, args.bindings)
    elif args.command == 'bind':
        need(args.inputs and args.out, 'bind requires --inputs and --out')
        result = bind(args.stage, args.inputs, args.out)
    else:
        result = globals()[args.command](args.stage)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
