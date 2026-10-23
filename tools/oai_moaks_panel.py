"""OAI MOAKS paired image-read panel, v1.0.3 (local research only).

Changelog 1.0.0: frozen cohort, authenticated serving, routing gate and paired CI.
Changelog 1.0.1: before scores, recognize the audited Sag0 copy series suffix.
Changelog 1.0.2: before scores, resolve sharded rows and allow authenticated
within-slot background masks while requiring every acquisition to be present.
Changelog 1.0.3: preserve the original registration and serving bytes in vendor;
isolate serving imports from live refactors, with a separate runtime lock.
Never reads report text; labels stay in memory. No training or network operations.
"""
from __future__ import annotations

import argparse
import builtins
import csv
import hashlib
import json
import re
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training/src'))

VERSION = '1.0.3'
REGISTRATION_VERSION = '1.0.2'
OUT = ROOT / 'artifacts/experiments/OAI_MOAKS_PANEL_V1'
SOURCE = 'storage/external_data/oai/clinical/extracted/mr_assess/Semi-Quant Scoring_ASCII/kMRI_SQ_MOAKS_BICL00.txt'
CACHES = {'64': 'storage/cache/derived/EXT_OAI_RAPTOR64_336_C140_v2',
          '96': 'storage/cache/derived/EXP155_OAI_RAPTOR96_384_C140_S02_v2'}
WEAK = ('Lateral Meniscus', 'PF OA', 'Lateral OA', 'Synovitis')
AUDIT = '423c8c4f6fcc87a1cd441535a9e31c5d05719f9192da18a80000be25a4ab6320'
TRAIN_IDS = 'a69580bd6a68dc802cce4f214ac7afffb9742a7f6ad7963c33245b9de72cace5'
INIT = '77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a'
SEED, DRAWS = 20261003, 1000
SERVING_SOURCES = ('tools/oai_moaks_panel.py', 'tools/extlabels.py', 'training/arms/mil.py', 'training/arms/coatnet.py',
    'training/src/raptor_common.py', 'training/src/raptor_radimagenet.py',
    'training/src/raptor_exp179.py', 'training/src/raptor_exp185.py',
    'training/src/exp143_own0949.py', 'tools/exp191_softprior.py', 'tools/exp191b_pfoa_softprior.py')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


class Vendor:
    """Private module graph; neither sys.modules aliases nor live imports leak in.

    Execute the exact hashed bytes (never a stale pyc). Redirect project imports
    inside those modules, including lazy imports and bare training/src names.
    Unknown project dependencies fail closed. This serial CLI restores any
    sys.path edits made by historical modules during their initialization.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = read(self.root / 'manifest.json')
        self.modules = {}
        self.prefix = '_oai_panel_vendor_' + hashlib.sha256(str(self.root).encode()).hexdigest()[:16]

    @staticmethod
    def canonical(name):
        if name.startswith(('raptor_', 'exp')) and '.' not in name:
            return 'training.src.' + name
        return name

    def load(self, name):
        name = self.canonical(name)
        if name in self.modules:
            return self.modules[name]
        rel = name.replace('.', '/') + '.py'
        package = name in ('tools', 'training', 'training.src', 'training.arms')
        need(package or rel in self.manifest['sources'], 'unvendored project import: ' + name)
        # These are historical provenance pins, never serving entry points.
        need(rel not in ('tools/oai_moaks_panel.py', 'tools/exp191_softprior.py',
                         'tools/exp191b_pfoa_softprior.py'), 'provenance-only vendor source: ' + name)
        parent, _, child = name.rpartition('.')
        parent_module = self.load(parent) if parent else None
        module = types.ModuleType(self.prefix + '.' + name)
        module.__package__ = module.__name__ if package else module.__name__.rpartition('.')[0]
        self.modules[name] = module
        sys.modules[module.__name__] = module  # dataclasses resolves its own namespace
        if parent_module is not None:
            setattr(parent_module, child, module)
        if package:
            module.__path__ = []
            return module
        path = self.root / rel
        data = path.read_bytes()
        need(hashlib.sha256(data).hexdigest() == self.manifest['sources'][rel]['sha256'],
             'vendored source changed: ' + rel)
        module.__file__ = str(path)
        module.__builtins__ = dict(vars(builtins), __import__=self.import_module)
        original_path = sys.path[:]
        try:
            exec(compile(data, str(path), 'exec'), module.__dict__)
        except BaseException:
            self.modules.pop(name, None)
            sys.modules.pop(module.__name__, None)
            if parent_module is not None:
                delattr(parent_module, child)
            raise
        finally:
            sys.path[:] = original_path
        return module

    def import_module(self, name, globals=None, locals=None, fromlist=(), level=0):
        canonical = self.canonical(name)
        if not level and (canonical.split('.')[0] in ('tools', 'training')):
            module = self.load(canonical)
            for item in fromlist or ():
                if item != '*' and not hasattr(module, item):
                    self.load(canonical + '.' + item)
            return module if fromlist else self.load(canonical.split('.')[0] if canonical == name else canonical)
        need(not level, 'relative vendor imports must be explicitly registered')
        return builtins.__import__(name, globals, locals, fromlist, level)


def validate_vendor(out=OUT):
    lock = read(out / 'runtime-v1.0.3.json')
    need(lock['version'] == VERSION, 'runtime version differs')
    need(sha(Path(__file__)) == lock['tool_sha256'], 'panel runtime changed')
    need(sha(out / 'registration.json') == lock['registration_sha256'], 'original registration changed')
    need(sha(out / 'vendor/manifest.json') == lock['vendor_manifest_sha256'], 'vendor manifest changed')
    manifest = read(out / 'vendor/manifest.json')
    need(manifest['original_registration_sha256'] == lock['registration_sha256'], 'vendor registration differs')
    registered = read(out / 'registration.json')['pins']
    for path, record in manifest['sources'].items():
        need(sha(out / 'vendor' / path) == record['sha256'], 'vendored source changed: ' + path)
        if record['originally_registered']:
            need(record['sha256'] == registered[path], 'original serving digest differs: ' + path)
    need({p for p in registered if p.endswith('.py')} ==
         {p for p, r in manifest['sources'].items() if r['originally_registered']}, 'bound sources missing')


def registered_path(path, out=OUT):
    return out / 'vendor' / path if path in SERVING_SOURCES else ROOT / path


validate_vendor(OUT)
vendor = Vendor(OUT / 'vendor')
labels = vendor.load('tools.extlabels')
TARGETS = labels.TARGETS


def complete(scores):
    fields = {'MACLTR': range(3), 'MEFFWK': range(4), 'MSYIC': range(4), 'MPOPCYS': range(2)}
    fields.update({f'MMT{s}{r}': range(9) for s in 'ML' for r in 'ABP'})
    fields.update({f'MMRT{s}': range(2) for s in 'ML'})
    fields.update({'MOS' + r: range(4) for regions in labels.OST_REGIONS.values() for r in regions})
    return (all(labels.grade(scores.get(f), a) is not None for f, a in fields.items())
            and all(labels.cartilage_parts(scores.get('MCM' + r)) is not None
                    for regions in labels.CART_REGIONS.values() for r in regions))


def readings():
    return labels.select_readings(labels.read_oai(ROOT / SOURCE, 'moaks'), labels.MOAKS_ORDER)[0]


def cache_metadata():
    m64, m96 = (read(ROOT / CACHES[k] / 'manifest.json') for k in ('64', '96'))
    picks = {r['uid']: r for r in map(json.loads, (ROOT / CACHES['64'] / 'picks.jsonl').read_text().splitlines())}
    return m64, m96, picks


def cohort():
    chosen = readings()
    _, m96, picks = cache_metadata()
    image_ids = set(m96['ids']) & set(picks)
    full = [k for k, r in chosen.items() if k.endswith('_V00') and k in image_ids and complete(r.scores)]
    # Pre-score acquisition eligibility: require all five rendered slots, including
    # the documented Sag0 fallback. Missing acquisitions are not negative labels.
    eligible = [k for k in full if all(s is not None and s['rendered'] > 0 for s in picks[k]['slots'])]
    return select_knees(eligible), {'selected_baseline_reads': len(chosen), 'complete_image_linked_knees': len(full),
                            'excluded_missing_acquisitions': len(full) - len(eligible), 'eligible_knees': len(eligible)}


def select_knees(eligible):
    seen, result = set(), []
    for k in sorted(eligible, key=lambda x: (x.split('_')[1], hashlib.sha256(x.encode()).hexdigest())):
        participant = k.split('_')[1]
        if participant not in seen:
            seen.add(participant)
            result.append(k)
    return sorted(result)


def outcomes(ids):
    chosen = readings()
    return np.asarray([[np.nan if (v := labels.map_moaks(chosen[k].scores)[t]) is None else v
                        for t in TARGETS] for k in ids], dtype=np.float64)


def counts(y):
    return {t: {'positive': int((y[:, j] == 1).sum()), 'negative': int((y[:, j] == 0).sum()),
                'unknown': int(np.isnan(y[:, j]).sum())} for j, t in enumerate(TARGETS)}


MAPPING = '''MOAKS BICL V00 expert MRI reads only; no POMA, X-ray, report silence, BML-to-trauma inference, or supplemental label fallback.
Complete means every native field used below has a valid grade in the selected reading. It does not mean every conservative binary OA proxy is observable.
Duplicate reads: extlabels.MOAKS_ORDER project priority, then reader R1/R2, descending version, filename, canonical native-score hash. Select reading before completeness; never select a more favorable read.

| Target | Positive | Negative / unknown |
|---|---|---|
| Medial / Lateral Meniscus | Any MMT{M/L}{A/B/P} >=2 (allowed 0..8), or posterior root MMRT{M/L}=1 | All morphology <2 and root=0; missing is unknown. Morphology includes maceration, not just an isolated tear. |
| ACL | MACLTR >=1 (allowed 0..2; partial/complete tear) | 0; missing/invalid unknown. Not an all-ACL-abnormality gold standard. |
| Effusion | MEFFWK >=2 (allowed 0..3) | 0/1. MOAKS whole-knee effusion / effusion-synovitis proxy. |
| Synovitis | MSYIC >=2 (allowed 0..3) | 0/1. Intercondylar/Hoffa signal proxy on noncontrast MRI; not contrast-enhanced synovial thickness or histology. Effusion is not substituted for MSYIC. |
| Baker's | MPOPCYS=1 | 0 (allowed 0/1); popliteal cyst proxy. |
| Medial OA | Any full-thickness cartilage >0 AND any osteophyte >=1 | All cartilage area/full=0 AND all osteophytes=0; all intermediate/discordant cases unknown. |
| Lateral OA | Same rule in lateral tibiofemoral regions | Same conservative negatives. |
| PF OA | Same rule in patellar/trochlear regions | Same conservative negatives. |
| MCL, Contusion, Fracture | Unmappable | Always unknown; never zero. |

Cartilage MCM is area.full with 0<=full<=area<=3; osteophyte MOS allowed 0..3.
Cartilage regions: Medial FMC,FMP,TMA,TMC,TMP; Lateral FLC,FLP,TLA,TLC,TLP; PF PM,PL,FMA,FLA.
Osteophytes: Medial FMC,FMP,TM; Lateral FLC,FLP,TL; PF PS,PI,PM,PL,FMA,FLA.
These OA definitions favor specificity and are MRI structural proxies with spectrum selection; they are not interchangeable with clinical OA or the competition's report labels.
'''


def register(out=OUT, dry_run=False):
    need(dry_run, 'v1.0.3 preserves the existing registration; only --dry-run is allowed')
    validate_vendor(out)
    ids, flow = cohort()
    y = outcomes(ids)
    panel = ('study_key\n' + '\n'.join(ids) + '\n').encode()
    pins = {p: sha(registered_path(p, out)) for p in (SOURCE, *SERVING_SOURCES,
        *(v + '/manifest.json' for v in CACHES.values()), CACHES['64'] + '/ids.npy',
        CACHES['64'] + '/masks.npy', CACHES['64'] + '/picks.jsonl')}
    metadata = dict(version=REGISTRATION_VERSION, panel_sha256=hashlib.sha256(panel).hexdigest(), n=len(ids),
                    targets=list(TARGETS), weak=list(WEAK), seed=SEED, draws=DRAWS, counts=counts(y), flow=flow, pins=pins)
    if dry_run:
        current = read(out / 'registration.json')
        need(all(current[k] == v for k, v in metadata.items()), 'registration reconstruction differs')
        return {'status': 'PASS', 'n': len(ids), 'counts': metadata['counts'], 'flow': flow}
    need(not out.exists() or not any(out.iterdir()), 'registration destination is not empty')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'panel.csv').open('xb') as f:
        f.write(panel)
    table = '\n'.join(f"| {t} | {v['positive']} | {v['negative']} | {v['unknown']} |" for t, v in metadata['counts'].items())
    protocol = f'''# OAI MOAKS PANEL V1 — pre-score registration

Version {VERSION}. Local research use only. Frozen before any panel model prediction or score.
Source: `{SOURCE}` (SHA256 `{pins[SOURCE]}`). Historical EXP137 QA is void (D-B-104, D-B-116); only corrected v2 caches are allowed. D-B-112 motivates image evidence for report-silent weak targets. This panel uses existing expert MRI assessments; it is not a newly convened reader study.

Baseline V00, complete mapped native MOAKS fields, present in both corrected caches, all five rendered slots available. One knee per participant chosen by smallest SHA256(study_key), then lexicographically sort final IDs. No score-based sampling. Missing acquisitions excluded before scoring. Fixed N={len(ids)}. Flow: {json.dumps(flow)}.
`panel.csv` contains IDs only. SHA256 `{metadata['panel_sha256']}`. The clinical source and serving/cache manifests are pinned in registration.json; labels are reconstructed in memory, never exported.

{MAPPING}

| Target | Positive | Negative | Unknown |
|---|---:|---:|---:|
{table}

Paired analysis: arm minus control AUC on identical target-specific observable patients. Patient bootstrap with replacement, numpy default_rng seed {SEED}, {DRAWS} draws, same patient draw for both arms and all targets. Percentile 2.5/97.5% CI; one-class draws undefined and counted. No replacement draws. A target needs >=20 of each class and >=950 valid draws for an interpretable CI. The weak-four mean (LM, PF OA, Lateral OA, Synovitis) is primary; compute only when all four deltas exist in a draw, never a varying-target nanmean. Mean seed-specific AUC deltas for S42/S43 comparisons, with shared patient draws (not AUC of averaged predictions); individual seeds descriptive. CoAtNet is S42 only. No post-score threshold/weight changes. Multiple comparisons exploratory; no confirmatory p-value or automatic promotion.

Registered pairs: EXP191 PA vs EXP185 CONTROL and P vs CONTROL, S42/S43; EXP191B PA_PFOA vs same CONTROL after collection; EXP185 MIL vs EXP179 M42/M43 mixed control; EXP181 AUG1 vs CONTROL; EXP188 COVERAGE vs EXP181 CONTROL. All own fold-0 components only, no public blend. CoAtNet checkpoints encode fold=-1 but their authenticated manifest restricts training to fold-0 train.

QA before use: authenticate corrected routing audit SHA `{AUDIT}`, both cache manifests, per-panel route plane/contrast/group agreement, masks and volume hashes. Sag0 may be an explicitly recorded fluid-series fallback; it must never be represented as a native non-FS acquisition. Reject legacy/misrouted caches. Metadata audit does not replace visual rereading of DICOM.
One fixed zero-shot sentinel: EXP179 M42 fold-0, K36 mixed center3, coronal/axial flip probability mean. On the full frozen panel require ACL, MM, Effusion AUC each >=0.50 and their mean >=0.60, finite nonconstant outputs for all 12 targets. This broad engineering sanity gate is not evidence of weak-target efficacy; failure means HOLD/investigate without retuning. Partial CPU smoke cannot pass it.

OAI-free admission: only registered checkpoint identities with exact SHA256, target order, fixed training population, offline ImageNet initialization, known report-label provenance and authenticated run manifests. Reject OAI/OAIZIB training, pseudo-label/teacher ancestry, EXP155/163/168 and public 0.949 R; unknown identity/ancestry is rejected. OAI cache use for inference is allowed. Never invoke label builders or read report text.

Serving: EXP179/183 global and EXP185/191 spatial MIL use original mixed K36 224px center3 loader, within-slot-valid centers, ImageNet normalization, mean probabilities for normal and coronal/axial horizontal mirror. CoAtNet uses original K94 adjacent3 320px center crop and its exact sagittal-channel/coronal-axial mirror, mean probabilities, fixed SWA+BN checkpoint. CPU FP32; GPU FP16, explicit device only. No model fitting, calibration or OAI tuning.

Decision: positive primary lower CI is image-panel support only; a mapped-target drop >0.010 is HOLD/REVIEW. Missing QA, sparse classes, undefined CIs or absent external harm evidence is HOLD. Preserve agents.md weak-target report primary and harm guards: macro12 delta >=-0.001, no other-target drop >0.010, Dread macro >=-0.003, plus all existing registered gates. Three unmappable targets mean this panel CANNOT establish macro12 safety. External guards remain required even after image support. Synovitis remains a noncontrast proxy. No submission authorization is conveyed.
'''
    with (out / 'PROTOCOL.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write(protocol)
    metadata['protocol_sha256'] = sha(out / 'PROTOCOL.md')
    write(out / 'registration.json', metadata)
    return {'status': 'REGISTERED', 'n': len(ids), 'counts': metadata['counts'], 'flow': flow}


def validate(out=OUT):
    validate_vendor(out)
    r = read(out / 'registration.json')
    need(r['version'] == REGISTRATION_VERSION and r['targets'] == list(TARGETS), 'registration schema differs')
    need(sha(out / 'PROTOCOL.md') == r['protocol_sha256'], 'protocol changed')
    need(sha(out / 'panel.csv') == r['panel_sha256'], 'panel changed')
    for path, digest in r['pins'].items():
        need(sha(registered_path(path, out)) == digest, 'registered source/cache changed: ' + path)
    with (out / 'panel.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    need(rows and set(rows[0]) == {'study_key'}, 'panel must contain IDs only')
    ids = [row['study_key'] for row in rows]
    need(len(ids) == r['n'] == len(set(ids)) and ids == sorted(ids), 'panel membership/order differs')
    need(all(re.fullmatch(r'OAI_\d+_[LR]_V00', x) for x in ids), 'invalid baseline ID')
    need(len({x.split('_')[1] for x in ids}) == len(ids), 'duplicate participant')
    need(counts(outcomes(ids)) == r['counts'], 'label aggregates changed')
    return ids


def check_slots(slots):
    expected = [('Sagittal', 1, 'SAG_IW_TSE', 'IW_TSE_FS'),
                ('Sagittal', 0, 'SAG_IW_TSE', 'IW_TSE_FS'),
                ('Coronal', 1, 'COR_MPR', 'DESS_WE'),
                ('Coronal', 0, 'COR_IW_TSE', 'IW_TSE_NON_FS'),
                ('Axial', -1, 'AX_MPR', 'DESS_WE')]
    need(len(slots) == 5, 'slot count differs')
    for i, (s, (plane, fluid, sequence, contrast)) in enumerate(zip(slots, expected)):
        need(s is not None and s['rendered'] > 0, 'missing panel acquisition')
        need(s['slot'] == [plane, fluid] and s['source_sequence'] == sequence
             and s['contrast_family'] == contrast, 'plane/contrast routing differs')
        if i == 1:
            need(s['fallback'] in ('copy_of_sagittal_fluid1', 'competition_unused_sagittal'), 'unregistered Sag0 fallback')
            if s['fallback'] == 'copy_of_sagittal_fluid1':
                need(s['group_key'] == slots[0]['group_key'] and
                     s['series'] == slots[0]['series'] + '#sag0_fallback_copy', 'Sag0 copy differs')
        else:
            need(s.get('fallback') is None, 'unexpected fallback')


def routing(out=OUT):
    ids = validate(out)
    m64, m96, picks = cache_metadata()
    need(m64['audit_summary_sha256'] == m96['audit_summary_sha256'] == AUDIT, 'routing audit differs')
    need(m64['gen_index_sha256'] == m96['gen_index_sha256'], 'GEN source differs')
    files96 = {r['uid']: r for r in m96['files']}
    masks = np.load(ROOT / CACHES['64'] / 'masks.npy', mmap_mode='r')
    ids64 = np.load(ROOT / CACHES['64'] / 'ids.npy').tolist()
    shards = {s['shard']: s for s in m64['shards']}
    copies, masked_slices96 = 0, 0
    for uid in ids:
        p = picks[uid]
        check_slots(p['slots'])
        shard = shards[p['shard']]
        global_row = shard['start'] + p['row']
        need(shard['start'] <= global_row < shard['stop'] and ids64[global_row] == uid
             and np.array_equal(masks[global_row], p['mask']), '64 cache row/mask differs')
        row = files96[uid]
        path = ROOT / CACHES['96'] / f"study_{row['row']:05d}.json"
        need(sha(path) == row['receipt_sha256'], '96 receipt changed')
        q = read(path)
        need(q['uid'] == uid and q['row'] == row['row'], '96 receipt row differs')
        check_slots(q['slots'])
        for a, b in zip(p['slots'], q['slots']):
            need(all(a[k] == b[k] for k in ('slot', 'source_sequence', 'contrast_family', 'group_key', 'series', 'fallback')),
                 'cross-cache routing differs')
        for rec, spec in ((p, m64['spec']), (q, m96['spec'])):
            check_mask(rec['mask'], spec)
        masked_slices96 += sum(v == 0 for v in q['mask'])
        need(p['gen_sha256'] == q['gen_sha256'], 'per-knee GEN differs')
        copies += p['slots'][1]['fallback'] == 'copy_of_sagittal_fluid1'
    return {'status': 'PASS', 'knees': len(ids), 'sag0_fluid_copy': copies,
            'sag0_unused_fluid_fallback': len(ids)-copies, 'registration_sha256': sha(out / 'registration.json'),
            'masked_background_slices96': masked_slices96,
            'note': 'Metadata routing and receipt audit; volume hashes verified on inference reads.'}


def check_mask(mask, spec):
    mask = np.asarray(mask)
    need(mask.shape == (spec['depth'],) and np.isin(mask, (0, 1)).all(), 'invalid cache mask')
    offset = 0
    for _, _, count in spec['slots']:
        need(mask[offset:offset+count].any(), 'missing masked acquisition')
        offset += count
    need(offset == len(mask), 'mask depth differs')


class Cache:
    def __init__(self, kind):
        self.kind, self.root = kind, ROOT / CACHES[kind]
        self.manifest = read(self.root / 'manifest.json')
        self.spec = self.manifest['spec']
        ids = np.load(self.root / 'ids.npy').tolist() if kind == '64' else self.manifest['ids']
        self.rows = {uid: i for i, uid in enumerate(ids)}
        self.masks = np.load(self.root / 'masks.npy', mmap_mode='r') if kind == '64' else None
        self.verified = set()

    def get(self, uid):
        row = self.rows[uid]
        if self.kind == '64':
            record = next(s for s in self.manifest['shards'] if s['start'] <= row < s['stop'])
            path = self.root / record['file']
            mask = self.masks[row]
        else:
            record = self.manifest['files'][row]
            need(record['uid'] == uid and record['row'] == row, '96 manifest row differs')
            path = self.root / f'study_{row:05d}.npy'
            mask = np.asarray(read(path.with_suffix('.json'))['mask'])
        if path not in self.verified:
            need(sha(path) == record['sha256'], 'volume hash differs')
            self.verified.add(path)
        data = np.load(path, mmap_mode='r')
        vol = data[row-record['start']] if self.kind == '64' else data
        size = (64, 336, 336) if self.kind == '64' else (96, 384, 384)
        need(vol.shape == size and vol.dtype == np.uint8 and mask.shape == (size[0],), 'cache shape/dtype differs')
        return vol, mask

    def mask(self, uid):
        return self.masks[self.rows[uid]]


def candidates():
    base = 'artifacts/experiments/'
    result = {}
    for seed in (42, 43):
        result[f'EXP179_M{seed}'] = (f'{base}EXP179_MULTISCALE/outputs/M{seed}/results/EXP179_M{seed}_F0/model_best.pt', 'global')
        result[f'EXP185_S{seed}'] = (f'{base}EXP185_FINDING_MIL/outputs/S{seed}/results/EXP185_FINDING_MIL_F0_S{seed}/model_best.pt', 'mil')
        for arm in ('P', 'PA'):
            result[f'EXP191_{arm}_S{seed}'] = (f'{base}EXP191_SOFTPRIOR/outputs/{arm}-S{seed}/results/EXP191_SOFTPRIOR_{arm}_S{seed}_F0/model_best.pt', 'mil')
        result[f'EXP191B_PA_PFOA_S{seed}'] = (f'{base}EXP191B_PFOA_SOFTPRIOR/outputs/PA_PFOA-S{seed}/results/EXP191B_PFOA_SOFTPRIOR_PA_PFOA_S{seed}_F0/model_best.pt', 'mil')
    for exp, arm in (('EXP181_COATNET_F0_BRIDGE', 'CONTROL'), ('EXP181_COATNET_F0_BRIDGE', 'AUG1'), ('EXP188_COATNET_COVERAGE_F0', 'COVERAGE')):
        result[exp[:6] + '_' + arm] = (f'{base}{exp}/{arm}/model_best.pt', 'coatnet')
    return result


def load_checkpoint(path):
    import torch
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location='cpu', weights_only=True)


def deny_ancestry(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k).lower() in ('oai', 'oaizib', 'oai_trained'):
                need(v is False, 'OAI ancestry is forbidden or unknown')
            else:
                deny_ancestry(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            deny_ancestry(v)
    elif isinstance(value, str):
        need(not re.search(r'(?i)\bOAI\b|OAIZIB|EXP(?:155|163|168)|public.?0\.949', value), 'OAI/public-R ancestry forbidden')


def inspect_candidate(key):
    need(key in candidates(), 'unregistered model identity')
    rel, family = candidates()[key]
    path = ROOT / rel
    ck = load_checkpoint(path)
    need(ck['lab'] == list(TARGETS), 'checkpoint target order differs')
    meta = {k: v for k, v in ck.items() if k != 'model'}
    deny_ancestry(meta)
    provenance_path = path.parent / 'provenance.json'
    p = read(provenance_path)
    deny_ancestry(p)
    need(p['protocol_sha256'] == ck['provenance']['protocol_sha256'], 'checkpoint/run protocol differs')
    if family == 'coatnet':
        manifest_path = path.parent.parent / 'protocol-v1' / f"manifest-{ck['tag']}.json"
        m = read(manifest_path)
        need(sha(manifest_path) == p['protocol_sha256'], 'CoAtNet protocol hash differs')
        need(m['fold']['fold'] == 0 and m['fold']['train_sha256'] == TRAIN_IDS, 'CoAtNet training population differs')
        need(m['init_provenance']['oai'] is False and m['label_provenance']['oai'] is False
             and m['recipe']['oai'] is False and ck.get('oai') is False, 'CoAtNet ancestry absent')
        deny_ancestry(m['init_provenance']); deny_ancestry(m['label_provenance'])
        need(ck['checkpoint_kind'] == 'fixed-swa-bn' and ck['inference_res'] == 320
             and ck['k_eval'] == 94 and ck['swa_epochs_1based'] == m['recipe']['swa_1based'], 'CoAtNet serving differs')
        need(p['initial_state_sha256'] == m['expected_initial_state_sha256'], 'CoAtNet init differs')
        need(p['initial_state_sha256'] == '8064b3a56c602e5e52abd3a10b77f235300dd7008f56fc4d7d47fca75e5d3c43', 'unknown CoAtNet initialization')
        need(m['label_provenance']['csv_sha256'] == '79ce307e246cc54a9a90f32fd1ab21d58ebd23b4b7e7b1a80207adcea1bcc2c9', 'unknown CoAtNet labels')
        need(ck['tag'] == key.split('_', 1)[1], 'CoAtNet arm identity differs')
        seed = ck['seed']
    else:
        manifest_path = path.parents[2] / 'manifest.json'
        m = read(manifest_path)
        entries = list(m.get('arms', m.get('runs', {})).values())
        need(any(v['sha256'] == p['protocol_sha256'] for v in entries), 'protocol not in exported manifest')
        need(ck['fold'] == 0 and p['train_ids_ordered_sha256'] == TRAIN_IDS
             and ck['init_weights_sha256'] == INIT and ck['init'] == 'timm-in22k', 'training population/init differs')
        need(p['initial_state_matches_registered'] is True and p['training_mode'] == 'full', 'training provenance incomplete')
        need(ck['window_policy'] == 'within-slot-valid' and ck['input_policy'] == 'center3'
             and ck['norm'] == 'imagenet' and ck['res'] == 224 and p['k_eval_extra'] == 36, 'ConvNeXt serving differs')
        need(p['exp179']['crop_policy'] == 'mixed', 'mixed-scale provenance absent')
        need((ck.get('model_kind') == 'finding_spatial_tokens_v1') == (family == 'mil'), 'pooling family differs')
        allowed_labels = {'l2_dread75_aquila25_v1.csv', 'l2_exp191_softprior_p.csv',
                          'l2_exp191_softprior_pa.csv', 'l2_exp191b_pa_pfoa.csv'}
        need(Path(p['label_file']).name in allowed_labels, 'unknown training label/teacher ancestry')
        expected_label = ('l2_exp191b_pa_pfoa.csv' if key.startswith('EXP191B_') else
                          'l2_exp191_softprior_pa.csv' if key.startswith('EXP191_PA_') else
                          'l2_exp191_softprior_p.csv' if key.startswith('EXP191_P_') else
                          'l2_dread75_aquila25_v1.csv')
        need(Path(p['label_file']).name == expected_label, 'arm training labels differ')
        need(p['label_file_sha256'] == m['files'][p['label_file']]['sha256'], 'label hash not bound to manifest')
        need(p['arch'] == ck['arch'] == 'convnext_tiny.fb_in22k_ft_in1k', 'unknown backbone')
        seed = p['seed']
        need(seed == int(key[-2:]), 'seed identity differs')
    evidence = {str(v.relative_to(ROOT)).replace('\\', '/'): sha(v) for v in (manifest_path, provenance_path)}
    return {'key': key, 'checkpoint': rel, 'sha256': sha(path), 'family': family, 'seed': seed,
            'targets': list(TARGETS), 'k': 94 if family == 'coatnet' else 36,
            'cache': '96' if family == 'coatnet' else '64', 'tta': 'mean_probability',
            'precision_cpu': 'fp32', 'precision_gpu': 'fp16', 'evidence': evidence,
            'ancestry': 'known fold-0 competition train; offline ImageNet; report-derived labels; no image teacher or OAI training'}


def prepare(out=OUT):
    validate(out)
    result, missing = {}, []
    for key, (path, _) in candidates().items():
        if (ROOT / path).exists():
            result[key] = inspect_candidate(key)
        else:
            missing.append(key)
    return {'version': VERSION, 'registration_sha256': sha(out / 'registration.json'),
            'recipes': result, 'missing': missing}


def admit(recipe_file, key, out=OUT):
    validate(out)
    recipe = read(recipe_file)
    need(recipe['registration_sha256'] == sha(out / 'registration.json'), 'recipe registered to another panel')
    need(key in recipe['recipes'], 'checkpoint not admitted; prepare after collection')
    actual = inspect_candidate(key)
    need(actual == recipe['recipes'][key], 'checkpoint or serving/ancestry recipe changed')
    return actual


def model_for(recipe):
    mil = vendor.load('training.arms.mil')
    FindingMIL, strict_load = mil.FindingMIL, mil.strict_load
    common = vendor.load('training.src.raptor_common')
    ck = load_checkpoint(ROOT / recipe['checkpoint'])
    if recipe['family'] == 'coatnet':
        coatnet = vendor.load('training.arms.coatnet')
        model = coatnet.build_model(coatnet.Config(), res=320)
    elif recipe['family'] == 'mil':
        model = FindingMIL(common.build_backbone(ck['arch'], pretrained=False))
    else:
        model = common.build_model(ck['arch'], pretrained=False)
    strict_load(model, ck['model'])
    return model.eval()


def predict_one(model, recipe, cache, uid, device='cpu'):
    import torch
    mil = vendor.load('training.arms.mil')
    select_centers = vendor.load('training.src.raptor_common').select_centers
    scale_windows = vendor.load('training.src.raptor_exp179').scale_windows
    normalise = vendor.load('training.src.raptor_radimagenet').normalise
    vol, mask = cache.get(uid)
    with torch.inference_mode():
        if recipe['family'] == 'coatnet':
            coatnet = vendor.load('training.arms.coatnet')
            a, b = coatnet.infer_pair(model, coatnet.Config(), vol, mask, device=device,
                precision='none' if device == 'cpu' else 'fp16', chunk=2 if device == 'cpu' else 16)
            return (a + b) * np.float32(.5)
        centers = select_centers(mask, 64, 36, cache.spec, 'within-slot-valid')
        x = torch.from_numpy(normalise(scale_windows(vol, centers, 224, 'mixed'), 'imagenet'))[None]
        views = (x, mil.flip_inputs(x, [uid], cache, 36))
        probs = []
        for view in views:
            with mil.autocast(device, 'fp16' if device == 'cuda' else 'none'):
                # Preserve all-token attention: chunk only the backbone, not heads.
                chunks = view.split(2 if device == 'cpu' else 12, dim=1)
                encoded = [model.encode(z.to(device)) for z in chunks]
                if recipe['family'] == 'mil':
                    # Padding validity is global; empty chunks cannot introduce
                    # their own fallback tokens into a populated study.
                    valid_windows = vendor.load('training.src.raptor_exp185').valid_windows
                    tokens = torch.cat([e[0] for e in encoded], 1)
                    valid = valid_windows(view.to(device)).repeat_interleave(tokens.shape[1] // view.shape[1], 1)
                    encoded = (tokens, valid)
                else:
                    encoded = torch.cat(encoded, 1)
                logits = model.head(encoded).float()
            probs.append(logits.sigmoid()[0].cpu().numpy())
        return (probs[0] + probs[1]) * np.float32(.5)


def infer(recipe_file, key, destination, out=OUT, device='cpu', limit=None, threads=4):
    import torch
    need(device in ('cpu', 'cuda'), 'device must be explicit cpu/cuda')
    need(limit is None or limit > 0, 'invalid smoke size')
    need(not Path(destination).exists() and not Path(str(destination) + '.json').exists(), 'prediction output exists')
    torch.set_num_threads(threads)
    ids = validate(out)
    route = routing(out)
    recipe = admit(recipe_file, key, out)
    use_ids = ids if limit is None else ids[:limit]
    started = time.monotonic()
    model = model_for(recipe).to(device)
    cache = Cache(recipe['cache'])
    predictions = np.asarray([predict_one(model, recipe, cache, uid, device) for uid in use_ids])
    need(predictions.shape == (len(use_ids), 12) and np.isfinite(predictions).all()
         and ((predictions >= 0) & (predictions <= 1)).all(), 'invalid probabilities')
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with Path(destination).open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['study_key', *TARGETS])
        writer.writerows([uid, *map(float, row)] for uid, row in zip(use_ids, predictions))
    receipt = {'version': VERSION, 'key': key, 'recipe': recipe, 'recipe_file_sha256': sha(recipe_file),
               'registration_sha256': sha(out / 'registration.json'), 'protocol_sha256': sha(out / 'PROTOCOL.md'),
               'predictions_sha256': sha(destination), 'rows': len(use_ids), 'full_panel': use_ids == ids,
               'device': device, 'seconds': time.monotonic()-started, 'routing': route,
               'volume_files_verified': len(cache.verified), 'tool_sha256': sha(Path(__file__)),
               'runtime_lock_sha256': sha(out / 'runtime-v1.0.3.json'),
               'vendor_manifest_sha256': sha(out / 'vendor/manifest.json')}
    write(str(destination) + '.json', receipt)
    return {k: receipt[k] for k in ('key', 'rows', 'full_panel', 'device', 'seconds', 'volume_files_verified')}


def prediction(path, ids, out=OUT, smoke=False):
    receipt = read(str(path) + '.json')
    if receipt.get('version') == VERSION:
        need(receipt['runtime_lock_sha256'] == sha(out / 'runtime-v1.0.3.json')
             and receipt['vendor_manifest_sha256'] == sha(out / 'vendor/manifest.json'),
             'prediction serving snapshot differs')
    need(sha(path) == receipt['predictions_sha256'], 'predictions changed')
    need(receipt['registration_sha256'] == sha(out / 'registration.json') and receipt['protocol_sha256'] == sha(out / 'PROTOCOL.md'), 'prediction registration differs')
    need(receipt['routing']['status'] == 'PASS', 'routing not passed')
    need(smoke or receipt['full_panel'], 'partial smoke predictions cannot be screened')
    need(receipt['recipe'] == inspect_candidate(receipt['key']), 'prediction model recipe no longer authentic')
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames == ['study_key', *TARGETS], 'prediction target order differs')
        rows = list(reader)
    got = [r['study_key'] for r in rows]
    need(len(got) == len(set(got)) and got == ids, 'prediction IDs missing/duplicate/out of order')
    p = np.asarray([[float(row[t]) for t in TARGETS] for row in rows])
    need(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'invalid probability')
    return p, receipt


def auc(y, p):
    from scipy.stats import rankdata
    valid = np.isfinite(y)
    y, p = y[valid], p[valid]
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if not pos or not neg:
        return np.nan
    ranks = rankdata(p, method='average')
    return float((ranks[y == 1].sum() - pos*(pos+1)/2)/(pos*neg))


def paired(y, arms, controls, draws=DRAWS, seed=SEED):
    need(arms.shape == controls.shape and arms.ndim == 3 and arms.shape[1:] == y.shape, 'paired shape differs')
    need(np.isfinite(arms).all() and np.isfinite(controls).all(), 'nonfinite predictions')
    def scores(indices):
        a = np.asarray([[auc(y[indices, j], p[indices, j]) for j in range(12)] for p in arms])
        c = np.asarray([[auc(y[indices, j], p[indices, j]) for j in range(12)] for p in controls])
        return a.mean(0), c.mean(0)
    point_a, point_c = scores(np.arange(len(y)))
    boot = np.empty((draws, 12))
    rng = np.random.default_rng(seed)
    for i in range(draws):
        a, c = scores(rng.integers(0, len(y), len(y)))
        boot[i] = a-c
    def estimate(value, samples, eligible):
        finite = samples[np.isfinite(samples)]
        return {'delta': float(value) if np.isfinite(value) else None,
                'ci95': [float(x) for x in np.percentile(finite, [2.5, 97.5])] if eligible and len(finite) >= .95*draws else None,
                'valid_draws': len(finite), 'undefined_draws': draws-len(finite),
                'eligible': bool(eligible and len(finite) >= .95*draws)}
    sufficient = [(y[:, j] == 1).sum() >= 20 and (y[:, j] == 0).sum() >= 20 for j in range(12)]
    per = {t: {**estimate(point_a[j]-point_c[j], boot[:, j], sufficient[j]),
               'arm_auc': float(point_a[j]) if np.isfinite(point_a[j]) else None,
               'control_auc': float(point_c[j]) if np.isfinite(point_c[j]) else None} for j, t in enumerate(TARGETS)}
    weak = [TARGETS.index(t) for t in WEAK]
    primary = estimate((point_a-point_c)[weak].mean(), boot[:, weak].mean(1), all(sufficient[j] for j in weak))
    return {'per_target': per, 'weak_four': primary, 'counts': counts(y), 'draws': draws, 'seed': seed}


def sanity(path, out=OUT, smoke=False):
    ids = validate(out)
    if smoke:
        ids = ids[:read(str(path)+'.json')['rows']]
    p, receipt = prediction(path, ids, out, smoke)
    need(receipt['key'] == 'EXP179_M42', 'only fixed sentinel may establish sanity')
    y = outcomes(ids)
    scores = {t: auc(y[:, j], p[:, j]) for j, t in enumerate(TARGETS)}
    sentinels = [scores[t] for t in ('ACL', 'Medial Meniscus', 'Effusion')]
    passed = (not smoke and all(np.isfinite(s) and s >= .5 for s in sentinels)
              and np.mean(sentinels) >= .6 and (np.ptp(p, axis=0) > 1e-6).all())
    return {'status': 'PASS' if passed else 'HOLD', 'smoke': smoke,
            'registration_sha256': sha(out / 'registration.json'), 'prediction': str(Path(path).resolve()),
            'predictions_sha256': sha(path), 'auc': {t: v if np.isfinite(v) else None for t, v in scores.items()},
            'reason': 'fixed full-panel sanity gate' if not smoke else 'partial CPU smoke never authorizes use'}


def valid_pair(arm, control):
    patterns = [('EXP191_PA_S', 'EXP185_S'), ('EXP191_P_S', 'EXP185_S'),
                ('EXP191B_PA_PFOA_S', 'EXP185_S'), ('EXP185_S', 'EXP179_M')]
    for a, c in patterns:
        for seed in (42, 43):
            if (arm, control) == (a+str(seed), c+str(seed)):
                return a, seed
    for pair in [('EXP181_AUG1', 'EXP181_CONTROL'), ('EXP188_COVERAGE', 'EXP181_CONTROL')]:
        if (arm, control) == pair:
            return pair[0], 42
    raise ValueError('unregistered arm/control pair')


def screen(arm_paths, control_paths, gate_path=None, out=OUT, smoke=False):
    ids = validate(out)
    need(len(arm_paths) == len(control_paths) and len(arm_paths) > 0, 'pair count differs')
    if smoke:
        ids = ids[:read(str(arm_paths[0])+'.json')['rows']]
    arms, controls, pairs = [], [], []
    for a, c in zip(arm_paths, control_paths):
        pa, ra = prediction(a, ids, out, smoke); pc, rc = prediction(c, ids, out, smoke)
        pairs.append(valid_pair(ra['key'], rc['key']))
        need(ra['device'] == rc['device'], 'paired inference devices differ')
        arms.append(pa); controls.append(pc)
    need(len({p[0] for p in pairs}) == 1 and len(set(pairs)) == len(pairs), 'mixed/duplicate comparisons')
    if not smoke:
        expected = {42} if pairs[0][0] in ('EXP181_AUG1', 'EXP188_COVERAGE') else {42, 43}
        need({p[1] for p in pairs} == expected, 'both preregistered seeds required')
        need(gate_path is not None, 'zero-shot gate is required')
        gate = read(gate_path)
        need(gate['status'] == 'PASS' and not gate['smoke'], 'zero-shot gate not passed')
        need(sanity(gate['prediction'], out) == gate, 'sanity gate cannot be reproduced')
    result = paired(outcomes(ids), np.asarray(arms), np.asarray(controls))
    primary = result['weak_four']
    harm = [t for t, v in result['per_target'].items() if v['delta'] is not None and v['delta'] < -.010]
    support = not smoke and primary['eligible'] and primary['ci95'][0] > 0 and not harm
    result.update(status='HOLD', image_support=bool(support), mapped_harm=harm, smoke=smoke,
                  comparison=pairs, registration_sha256=sha(out / 'registration.json'),
                  reason='External macro12/other-target/Dread harm guards and original registered gates still required; panel cannot authorize promotion.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-dir', type=Path, default=OUT)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('register'); r.add_argument('--dry-run', action='store_true')
    sub.add_parser('validate')
    r = sub.add_parser('routing'); r.add_argument('--output', type=Path)
    r = sub.add_parser('prepare'); r.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('admit'); r.add_argument('--recipes', type=Path, required=True); r.add_argument('--key', required=True)
    r = sub.add_parser('infer'); r.add_argument('--recipes', type=Path, required=True); r.add_argument('--key', required=True)
    r.add_argument('--output', type=Path, required=True); r.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    r.add_argument('--limit', type=int); r.add_argument('--threads', type=int, default=4)
    r = sub.add_parser('sanity'); r.add_argument('--predictions', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True); r.add_argument('--smoke', action='store_true')
    r = sub.add_parser('screen'); r.add_argument('--arms', type=Path, nargs='+', required=True)
    r.add_argument('--controls', type=Path, nargs='+', required=True); r.add_argument('--gate', type=Path)
    r.add_argument('--output', type=Path, required=True); r.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); out = args.panel_dir
    if args.command == 'register': result = register(out, args.dry_run)
    elif args.command == 'validate': result = {'status': 'PASS', 'knees': len(validate(out))}
    elif args.command == 'routing':
        result = routing(out)
        if args.output: write(args.output, result)
    elif args.command == 'prepare':
        result = prepare(out); write(args.output, result)
        result = {'status': 'PREPARED', 'admitted': list(result['recipes']), 'missing': result['missing']}
    elif args.command == 'admit':
        result = admit(args.recipes, args.key, out)
        result = {'status': 'ADMITTED', 'key': args.key, 'sha256': result['sha256']}
    elif args.command == 'infer': result = infer(args.recipes, args.key, args.output, out, args.device, args.limit, args.threads)
    elif args.command == 'sanity':
        result = sanity(args.predictions, out, args.smoke); write(args.output, result)
    else:
        result = screen(args.arms, args.controls, args.gate, out, args.smoke); write(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
