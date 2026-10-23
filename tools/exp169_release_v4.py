"""EXP169 v4 offline A4/B4 builder. No Kaggle account action is performed.

S43 is deliberately a hard dependency. A stage made before its authenticated
average exists is a reviewable, non-runnable placeholder and must be restaged.
Run as a module (python -m tools.exp169_release_v4 ...); the script form
cannot import the tools package.

Changelog
- 4.1.3 (2026-10-11): remove every v4 internal wall-clock deadline,
  remaining-time gate and A4/B4 notebook alarm. Keep timing and pilot
  projections as diagnostics. Kaggle's 9 h runtime limit applies. Pass an
  infinite deadline to the frozen partner producer/consumers for both pilot
  and full serving; keep all correctness and atomic-publication checks.
- 4.1.2 (2026-10-11): raise the v4 hidden-run wall-clock guard to 30,000 s
  (8 h 20 m), including the A4/B4 notebook alarm and all serving checks.
  The guard is independent of the frozen v2/v3 constants; pilot projections
  remain diagnostic only. Kaggle's 9 h limit leaves about 40 m for setup and
  T4 variance.
- 4.1.1 (2026-10-11), fix for the v4.1.0 independent verification FAIL:
  * overlay_c4 replaced the extracted B172 release-manifest.json with the
    frozen create-only write_json, so every A4/B4 run raised FileExistsError
    after the anchor passes (inherited from 4.0.0). It is now replaced
    atomically (_replace_json).
  * serve also rejects a stale partner-raw.npz up front (the other
    create-only output in the serving path) instead of after the partner pass.
  * Markdown cell 0 states the real S43/timing status instead of the stale
    placeholder sentence.
  * Tests now execute overlay_c4 on the real B172/C4 archives (plus a CPU
    strict load of the five C4 checkpoints through the patched loader) and
    run the generated A4/B4 cells end to end with GPU work mocked.
- 4.1.0 (2026-10-11), fixes for the astra relv4 verification FAIL:
  * Hidden-run timing: the pilot projection is diagnostic only. Serving is
    rejected only when the then-current 24,300 s guard is actually exhausted (remaining
    time from the accounted anchor + pilot + serve seconds, or from the real
    notebook start when supplied, whichever is larger).
  * Outputs: a file present when an invocation starts is never deleted or
    rewritten. The final CSV is staged, every check (including the wall-clock
    guard) completes, and only then is it published by one atomic rename.
    Notebook handlers remove only an uncertified CSV created by the current
    invocation and never overwrite a pre-existing or certified receipt.
  * Failure provenance: the notebook failure envelope binds the v4 code
    manifest instead of A172's / B174's manifest and verification note.
  * A4 trio arithmetic: A172's exact blend3 order and its float32 CSV
    quantisation, so S43 := S42 reproduces A172 exactly (float64 summation
    noise split exact ties in 4.0.0; no member computation changed).
  * C4 training/serving provenance and a CPU window-parity check are
    recorded in the stage receipt; hidden-runtime estimates in timing plans.
- 4.0.0 (2026-10-11): first A4/B4 build.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap
import time

try:
    from tools import exp169_release_v2 as v2
except ImportError:
    import exp169_release_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
VERSION = '4.1.3'
SCHEMA = 'exp169-release-v4'
S43_REL = 'artifacts/experiments/EXP182_OWN0949_S43'
S43_RUN = S43_REL + '/S43'
S43_NAME = 'exp182-s43-e14-e11-e12-bn.pt'
S42_SHA = 'a163f3f8fa7ae57a32ea5237ce36501f74331d7fcec826bc50eb904a8e2f3295'
C4_DATASET = 'taeyangg4/knee-exp169-c4-v4-assets'
S43_DATASET = 'taeyangg4/knee-exp169-s43-v4-assets'
CODE_DATASET = 'taeyangg4/knee-exp169-v4-code'
CODE_ARCHIVE = 'exp169-v4-code.zip.bin'
C4_ARCHIVE = 'exp169-c4-v4.zip.bin'
S43_ARCHIVE = 'exp169-s43-v4.zip.bin'
A172 = 'artifacts/releases/EXP169_A172_v2_0_2_stage'
B172 = 'artifacts/releases/EXP169_B172_v2_0_2_stage'
B174 = 'artifacts/releases/EXP169_B174_v3_0_0_verified_stage'
TOL_REL = 'artifacts/experiments/EXP169_RELEASE_V2/pre-reading/partner-crossgpu-tol.json'
RECEIPT = 'exp169-executed-candidate.json'
STAGED_CSV = 'exp169-v4-submission.staged.csv'
KAGGLE_SECONDS = 9 * 3600  # platform limit, informational; never enforced by v4
# Private datasets already uploaded from EXP169_V4_A4_B4_s43_stage (v4.0.0).
# Their archives must stay byte-identical; only the small code dataset changes.
UPLOADED = {C4_DATASET: dict(archive=C4_ARCHIVE, bytes=560952802,
                             sha256='df11aa92535c9487ce6852b3ac159a162a6fb8cea989ac5a625d35faff4dd634'),
            S43_DATASET: dict(archive=S43_ARCHIVE, bytes=293031743,
                              sha256='887e294832943b8a0b5afa937459167d4425d761c05c1a154623784d60c81ef5')}
C4_LOADER = 'training/src/raptor_exp179.py'
FAILURE_MARKER = '\nexcept BaseException as _failure_error:\n'
C4_SCREEN = 'artifacts/experiments/EXP183_MULTISCALE_5FOLD/screen/result.json'
TIMING_REL = 'artifacts/experiments/EXP176_RELEASE_TIMING/stage-v1/report/report.json'
need, sha, read_json, write_json = v2.need, v2.sha, v2.read_json, v2.write_json


def _prob(a, name):
    import numpy as np
    a = np.asarray(a, np.float64)
    need(a.ndim == 2 and a.shape[0] > 0 and a.shape[1] == 12 and
         np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), 'Invalid probabilities: ' + name)
    return a


def own_values(s42, s43):
    a, b = _prob(s42, 'S42'), _prob(s43, 'S43')
    need(a.shape == b.shape, 'OWN seed shapes differ')
    return (a + b) / 2


def q4_values(scores):
    need(set(scores) == {'l2', 'dino', 'convnext'}, 'Q4 family set differs')
    # v2.aggregate averages member probabilities before ranking each family;
    # its last rank is exactly r((r(R)+r(C4)+r(D))/3).
    return v2.aggregate(scores)[0]


def trio_values(public, own, exp155):
    """A172's blend3 with OWN in the old EXP143 member slot, bit for bit.

    exp155_own0949_oai_release_v12.output_values sums [r(EXP155), r(public),
    r(member)] in that order and divides by 3.0; the public serving writes
    those values as float32 and A172 ranks that CSV. The float32 step merges
    float64 summation noise, so exactly equal rank sums stay tied (4.0.0's
    float64 sum in another order split such ties and did not reproduce A172).
    """
    import numpy as np
    ranks = [v2.rank(exp155), v2.rank(public), v2.rank(own)]
    return np.asarray(sum(ranks) / 3.0, np.float32).astype(np.float64)


def final_values(candidate, s42, s43, q4, *, public=None, exp155=None, exp174=None):
    import numpy as np
    own, q = own_values(s42, s43), _prob(q4, 'Q4')
    need(own.shape == q.shape, 'OWN/Q4 cohort differs')
    if candidate == 'A4':
        pub, e155 = _prob(public, 'public'), _prob(exp155, 'EXP155')
        need(pub.shape == own.shape == e155.shape, 'A4 cohort differs')
        # Same expression as A172: .85 * rank(trio CSV) + .15 * rank(Q).
        result = .85 * v2.rank(trio_values(pub, own, e155)) + .15 * v2.rank(q)
    elif candidate == 'B4':
        e174 = _prob(exp174, 'EXP174')
        need(e174.shape == own.shape, 'B4 cohort differs')
        result = .35 * v2.rank(own) + .35 * v2.rank(e174) + .30 * v2.rank(q)
    else:
        raise ValueError('Unknown v4 candidate')
    return _prob(result, candidate)


def c4_sources(root=ROOT):
    result = []
    for fold in range(5):
        tag = f'EXP179_M42_F0' if fold == 0 else f'EXP183_M42_F{fold}'
        parent = (root/'artifacts/experiments/EXP179_MULTISCALE/outputs/M42' if fold == 0 else
                  root/f'artifacts/experiments/EXP183_MULTISCALE_5FOLD/outputs/fold{fold}')
        check = parent/'results'/tag/'model_best.pt'
        inventory = parent/'results'/tag/'output-inventory.json'
        validated = parent/('validated-M42.json' if fold == 0 else f'validated-fold{fold}.json')
        vr, inv = read_json(validated), read_json(inventory)
        need(vr['status'] == 'VALIDATED_SUCCESS' and (fold == 0 or vr['fold'] == fold) and
             vr['output_sha256']['model_best.pt'] == inv['model_best.pt']['sha256'] == sha(check) and
             inv['model_best.pt']['bytes'] == check.stat().st_size,
             'C4 validated checkpoint differs: ' + tag)
        result.append(dict(family='convnext', seed=42, fold=fold, tag=tag,
                           path=f'models/convnext_s42_f{fold}.pt',
                           source=str(check.relative_to(root)), source_sha256=sha(check),
                           sha256=sha(check), bytes=check.stat().st_size,
                           format='authenticated-original', crop_policy='mixed',
                           validated_sha256=sha(validated), inventory_sha256=sha(inventory)))
    need(len({e['sha256'] for e in result}) == 5, 'C4 folds are not distinct')
    return result


def c4_training_evidence(root=ROOT):
    """Bind each C4 checkpoint to its validated EXP179/EXP183 training record.

    model_best.pt carries no crop_policy, so the policy is established from
    the validated run receipt, which hashes the checkpoint together with the
    run's provenance/summary (stamped with the registered EXP179 protocol)
    and the K36 flip-TTA OOF that D-B-117 screened.
    """
    root = Path(root)
    loader_sha = sha(root/C4_LOADER)
    serving_code = root/B172/'assets/code'
    screen = read_json(root/C4_SCREEN)
    folds = []
    for entry in c4_sources(root):
        fold, tag = entry['fold'], entry['tag']
        parent = (root/'artifacts/experiments/EXP179_MULTISCALE/outputs/M42' if fold == 0 else
                  root/f'artifacts/experiments/EXP183_MULTISCALE_5FOLD/outputs/fold{fold}')
        run = parent/'results'/tag
        validated = read_json(parent/('validated-M42.json' if fold == 0 else f'validated-fold{fold}.json'))
        hashes = validated['output_sha256']
        for name in ('model_best.pt', 'provenance.json', 'summary.json', 'oof_k36_flip_tta.csv'):
            need(sha(run/name) == hashes[name], f'C4 training output differs: {tag}/{name}')
        prov, summary = read_json(run/'provenance.json'), read_json(run/'summary.json')
        protocol, args = prov['exp179'], summary['args']
        need(validated['status'] == 'VALIDATED_SUCCESS' and summary['exp179'] == protocol and
             hashes['model_best.pt'] == entry['sha256'] and
             protocol['arm'] == 'M42' and protocol['crop_policy'] == 'mixed' and
             protocol['allocation'] == 'even=context,odd=detail' and
             protocol['detail_slice'] == [48, 288] and protocol['nominal_mm'] == [140, 100] and
             protocol['loader_sha256'] == loader_sha and
             prov['flip_tta'] == 'coronal_axial_horizontal_mean_probability' and
             prov['k_eval_extra'] == 36 and prov['input_policy'] == 'center3' and
             prov['norm'] == 'imagenet' and prov['seed'] == 42 and prov['fold'] == fold and
             prov['arch'] == 'convnext_tiny.fb_in22k_ft_in1k' and
             args['window_policy'] == 'within-slot-valid' and args['res'] == 224 and
             args['eval_amp'] == 'fp16' and
             prov['radimagenet_module_sha256'] == sha(serving_code/'raptor_radimagenet.py') and
             all(prov['code_sha256'][name] == sha(serving_code/name)
                 for name in ('raptor_common.py', 'raptor_runtime.py')) and
             (fold == 0 or prov['exp183']['fold'] == fold) and
             screen['input_files'][f'mixed_f{fold}']['sha256'] == hashes['oof_k36_flip_tta.csv'],
             'C4 training/serving record differs: ' + tag)
        folds.append(dict(fold=fold, tag=tag, checkpoint_sha256=entry['sha256'],
            validated_sha256=entry['validated_sha256'],
            provenance_sha256=hashes['provenance.json'], summary_sha256=hashes['summary.json'],
            oof_k36_flip_tta_sha256=hashes['oof_k36_flip_tta.csv'],
            crop_policy=protocol['crop_policy'], allocation=protocol['allocation'],
            detail_slice=protocol['detail_slice'], nominal_mm=protocol['nominal_mm'],
            loader_sha256=protocol['loader_sha256'], flip_tta=prov['flip_tta'],
            k_train=args['k'], k_select=prov['k_eval'], k_oof=prov['k_eval_extra'],
            window_policy=args['window_policy'], eval_amp=args['eval_amp'],
            training_verification_sha256=protocol['verification_sha256']))
    return dict(status='TRAINING_POLICY_BOUND', d_b_117_screen_sha256=sha(root/C4_SCREEN),
        screen_status=screen['status'], loader=C4_LOADER, folds=folds,
        same_as_oof=['mixed allocation: even K36 positions full 336 px (nominal 140 mm), odd '
                     'positions vol[:, 48:288, 48:288] (nominal 100 mm), each resized to 224 by '
                     'the same center3_windows_from',
                     'K36 within-slot-valid centres, no RNG (eval), -1 centres stay zero',
                     'ImageNet normalisation (same float32 constants)',
                     'coronal/axial horizontal mirror of the constructed views; mean of the '
                     'two post-sigmoid probabilities',
                     'fp16 autocast, float32 sigmoid; checkpoint = fixed-final state that '
                     'produced oof_k36_flip_tta.csv',
                     'raptor_radimagenet/raptor_common/raptor_runtime bytes equal to the '
                     'training run records'],
        differences=['volume source: training/OOF read the cached RAPTOR64 uint8 corpus; '
                     'serving decodes DICOM with raptor_dicom (same contract '
                     'raptor64_336_c140_s02, inherited unchanged from the B172 partner path)',
                     'batching/GPU: OOF batches of bs//2 studies on an RTX 3090; serving one '
                     'study per call on T4 (numerical only)'],
        not_checked=['GPU execution of the five C4 checkpoints on T4',
                     'member-level parity of served C4 probabilities against the OOF tables'])


def c4_window_parity(root=ROOT, seed=20261011):
    """CPU check (torch CPU only): v4 serving C4 inputs equal the EXP179 loader.

    Executes the exact serving overlay text from partner_scores_c4 and the
    training/eval code (scale_windows, normalise, exp124 flip mask/mirror)
    on a synthetic uint8 RAPTOR64 volume with K36 centres, including -1.
    """
    import ast
    import importlib
    import numpy as np
    root = Path(root)
    tree = ast.parse(textwrap.dedent(inspect.getsource(partner_scores_c4)))
    inserted = [ast.literal_eval(node.args[2]) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr == 'replace_once' and len(node.args) == 3 and
                isinstance(node.args[1], ast.Constant) and
                node.args[1].value == 'center3 = center3_windows_from(volume, centers, 224)']
    need(len(inserted) == 1, 'C4 serving overlay anchor differs')
    overlay = textwrap.dedent('\n'.join(line.strip() for line in inserted[0].splitlines()))
    added = [p for p in (str(root/'training/src'),) if p not in sys.path]
    sys.path[:0] = added
    try:
        exp179 = importlib.import_module('raptor_exp179')
        common = importlib.import_module('raptor_common')
        radimagenet = importlib.import_module('raptor_radimagenet')
        runtime = importlib.import_module('raptor_runtime')
        import torch
    finally:
        for p in added:
            sys.path.remove(p)
    # The exact EXP124 eval helpers, without that script's import side effects.
    module = ast.parse((root/'tools/exp124_flip_tta.py').read_text(encoding='utf-8'))
    helpers = [node for node in module.body if isinstance(node, ast.FunctionDef) and
               node.name in ('plane_ranges', 'flip_mask', 'mirror_windows')]
    need(len(helpers) == 3, 'EXP124 flip helpers differ')
    flip = type(sys)('exp124_flip_helpers')
    exec(compile(ast.Module(body=helpers, type_ignores=[]), 'tools/exp124_flip_tta.py', 'exec'),
         flip.__dict__)
    spec = dict(slots=[list(s) for s in common.SLOTS64], depth=64)
    rng = np.random.default_rng(seed)
    volume = rng.integers(0, 256, (64, 336, 336), dtype=np.uint8)
    mask = np.ones(64, np.uint8)
    mask[[5, 40]] = 0
    centers = common.select_centers(mask, 64, 36, spec, 'within-slot-valid', None)
    centers[3] = centers[8] = -1
    ns = dict(volume=volume, centers=centers,
              center3_windows_from=radimagenet.center3_windows_from)
    exec(compile(overlay, '<v4-c4-overlay>', 'exec'), ns)
    served = runtime.normalize_input(ns['c4'], 'imagenet')
    flags = v2.base.plane_flags(centers, spec)
    mirror = served.copy()
    mirror[np.asarray(flags, bool)] = mirror[np.asarray(flags, bool), :, :, ::-1]
    trained = radimagenet.normalise(exp179.scale_windows(volume, centers, 224, 'mixed'), 'imagenet')
    trained_mirror = flip.mirror_windows(torch.from_numpy(trained),
                                         flip.flip_mask(centers, flip.plane_ranges(spec))).numpy()
    result = dict(windows=len(centers), detail_positions=list(range(1, 36, 2)),
                  empty_positions=[3, 8], mirrored=int(sum(flags)),
                  overlay=overlay, plain_equal=bool(np.array_equal(served, trained)),
                  mirror_equal=bool(np.array_equal(mirror, trained_mirror)),
                  flags_equal=flags == flip.flip_mask(centers, flip.plane_ranges(spec)),
                  dtype=str(served.dtype), seed=seed)
    need(result['plain_equal'] and result['mirror_equal'] and result['flags_equal'] and
         served.dtype == trained.dtype == np.float32, 'C4 serving windows differ from EXP179')
    return result


def runtime_estimate(root=ROOT, own_fraction=.5):
    """Hidden-runtime estimate from the recorded EXP176 v1.0.2 timing (planning only).

    EXP176 measured the unchanged v1.0.2 release (blend3 trio + partner) on
    48 train studies on 2xT4 and scaled to 1,300 studies. One OWN pass is a
    single CoAtNet arm through the same serving script: assumed to cost
    ``own_fraction`` of the three-arm trio (visible T4: own 19-24 s vs blend3
    45 s for three studies incl. fixed setup). C4 keeps the EXP170
    ConvNeXt-T graph; its extra crop is one CPU resize per odd window.
    """
    p = read_json(Path(root)/TIMING_REL)['projection']
    trio, partner = p['trio_scaled_seconds'], p['partner_scaled_seconds']
    fixed = p['pilot_fixed_seconds'] + p['setup_aggregation_packaging_residual_seconds']
    scaled_other = p['partner_parity_aggregation_packaging_scaled_seconds']
    unpack = 120.  # S43 293 MB + C4 561 MB hash/extract, generous
    own = own_fraction * trio
    passes = {'A4': [trio, own, own], 'B4': [own, own, own]}
    result = {}
    for candidate, anchors in passes.items():
        scaled = sum(anchors) + partner + scaled_other
        linear = scaled + fixed + unpack
        conservative = 1.5 * scaled + 600 + fixed + unpack
        result[candidate] = dict(anchor_passes=(['EXP155 blend3 trio (3 arms incl. unused old EXP143)', 'S42 OWN', 'S43 OWN']
                                                if candidate == 'A4' else ['S42 OWN', 'EXP174', 'S43 OWN']),
                                 linear_seconds=round(linear, 1), conservative_seconds=round(conservative, 1),
                                 kaggle_linear_margin_seconds=round(KAGGLE_SECONDS - linear, 1),
                                 kaggle_conservative_margin_seconds=round(KAGGLE_SECONDS - conservative, 1),
                                 stress_own_equals_trio_linear_seconds=round(
                                     linear + (len(anchors) - (candidate == 'A4')) * (trio - own), 1))
    return dict(status='PLANNING_ESTIMATE_NOT_A_BOUND', internal_time_limit_seconds=None,
                kaggle_seconds=KAGGLE_SECONDS,
                basis=TIMING_REL, basis_sha256=sha(Path(root)/TIMING_REL),
                assumed_hidden_studies=p['assumed_hidden_studies'], own_fraction_of_trio=own_fraction,
                parent_v102=dict(trio_scaled_seconds=trio, partner_scaled_seconds=partner,
                                 linear_seconds=p['linear_total_seconds'],
                                 conservative_seconds=p['conservative_total_seconds']),
                method='EXP176: linear = scaled stages + fixed pilot/setup; conservative = '
                       '1.5 x scaled + 600 s + fixed; +120 s v4 archive unpacking',
                candidates=result)


def c4_model_meta(ck, entry):
    """The extra crop-policy gate applies only to the five C4 originals."""
    base = v2.base
    if entry['family'] != 'convnext' or entry.get('crop_policy') != 'mixed':
        return base.model_meta(ck, entry)
    need(ck.get('arch') == 'convnext_tiny.fb_in22k_ft_in1k' and ck.get('res') == 224 and
         ck.get('lab') == v2.TARGETS and ck.get('contract') == 'raptor64_336_c140_s02' and
         ck.get('window_policy') == 'within-slot-valid' and ck.get('norm') == 'imagenet' and
         ck.get('tag') == entry['tag'] and isinstance(ck.get('model'), dict) and ck['model'] and
         ck.get('input_policy') in ('center3', 'center3-mixed') and
         ck.get('crop_policy', 'mixed') == 'mixed', 'C4 mixed-crop checkpoint metadata differs')


def partner_scores_c4(work, competition, ids, manifest, deadline):
    """Scoped serving overlay: mixed windows only for ConvNeXt; flip TTA stays."""
    base = v2.base
    loader = inspect.getsource(base._load_models)
    loader = v2.replace_once(loader,
        "need(ck.get('source_sha256') == entry['source_sha256'],",
        "need((entry.get('format') == 'authenticated-original' and sha(path) == entry['source_sha256']) or ck.get('source_sha256') == entry['source_sha256'],")
    scope = {**base.__dict__, 'model_meta': c4_model_meta}
    exec(compile(loader, '<v4-c4-loader>', 'exec'), scope)
    source = inspect.getsource(base.partner_scores)
    source = v2.replace_once(source,
        'center3 = center3_windows_from(volume, centers, 224)',
        "center3 = center3_windows_from(volume, centers, 224)\n                c4 = center3.copy()\n                odd = list(range(1, len(centers), 2))\n                c4[odd] = center3_windows_from(volume[:, 48:288, 48:288], [centers[j] for j in odd], 224)")
    source = v2.replace_once(source,
        "normalize_input(center3, 'imagenet'),",
        "normalize_input(center3, 'imagenet'), normalize_input(c4, 'imagenet'),")
    source = v2.replace_once(source,
        'i, rad, img, triplet, flags = job', 'i, rad, img, mixed_img, triplet, flags = job')
    source = v2.replace_once(source,
        "x = triplet if family == 'dino' else rad if family == 'l2' else img",
        "x = triplet if family == 'dino' else rad if family == 'l2' else mixed_img")
    scope['_load_models'] = scope['_load_models']
    v2.reset_backend()
    exec(compile(source, '<v4-c4-partner>', 'exec'), scope)
    return scope['partner_scores'](work, competition, ids, manifest, deadline)


def pilot_c4(work, competition, out, studies=40):
    """Use the same C4 scorer for the pilot and the full cohort."""
    v2.reset_backend()
    work = Path(work)
    def writer(path, receipt):
        need(0 < receipt['peak_ram_bytes'] < 27*2**30, 'Pilot RAM unavailable/exceeded')
        receipt.update(generator_version=VERSION,
                       manifest_sha256=sha(work/'release-manifest.json'))
        write_json(path, receipt)
    source = inspect.getsource(v2.base.pilot)
    source = v2.replace_once(source, 'started + 3600', "float('inf')")
    scope = {**v2.base.__dict__, 'partner_scores': partner_scores_c4, 'write_json': writer}
    exec(compile(source, '<v4-c4-pilot>', 'exec'), scope)
    return scope['pilot'](work, competition, out, studies)


def overlay_c4(partner, c4_dir, c4_manifest_sha):
    """Authenticate B172, then replace only its five ConvNeXt files in work."""
    partner, c4_dir = Path(partner), Path(c4_dir)
    original = read_json(partner/'release-manifest.json')
    v2.verify_assets(partner, original)
    cm = read_json(c4_dir/'release-manifest.json')
    need(sha(c4_dir/'release-manifest.json') == c4_manifest_sha and
         cm['schema'] == 'exp169-c4-v4' and cm['crop_policy'] == 'mixed' and
         cm['k_eval'] == 36 and len(cm['entries']) == 5,
         'C4 archive manifest differs')
    entries = copy.deepcopy(original['entries'])
    files = copy.deepcopy(original['files'])
    for fold, replacement in enumerate(cm['entries']):
        old = entries[15+fold]
        need(old['family'] == 'convnext' and old['fold'] == fold and
             replacement['family'] == 'convnext' and replacement['fold'] == fold and
             replacement['seed'] == 42 and replacement['path'] == old['path'],
             'C4 fold/path identity differs')
        incoming = c4_dir/replacement['path']
        need(incoming.stat().st_size == replacement['bytes'] and
             sha(incoming) == replacement['sha256'] == replacement['source_sha256'],
             'C4 checkpoint bytes differ')
        (partner/old['path']).unlink()
        shutil.copyfile(incoming, partner/old['path'])
        entries[15+fold] = replacement
        files[old['path']] = dict(bytes=replacement['bytes'], sha256=replacement['sha256'])
    original['entries'], original['files'] = entries, files
    original['c4_manifest_sha256'] = c4_manifest_sha
    # The extracted B172 manifest already exists here; the frozen write_json is
    # create-only ('x'), which made every 4.0.0/4.1.0 run fail (4.1.1 fix).
    _replace_json(partner/'release-manifest.json', original)
    v2.verify_assets(partner, original)
    return original


def s43_ready(root=ROOT):
    from tools import exp152_own0949_release as e152
    average = root/S43_REL/'average-v1'
    receipt_path = average/'average-receipt.json'
    check = average/S43_NAME
    if not receipt_path.exists() or not check.exists():
        return None
    r = read_json(receipt_path)
    protocol = root/S43_REL/'protocol-v1/manifest.json'
    need(r.get('status') == 'VERIFIED_FIXED_SWA_BN' and r.get('checkpoint') == S43_NAME and
         r.get('epochs_1based') == [14, 11, 12] and r.get('run_dir') == S43_RUN and
         r.get('protocol_sha256') == sha(protocol) and
         r.get('checkpoint_sha256') == sha(check) and
         isinstance(r.get('state_schema'), dict) and r['state_schema'],
         'S43 fixed SWA/BN receipt invalid')
    ck = e152.safe_load(check)
    expected = dict(arch=e152.ARCH, res=384, inference_res=320, lab=e152.TARGETS,
                    contract=e152.CONTRACT, k_eval=94, norm='imagenet', window_policy='legacy',
                    oai=False, checkpoint_kind='fixed-swa-bn', swa_epochs_1based=[14, 11, 12], seed=43)
    need(all(ck.get(k) == v for k, v in expected.items()) and
         ck.get('provenance', {}).get('protocol_sha256') == sha(protocol) and
         ck['provenance'].get('run_dir') == S43_RUN and
         e152.state_schema(ck['model']) == r['state_schema'], 'S43 metadata/state/provenance differs')
    return dict(checkpoint=check, checkpoint_sha256=sha(check), receipt_sha256=sha(receipt_path),
                protocol_sha256=sha(protocol), receipt=r)


def bind_s43(ns, manifest, files, work, module):
    """Bind the unchanged EXP152 serving math to the authenticated S43 twin."""
    need(manifest['schema'] == 'exp169-s43-release-v4' and manifest['seed'] == 43 and
         manifest['checkpoint'] == S43_NAME and
         sha(files[S43_NAME]) == manifest['checkpoint_sha256'] and
         sha(files['public-serving.py']) == manifest['serving_sha256'],
         'S43 serving assets differ')
    module.OWN_NAME = S43_NAME
    module.RUN = S43_RUN
    def s43_meta(ck, protocol_sha, run_dir=S43_RUN):
        expected = dict(arch=module.ARCH, res=384, inference_res=320, lab=module.TARGETS,
                        contract=module.CONTRACT, k_eval=94, norm='imagenet',
                        window_policy='legacy', oai=False, checkpoint_kind='fixed-swa-bn',
                        swa_epochs_1based=[14, 11, 12], seed=43)
        need(all(ck.get(k) == v for k, v in expected.items()) and
             ck.get('provenance', {}).get('protocol_sha256') == protocol_sha and
             ck['provenance'].get('run_dir') == run_dir and
             isinstance(ck.get('model'), dict) and ck['model'], 'S43 checkpoint metadata differs')
    module.own_meta = s43_meta
    serving_manifest = dict(protocol_sha256=manifest['protocol_sha256'],
        serving_sha256=manifest['serving_sha256'],
        own_state_schema=manifest['own_state_schema'],
        checkpoints={S43_NAME: {'sha256': manifest['checkpoint_sha256']}})
    module.bind(ns, serving_manifest, {S43_NAME: files[S43_NAME]}, 'own', work)
    return serving_manifest


def authenticated_raw(work, sample, expected_keys, binding=None):
    """Read a completed anchor by exact study ID, roster and file receipt."""
    import numpy as np
    work = Path(work)
    receipt = read_json(work/'receipt.json')
    need(receipt['status'] == 'COMPLETE_INFERENCE' and
         isinstance(receipt.get('seconds'), (int, float)) and receipt['seconds'] > 0,
         'Anchor incomplete: ' + str(work))
    for name in ('raw.npz', 'inputs.json', 'runtime.json', 'binding.json', 'submission.csv'):
        need(sha(work/name) == receipt['files'][name], 'Anchor output drift: ' + name)
    ids, _ = v2.base.validate_csv(work/'submission.csv', sample)
    if binding is not None:
        need(read_json(work/'binding.json') == binding, 'Anchor binding differs')
    with np.load(work/'raw.npz', allow_pickle=False) as raw:
        need(set(raw.files) == {'ids', 'targets', *expected_keys} and
             raw['ids'].tolist() == ids and raw['targets'].tolist() == v2.TARGETS,
             'Anchor raw IDs/targets differ')
        values = {name: _prob(raw[name].copy(), name) for name in expected_keys}
        need(all(a.shape[0] == len(ids) for a in values.values()), 'Anchor cohort differs')
    return ids, values, receipt['seconds']


def _replace_json(path, value):
    """Atomically replace a JSON file owned by the current invocation."""
    path = Path(path)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    write_json(temporary, value)
    os.replace(temporary, path)


def certified_output(output):
    """True only for a PASS receipt whose recorded hash is the root CSV's."""
    output = Path(output)
    try:
        receipt = read_json(output/RECEIPT)
        return (isinstance(receipt, dict) and receipt.get('status') == 'PASS_BUNDLE' and
                receipt.get('submission_sha256') == sha(output/'submission.csv'))
    except (OSError, ValueError):
        return False


def serve(partner, s43_assets, work42, work43, output, competition, candidate,
          *, work_trio=None, work174=None, b174=None, release_manifest=None,
          clock=time.monotonic, started=None):
    """The frozen v1.0.2 fail-loud envelope with v4 raw arms.

    Timing and the pilot projection are recorded, never used to reject.
    Kaggle's 9 h runtime limit applies. Outputs: nothing present at entry is deleted or
    rewritten; the CSV is staged and published by one atomic rename only
    after every check has passed.
    """
    partner, s43_assets, work42, work43, output, competition = map(
        Path, (partner, s43_assets, work42, work43, output, competition))
    need(candidate in ('A4', 'B4'), 'Unknown v4 candidate')
    output.mkdir(parents=True, exist_ok=True)
    final_csv, receipt_path, staged = output/'submission.csv', output/RECEIPT, output/STAGED_CSV
    # partner-raw.npz is written create-only after the partner pass; a stale
    # copy is rejected here, before any GPU time is spent (4.1.1).
    preexisting = {p.name for p in (final_csv, receipt_path, staged, output/'partner-raw.npz')
                   if p.exists()}
    release_sha = None if release_manifest is None else sha(release_manifest)

    try:
        need(not preexisting, 'Stale v4 output: ' + ', '.join(sorted(preexisting)))
        manifest = read_json(partner/'release-manifest.json')
        v2.verify_assets(partner, manifest)
        need(isinstance(manifest.get('c4_manifest_sha256'), str) and
             len(manifest['c4_manifest_sha256']) == 64, 'C4 overlay missing')
        sample = competition/'sample_submission.csv'
        need(b174 is not None, 'Authenticated S42 reference missing')
        b174 = Path(b174)
        bm = read_json(b174 if b174.is_file() else b174/'release-manifest.json')
        need(bm['candidate'] == 'B174' and
             bm['anchors']['EXP143V2']['binding']['own_sha256'] == S42_SHA and
             bm['anchors']['EXP174']['binding']['own_sha256'] != S42_SHA,
             'B174/S42 anchor manifest differs')
        ids, a, s42_seconds = authenticated_raw(work42, sample, {'own'},
            bm['anchors']['EXP143V2']['binding'])
        s42 = a['own']
        if candidate == 'A4':
            need(work_trio is not None, 'A4 public/EXP155 reference missing')
            work_trio = Path(work_trio)
            ids_trio, trio, trio_seconds = authenticated_raw(work_trio, sample,
                {'own', 'public', 'exp143'})
            need(ids == ids_trio, 'A4 trio IDs/order differ')
            others = dict(public=trio['public'], exp155=trio['own'])
        else:
            need(work174 is not None, 'B4 EXP174 reference missing')
            work174 = Path(work174)
            ids174, e, s174_seconds = authenticated_raw(work174, sample, {'own'},
                bm['anchors']['EXP174']['binding'])
            need(ids == ids174, 'B4 EXP174 IDs/order differ')
            others = dict(exp174=e['own'])
        s43m = read_json(s43_assets/'release-manifest.json')
        need(s43m['schema'] == 'exp169-s43-release-v4' and s43m['seed'] == 43 and
             sha(s43_assets/S43_NAME) == s43m['checkpoint_sha256'], 'S43 asset binding differs')
        ids43, s43, s43_seconds = authenticated_raw(work43, sample, {'own'})
        b43 = read_json(work43/'binding.json')
        need(ids43 == ids and b43['variant'] == 'own' and
             b43['own_sha256'] == s43m['checkpoint_sha256'] and
             b43['serving_sha256'] == s43m['serving_sha256'] and
             b43['source_sha256'] == s43m['source_sha256'] and
             b43['public_sha256'] is None, 'S43 raw binding differs')
        anchor_seconds = s42_seconds + s43_seconds + (s174_seconds if candidate == 'B4' else trio_seconds)
        pilot = read_json(partner/'partner-pilot.json')
        need(pilot.get('manifest_sha256') == sha(partner/'release-manifest.json') and
             pilot.get('full_cohort') == len(ids) and
             pilot.get('peak_ram_bytes', 0) > 0, 'V4 pilot provenance differs')
        source = inspect.getsource(v2.base.serve)
        swaps = [
            ("manifest.get('weights') == {'trio': .85, 'bundle': .15}",
             "manifest.get('weights') == partner_weights"),
            ("candidate='TRIO_PLUS_BUNDLE', executed_arms=['EXP155_BLEND3']",
             "candidate=candidate, executed_arms=list(anchor_arms)"),
            ("'VISIBLE_ONLY' if len(ids) < 30 else 'FULL_COHORT'",
             "'VISIBLE_ONLY' if len(ids) < 30 else 'INTENDED_COHORT_PROOF_REQUIRED'"),
            ("weights=manifest['weights']", "weights=dict(v4_weights)"),
            ("trio_seconds = read_json(trio_work / 'receipt.json')['seconds']",
             "trio_seconds = anchor_seconds"),
            ("['L2_RADR50_10', 'DINO_BASE_5', 'EXP170_L2_CONVNEXT_T_5']",
             "list(partner_arms)"),
            ("receipt['candidate'] = 'TRIO_ONLY_FALLBACK'", "receipt['candidate'] = candidate"),
            ("        shutil.copyfile(trio_work / 'submission.csv', output / 'submission.csv')\n        validate_csv(output / 'submission.csv', sample)",
             '        raise'),
            # Keep the frozen pilot's status/cohort/memory gate, but remove
            # its projection and remaining-time conditions.
            ("remaining = LIMIT_SECONDS - trio_seconds - p.get('elapsed_seconds', math.inf) - RESERVE_SECONDS",
             ''),
            ("p.get('projected_full_seconds', math.inf) <= remaining and\n         ", ''),
            ("'Measured partner pilot does not fit the seven-hour/memory gate'",
             "'Partner pilot status/cohort/memory gate failed'"),
            ("deadline = start + remaining", "deadline = float('inf')"),
            ("        need(clock() < deadline, 'Partner/global rank exceeded deadline')\n", ''),
            ("        need(trio_seconds + p['elapsed_seconds'] + clock() - start <= LIMIT_SECONDS - RESERVE_SECONDS,\n             'Full-cohort seven-hour deadline exceeded')\n", ''),
        ]
        for old, new in swaps:
            source = v2.replace_once(source, old, new)
        # 4.1.0: the frozen body writes, removes and hashes only the staged
        # CSV; publication happens below, after every check has passed.
        need(source.count("output / 'submission.csv'") == 5, 'Frozen output anchors changed')
        source = source.replace("output / 'submission.csv'", 'staged_csv')
        def csv_adapter(path, csv_sample):
            if Path(path) == work42/'submission.csv':
                return ids, None
            return v2.base.validate_csv(path, csv_sample)
        def receipt_writer(path, receipt):
            pilot_receipt = receipt.get('pilot') or {}
            receipt.update(generator_version=VERSION, manifest_sha256=release_sha,
                staged_submission=STAGED_CSV,
                timing_policy=dict(projection='diagnostic-only', internal_time_limit_seconds=None,
                    kaggle_seconds=KAGGLE_SECONDS,
                    projected_full_seconds=pilot_receipt.get('projected_full_seconds'),
                    wall_clock_start_supplied=started is not None,
                    wall_clock_seconds=None if started is None else clock() - started),
                c4_manifest_sha256=manifest['c4_manifest_sha256'],
                partner_manifest_sha256=sha(partner/'release-manifest.json'),
                s43_checkpoint_sha256=s43m['checkpoint_sha256'],
                anchors={name: dict(seconds=read_json(w/'receipt.json')['seconds'],
                                    raw_sha256=sha(w/'raw.npz'),
                                    submission_sha256=sha(w/'submission.csv'))
                         for name, w in [('S42', work42), ('S43', work43)] +
                                        ([('EXP174', work174)] if candidate == 'B4' else
                                         [('public_EXP155', work_trio)])},
                backend_reset={'benchmark': False, 'deterministic': True})
            write_json(path, receipt)
        partner_arms = ['L2_RADR50_10', 'EXP172_L2_DINO_BASE_5', 'EXP179_183_L2_CONVNEXT_T_5']
        anchor_arms = (['public', 'OWN_S42_S43', 'EXP155'] if candidate == 'A4' else
                       ['OWN_S42_S43', 'EXP174'])
        scope = {**v2.base.__dict__, 'candidate': candidate, 'anchor_arms': anchor_arms,
            'partner_arms': partner_arms, 'anchor_seconds': anchor_seconds,
            'partner_weights': manifest['weights'],
            'v4_weights': {'trio': .85, 'bundle': .15} if candidate == 'A4' else
                          {'OWN': .35, 'EXP174': .35, 'Q4': .30},
            'validate_csv': csv_adapter, 'aggregate': v2.aggregate,
            'write_json': receipt_writer, 'staged_csv': staged,
            'final_values': lambda _ignored, q: final_values(candidate, s42, s43['own'], q,
                                                             **others)}
        exec(compile(source, '<exp169-v4-serve>', 'exec'), scope)
        result = scope['serve'](partner, work42, output, competition, clock=clock,
                                partner=partner_scores_c4)
        need(result['status'] == 'PASS_BUNDLE' and result['candidate'] == candidate and
             result['executed_arms'] == anchor_arms + partner_arms,
             'V4 candidate incomplete')
        # Every check has passed: publish exactly the certified bytes, atomically.
        written = read_json(receipt_path)
        need(written.get('status') == 'PASS_BUNDLE' and staged.is_file() and
             written.get('submission_sha256') == sha(staged) and
             written.get('generator_version') == VERSION, 'Staged submission/receipt binding differs')
        need(not final_csv.exists(), 'Refuse to replace an existing submission.csv')
        os.replace(staged, final_csv)
        return result
    except BaseException as exc:
        if STAGED_CSV not in preexisting:
            staged.unlink(missing_ok=True)
        # submission.csv is never removed here: it either existed at entry
        # (kept) or was published by this call after every check passed.
        if RECEIPT not in preexisting and not certified_output(output):
            try:
                info = read_json(receipt_path) if receipt_path.exists() else {}
            except (OSError, ValueError):
                info = {}
            info.update(status='FAILED_BUNDLE', candidate=candidate, generator_version=VERSION,
                        manifest_sha256=release_sha, error=type(exc).__name__ + ': ' + str(exc))
            _replace_json(receipt_path, info)
        raise


def _dataset_metadata(dataset, description):
    return dict(title=dataset.split('/')[1], id=dataset, isPrivate=True,
                licenses=[{'name': 'other'}], description=description)


def _source_binding(root):
    try:
        from tools import exp169_release_v3 as v3
    except ImportError:
        import exp169_release_v3 as v3
    from tools.exp169_v102_replay import regenerate
    need(regenerate(root) == (root/v2.BASE_STAGE_REL/'kernel/release.ipynb').read_bytes(),
         'v1.0.2 replay drift')
    a, b = v2.verify_stage(root/A172), v2.verify_stage(root/B172)
    bv = v3.verify_stage(root/B174)
    need(a['candidate'] == 'A172' and b['candidate'] == 'B172' and bv['candidate'] == 'B174' and
         b['archive_sha256'] == v3.PARTNER_ARCHIVE_SHA and
         read_json(root/B174/'code/assets/release-manifest.json')
             ['anchors']['EXP143V2']['binding']['own_sha256'] == S42_SHA and
         sha(root/'artifacts/releases/EXP143_S42v2_average_v101/exp143-v2-e14-e11-e12-bn.pt') == S42_SHA and
         bv['anchor174_archive_sha256'] == sha(root/B174/'anchor174/dataset'/v3.ANCHOR174_ARCHIVE),
         'Reused release stage differs')
    return a, b, bv


def stage(out, root=ROOT):
    root, out = Path(root).resolve(), Path(out).resolve()
    need(root == ROOT.resolve() and out.is_relative_to(root) and not out.exists(),
         'Fresh v4 stage inside project required')
    a, b, bv = _source_binding(root)
    c4 = c4_sources(root)
    c4_evidence = c4_training_evidence(root)
    c4_evidence['serving_window_parity'] = c4_window_parity(root)
    s43 = s43_ready(root)
    out.mkdir(parents=True)
    c4_dir = out/'c4/dataset'
    c4_dir.mkdir(parents=True)
    c4_manifest = dict(schema='exp169-c4-v4', crop_policy='mixed', k_eval=36,
                       tta='plane-aware-coronal-axial-flip', entries=c4)
    c4_meta = out/'c4/release-manifest.json'
    write_json(c4_meta, c4_manifest)
    c4_archive = c4_dir/C4_ARCHIVE
    v2.deterministic_zip(c4_archive, {'release-manifest.json': c4_meta,
        **{e['path']: root/e['source'] for e in c4}})
    write_json(c4_dir/'dataset-metadata.json', _dataset_metadata(C4_DATASET,
        'Five validated EXP179/183 M42 mixed-crop L2 ConvNeXt-T checkpoints; no data or labels.'))
    s43_dir = out/'s43/dataset'
    s43_dir.mkdir(parents=True)
    if s43:
        # This branch is deliberately reached only after receipt and checkpoint
        # authentication. No placeholder is ever represented as a checkpoint.
        s43_assets = out/'s43/assets'
        s43_assets.mkdir()
        shutil.copyfile(s43['checkpoint'], s43_assets/S43_NAME)
        shutil.copyfile(root/'tools/exp152_own0949_release.py', s43_assets/'exp152_own0949_release.py')
        from tools import exp152_own0949_release as e152
        source, _ = e152.public_source()
        e152.write_text(s43_assets/'public-serving.py', e152.serving_source(source))
        s43_manifest = dict(schema='exp169-s43-release-v4', protocol_sha256=s43['protocol_sha256'],
            receipt_sha256=s43['receipt_sha256'], run_dir=S43_RUN, seed=43,
            checkpoint=S43_NAME, checkpoint_sha256=s43['checkpoint_sha256'],
            own_state_schema=s43['receipt']['state_schema'],
            source_sha256=e152.SOURCE_SHA,
            serving_sha256=sha(s43_assets/'public-serving.py'))
        write_json(s43_assets/'release-manifest.json', s43_manifest)
        s43_archive = s43_dir/S43_ARCHIVE
        v2.deterministic_zip(s43_archive, {p.name: p for p in s43_assets.iterdir()})
    else:
        s43_archive = None
    write_json(s43_dir/'dataset-metadata.json', _dataset_metadata(S43_DATASET,
        'Reserved for authenticated EXP182 seed-43 fixed SWA/BN average. No archive until it exists.'))
    code_dir = out/'code/dataset'
    code_dir.mkdir(parents=True)
    code_assets = out/'code/assets'
    code_assets.mkdir()
    shutil.copyfile(__file__, code_assets/'exp169_release_v4.py')
    shutil.copyfile(root/B174/'code/assets/release-manifest.json',
                    code_assets/'b174-release-manifest.json')
    manifest = dict(schema=SCHEMA, version=VERSION, candidates=['A4', 'B4'],
        qualification='PLAN_ONLY_UNQUALIFIED', s43_status='READY' if s43 else 'FAIL_CLOSED_PLACEHOLDER',
        s43=None if not s43 else {k: v for k, v in s43.items() if k != 'receipt' and k != 'checkpoint'},
        c4_manifest_sha256=sha(c4_meta), c4_archive_sha256=sha(c4_archive),
        reused=dict(a172_notebook_sha256=a['notebook_sha256'], b172_archive_sha256=b['archive_sha256'],
                    b174_exp174_archive_sha256=bv['anchor174_archive_sha256'],
                    b174_manifest_sha256=sha(code_assets/'b174-release-manifest.json')),
        tolerance_path=TOL_REL, tolerance_sha256=sha(root/TOL_REL),
        formula={'A4': '.85*r((r(public)+r(OWN)+r(EXP155))/3)+.15*r(Q4)',
                 'B4': '.35*r(OWN)+.35*r(EXP174)+.30*r(Q4)',
                 'OWN': '(P_S42+P_S43)/2', 'Q4': 'r((r(R)+r(C4)+r(D))/3)'},
        arithmetic={'A4_trio': 'A172 blend3 exactly: sum([r(EXP155), r(public), r(OWN)])/3.0, '
                               'float32 as in the parent trio CSV, then ranked',
                    'B4': '(.35*r(OWN) + .35*r(EXP174)) + .30*r(Q4), the B174 evaluation order',
                    'rank': 'average-tie (rank-1)/max(1,N-1) per target, float64'},
        timing_policy='projection diagnostic-only; no internal time limit; Kaggle 9 h',
        output_policy='never delete/rewrite pre-existing outputs; staged CSV published atomically after all checks')
    write_json(code_assets/'release-manifest.json', manifest)
    code_archive = code_dir/CODE_ARCHIVE
    v2.deterministic_zip(code_archive, {p.name: p for p in code_assets.iterdir()})
    write_json(code_dir/'dataset-metadata.json', _dataset_metadata(CODE_DATASET,
        'EXP169 v4 wrapper and frozen recipe; no model weights.'))
    for candidate in ('A4', 'B4'):
        _candidate_stage(out, root, candidate, manifest, s43, c4, c4_archive, code_archive)
    built = {C4_DATASET: c4_archive, S43_DATASET: s43_archive}
    uploaded = {}
    for dataset, pin in UPLOADED.items():
        archive = built[dataset]
        actual = None if archive is None else dict(sha256=sha(archive), bytes=archive.stat().st_size)
        uploaded[dataset] = dict(pin, actual=actual, unchanged=actual is not None and
                                 actual == dict(sha256=pin['sha256'], bytes=pin['bytes']))
    need(uploaded[C4_DATASET]['unchanged'] and (s43 is None or uploaded[S43_DATASET]['unchanged']),
         'Already uploaded C4/S43 archive bytes would change')
    uploads = {CODE_DATASET: code_archive.stat().st_size}
    receipt = dict(status='PLAN_ONLY_NOT_PUSHED' if s43 else 'FAIL_CLOSED_S43_PLACEHOLDER',
                   version=VERSION, uploads_needed=uploads, uploaded_assets=uploaded,
                   c4_provenance=c4_evidence, hidden_runtime_estimate=runtime_estimate(root),
                   s43_pending=None if s43 else [str(root/S43_REL/'average-v1'/S43_NAME),
                       str(root/S43_REL/'average-v1/average-receipt.json')],
                   files={p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()},
                   source_sha256=sha(__file__), uploads=0, submissions=0)
    write_json(out/'build-receipt.json', receipt)
    return receipt


def _runtime_partner_block(candidate, manifest, s43, c4_archive, code_archive):
    """Generate the shared authenticated runtime inside the old error envelope."""
    body = '''
    import importlib.util as _v4_import, shutil as _shutil
    from pathlib import Path
    _competition = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
    def _v4_hash(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    def _v4_unpack(dataset, archive_name, expected_sha, destination, expected_members=None):
        mounts = [Path('/kaggle/input/datasets')/dataset, Path('/kaggle/input')/dataset.split('/')[-1]]
        archive = next((p/archive_name for p in mounts if (p/archive_name).is_file()), None)
        if archive is None: raise FileNotFoundError('Required v4 dataset missing: ' + dataset)
        if _v4_hash(archive) != expected_sha: raise ValueError('v4 archive SHA differs: ' + dataset)
        destination.mkdir(exist_ok=False)
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(names) != len(set(names)) or any(n.startswith('/') or '..' in Path(n).parts for n in names):
                raise ValueError('Unsafe v4 archive members')
            if expected_members is not None and set(names) != set(expected_members):
                raise ValueError('v4 archive member inventory differs')
            z.extractall(destination)
        return destination
    _partner = _v4_unpack('taeyangg4/knee-exp169-b172-v2-assets', 'exp169-assets.zip.bin',
        '__PARTNER_SHA__', Path('/kaggle/working/exp169-partner'))
    if _v4_hash(_partner/'release-manifest.json') != '__PARTNER_MANIFEST_SHA__':
        raise ValueError('B172 manifest SHA differs')
    sys.path.insert(0, str(_partner/'code'))
    import exp169_release_v2 as _e169
    os.environ['KNEE_WORKSPACE_ROOT'] = str(_partner)
    _e169.verify_assets(_partner, _e169.read_json(_partner/'release-manifest.json'))
    _v4code = _v4_unpack('__CODE_DATASET__', '__CODE_ARCHIVE__',
        '__CODE_SHA__', Path('/kaggle/working/exp169-v4-code'),
        {'exp169_release_v4.py', 'b174-release-manifest.json', 'release-manifest.json'})
    if (_v4_hash(_v4code/'release-manifest.json') != '__CODE_MANIFEST_SHA__' or
        _v4_hash(_v4code/'exp169_release_v4.py') != '__SOURCE_SHA__'):
        raise ValueError('v4 code binding differs')
    sys.path.insert(0, str(_v4code))
    import exp169_release_v4 as _e169v4
    _v4_manifest = _e169v4.read_json(_v4code/'release-manifest.json')
    if _v4_manifest['reused']['b174_manifest_sha256'] != _v4_hash(_v4code/'b174-release-manifest.json'):
        raise ValueError('B174 copied manifest SHA differs')
    _c4 = _v4_unpack('__C4_DATASET__', '__C4_ARCHIVE__', '__C4_SHA__',
        Path('/kaggle/working/exp169-c4'),
        {'release-manifest.json', *('models/convnext_s42_f%d.pt' % f for f in range(5))})
    _e169v4.overlay_c4(_partner, _c4, _v4_manifest['c4_manifest_sha256'])
    _s43 = _v4_unpack('__S43_DATASET__', '__S43_ARCHIVE__', '__S43_SHA__',
        Path('/kaggle/working/exp169-s43'),
        {'__S43_NAME__', 'exp152_own0949_release.py', 'public-serving.py', 'release-manifest.json'})
    if _v4_hash(_s43/'release-manifest.json') != '__S43_MANIFEST_SHA__':
        raise ValueError('S43 serving manifest SHA differs')
    _s43_manifest = _e169v4.read_json(_s43/'release-manifest.json')
    if _v4_hash(_s43/'exp152_own0949_release.py') != '__E152_SHA__':
        raise ValueError('S43 serving helper SHA differs')
    _s43_spec = _v4_import.spec_from_file_location('exp169_s43_own_runtime', _s43/'exp152_own0949_release.py')
    _s43_module = _v4_import.module_from_spec(_s43_spec)
    sys.modules[_s43_spec.name] = _s43_module
    _s43_spec.loader.exec_module(_s43_module)
    _work42 = _work
    _work43 = Path('/kaggle/working/exp169-s43-own')
    _work43.mkdir(exist_ok=False)
    exec(compile((_s43/'public-serving.py').read_text(), 's43-public-serving.py', 'exec'), globals())
    _e169v4.bind_s43(globals(), _s43_manifest,
        {'__S43_NAME__': _s43/'__S43_NAME__', 'public-serving.py': _s43/'public-serving.py'},
        _work43, _s43_module)
    main()
    _work = _work42
    import torch as _torch
    _torch.backends.cudnn.benchmark = False
    _torch.backends.cudnn.deterministic = True
    _pilot_dir = Path('/kaggle/working/exp169-v4-pilot')
    _e169v4.pilot_c4(_partner, _competition, _pilot_dir,
        studies=min(40, len(_e169.validate_csv(_work42/'submission.csv',
                                                _competition/'sample_submission.csv')[0])))
    _shutil.copyfile(_pilot_dir/'partner-pilot.json', _partner/'partner-pilot.json')
    _e169v4.serve(_partner, _s43, _work42, _work43, Path('/kaggle/working'),
        _competition, '__CANDIDATE__'__ANCHOR_ARGS__,
        release_manifest=_v4code/'release-manifest.json', started=_exp169_started)
    _r = _e169.read_json(Path('/kaggle/working/exp169-executed-candidate.json'))
    if _r.get('status') != 'PASS_BUNDLE':
        raise RuntimeError('EXP169 v4 did not pass: ' + str(_r.get('stages', {}).get('partner')))
'''
    replacements = {
        '__PARTNER_SHA__': manifest['reused']['b172_archive_sha256'],
        '__PARTNER_MANIFEST_SHA__': '6c47cfc48f74d8422a1909ced52356826ccad7ad2e519fb3c57f4a44a10c61cd',
        '__CODE_DATASET__': CODE_DATASET, '__CODE_ARCHIVE__': CODE_ARCHIVE,
        '__CODE_SHA__': sha(code_archive),
        '__CODE_MANIFEST_SHA__': sha(code_archive.parent.parent/'assets/release-manifest.json'),
        '__SOURCE_SHA__': sha(__file__), '__C4_DATASET__': C4_DATASET,
        '__C4_ARCHIVE__': C4_ARCHIVE, '__C4_SHA__': sha(c4_archive),
        '__S43_DATASET__': S43_DATASET, '__S43_ARCHIVE__': S43_ARCHIVE,
        '__S43_SHA__': sha(code_archive) if s43 is None else sha(code_archive.parent.parent.parent/'s43/dataset'/S43_ARCHIVE),
        '__S43_MANIFEST_SHA__': 'PENDING' if s43 is None else sha(code_archive.parent.parent.parent/'s43/assets/release-manifest.json'),
        '__S43_NAME__': S43_NAME, '__E152_SHA__': sha(ROOT/'tools/exp152_own0949_release.py'),
        '__CANDIDATE__': candidate,
        '__ANCHOR_ARGS__': (", work174=_work174, b174=_v4code/'b174-release-manifest.json'"
                            if candidate == 'B4' else
                            ", work_trio=_work155, b174=_v4code/'b174-release-manifest.json'"),
    }
    for old, new in replacements.items():
        body = body.replace(old, str(new))
    return textwrap.indent(textwrap.dedent(body).lstrip('\n'), '        ')


_ENVELOPE_PRELUDE = '''from pathlib import Path
print('EXP169 __CANDIDATE__ release v__VERSION__', flush=True)
_exp169_root = Path('/kaggle/working')
# v4.1: outputs present before this invocation are never deleted or rewritten.
_exp169_preexisting = {_name for _name in ('submission.csv', 'exp169-executed-candidate.json')
                       if (_exp169_root/_name).exists()}
def _exp169_cleanup():
    """Remove only an uncertified root CSV created by this invocation."""
    import hashlib as _cleanup_hash, json as _cleanup_json
    _csv = _exp169_root/'submission.csv'
    if 'submission.csv' in _exp169_preexisting or not _csv.is_file():
        return
    try:
        _receipt = _cleanup_json.loads((_exp169_root/'exp169-executed-candidate.json').read_text())
        with _csv.open('rb') as _stream:
            _certified = (_receipt.get('status') == 'PASS_BUNDLE' and
                          _receipt.get('submission_sha256') == _cleanup_hash.file_digest(_stream, 'sha256').hexdigest())
    except Exception:
        _certified = False
    if not _certified:
        _csv.unlink()
try:
'''
_INNER_HANDLER_OLD = '''    except BaseException as _bundle_error:
        # B v1.0.2: no trio-only fallback CSV; remove any root CSV so the run fails visibly.
        _fallback = Path('/kaggle/working/submission.csv')
        if _fallback.exists(): _fallback.unlink()
        print('EXP169 FAILED_BUNDLE (no submission written):', type(_bundle_error).__name__, str(_bundle_error), flush=True)
'''
_INNER_HANDLER_NEW = '''    except BaseException as _bundle_error:
        # v4.1: no fallback CSV; remove only an uncertified CSV created by this invocation.
        _exp169_cleanup()
        print('EXP169 FAILED_BUNDLE (no new submission published):', type(_bundle_error).__name__, str(_bundle_error), flush=True)
'''
_FAILURE_HANDLER = '''except BaseException as _failure_error:
    import json as _failure_json, os as _failure_os, time as _failure_time
    _exp169_cleanup()
    _failure_path = _exp169_root/'exp169-executed-candidate.json'
    if ('exp169-executed-candidate.json' in _exp169_preexisting or
            ('submission.csv' not in _exp169_preexisting and (_exp169_root/'submission.csv').is_file())):
        # Never rewrite a pre-existing receipt or the PASS receipt of a kept, certified CSV.
        _failure_path = _exp169_root/('exp169-v4-failure-%d.json' % _failure_time.time_ns())
    try:
        _failure_info = _failure_json.loads(_failure_path.read_text()) if _failure_path.exists() else {}
    except ValueError:
        _failure_info = {}
    _failure_info.update(status='FAILED_BUNDLE', candidate=__CANDIDATE__, generator_version=__VERSION__,
        manifest_sha256=__MANIFEST__, verification_note=__NOTE__,
        error=type(_failure_error).__name__ + ': ' + str(_failure_error))
    _failure_tmp = _failure_path.with_name(_failure_path.name + '.tmp')
    _failure_tmp.write_text(_failure_json.dumps(_failure_info, indent=2, allow_nan=False) + '\\n')
    _failure_os.replace(_failure_tmp, _failure_path)
    raise
'''


def failure_note(manifest_sha, code_archive_sha):
    """Provenance of the v4 artifact itself (no v4 verification note exists yet)."""
    return dict(status='PENDING_INDEPENDENT_VERIFICATION', release=SCHEMA, version=VERSION,
                code_dataset=CODE_DATASET, code_archive_sha256=code_archive_sha,
                code_manifest_sha256=manifest_sha)


def _output_envelope(cell, candidate, manifest_sha, code_archive_sha):
    """Replace the inherited destructive cleanup and parent failure provenance."""
    parent, parent_version = ('A172', '2.0.2') if candidate == 'A4' else ('B174', '3.0.0')
    cell = v2.replace_once(cell, 'from pathlib import Path\ntry:\n',
        _ENVELOPE_PRELUDE.replace('__CANDIDATE__', candidate).replace('__VERSION__', VERSION))
    need(cell.startswith('from pathlib import Path\nprint('), 'Notebook envelope header differs')
    cell = v2.replace_once(cell, _INNER_HANDLER_OLD, _INNER_HANDLER_NEW)
    marker = '\nexcept BaseException as _failure_error:\n'
    need(cell.count(marker) == 1, 'Notebook failure envelope differs')
    head, tail = cell.split(marker)
    need(tail.endswith('    raise\n') and
         "    Path('/kaggle/working/submission.csv').unlink(missing_ok=True)\n" in tail and
         f"candidate={parent!r}" in tail and f"generator_version={parent_version!r}" in tail and
         'verification_note=' in tail and 'manifest_sha256=' in tail,
         'Parent failure envelope differs')
    handler = (_FAILURE_HANDLER.replace('__CANDIDATE__', repr(candidate))
               .replace('__VERSION__', repr(VERSION)).replace('__MANIFEST__', repr(manifest_sha))
               .replace('__NOTE__', repr(failure_note(manifest_sha, code_archive_sha))))
    return head + '\n' + handler


def _candidate_stage(out, root, candidate, manifest, s43, c4, c4_archive, code_archive):
    folder = out/candidate
    folder.mkdir()
    source_stage = root/(A172 if candidate == 'A4' else B174)
    nb = copy.deepcopy(read_json(source_stage/'kernel/release.ipynb'))
    # Preserve the inherited replay and anchors, but remove its alarm from
    # the generated v4 cell only. The frozen parent notebooks are unchanged.
    cell = ''.join(nb['cells'][-1]['source'])
    cell = v2.replace_once(cell,
        'import os, signal as _exp169_signal, time as _exp169_time',
        'import os, time as _exp169_time')
    cell = v2.replace_once(cell,
        "    def _exp169_deadline(_signum, _frame): raise TimeoutError('EXP169 seven-hour deadline guard')\n"
        "    _exp169_signal.signal(_exp169_signal.SIGALRM, _exp169_deadline)\n"
        f"    _exp169_signal.alarm({v2.base.LIMIT_SECONDS - v2.base.RESERVE_SECONDS})\n",
        '')
    cell = v2.replace_once(cell, '    finally:\n        _exp169_signal.alarm(0)\n', '')
    if candidate == 'A4':
        # A172's bundled exp143 is a different checkpoint; obtain the exact
        # EXP143 S42v2 own arm through B172's already uploaded dataset/serving.
        b174_cell = ''.join(read_json(root/B174/'kernel/release.ipynb')['cells'][-1]['source'])
        own_start = b174_cell.index('    # No pip, network, training data or credentials.')
        own_end = b174_cell.index('    _work143 = _work', own_start) + len('    _work143 = _work\n')
        own_block = v2.replace_once(b174_cell[own_start:own_end],
            "_work = Path('/kaggle/working/exp169-trio')",
            "_work = Path('/kaggle/working/exp169-own42')")
        anchor_start = cell.index('    try:\n        # Partner only starts after a completed trio artifact.')
        cell = (cell[:anchor_start] + '    _work155 = _work\n' +
                own_block + cell[anchor_start:])
    start = cell.index('    try:\n        # Partner only starts after a completed trio artifact.')
    stop = cell.index('    except BaseException as _bundle_error:', start)
    runtime = _runtime_partner_block(candidate, manifest, s43, c4_archive, code_archive)
    cell = cell[:start] + '    try:\n' + runtime + cell[stop:]
    cell = _output_envelope(cell, candidate,
                            sha(code_archive.parent.parent/'assets/release-manifest.json'),
                            sha(code_archive))
    if s43 is None:
        cell = ("raise RuntimeError('EXP182 S43 average checkpoint/receipt absent: v4 is a fail-closed placeholder')\n"
                + cell)
    nb['cells'][-1]['source'] = cell
    status = ('S43 (seed-43 fixed SWA/BN, sha ' + s43['checkpoint_sha256'][:12] + '…) is authenticated and '
              'bound. Pilot projection is diagnostic only; no internal time limit; Kaggle 9 h. '
              'No fallback CSV; GPU parity and visible timing checks are still required.'
              if s43 else
              'S43 is a fail-closed dependency. This staged notebook must be replaced after '
              'authenticated S43 integration, parity and timing checks.')
    nb['cells'][0]['source'] = ('# EXP169 ' + candidate + ' v' + VERSION + ' — PLAN ONLY / UNQUALIFIED\n\n' +
                                manifest['formula'][candidate] + '. ' + status)
    compile(cell, '<v4-placeholder-notebook>', 'exec')
    kernel = folder/'kernel'
    kernel.mkdir()
    write_json(kernel/'release.ipynb', nb)
    original_meta = read_json(source_stage/'kernel/kernel-metadata.json')
    datasets = list(original_meta['dataset_sources'])
    if candidate == 'A4':
        datasets = ['taeyangg4/knee-exp169-b172-v2-assets',
                    'taeyangg4/knee-exp169-b172-own-v2-assets',
                    'taeyangg4/knee-exp155-own0949-oai-v1-2-assets',
                    'nartaa/rsna-knee-publication-swa-weights-20261007']
    else:
        # B4 keeps the inherited EXP143v2/EXP174 anchor block verbatim.
        # The old B174 v3 code block is entirely replaced below; its dataset
        # is therefore not needed by the executable notebook.
        datasets = [name for name in datasets if name != 'taeyangg4/knee-exp169-b174-v3-code']
    for name in (C4_DATASET, S43_DATASET, CODE_DATASET):
        if name and name not in datasets:
            datasets.append(name)
    write_json(kernel/'kernel-metadata.json', dict(original_meta,
        id=f'taeyangg4/knee-exp169-{candidate.lower()}-v4',
        title=f'knee-exp169-{candidate.lower()}-v4', dataset_sources=datasets,
        is_private=True, enable_internet=False))
    if candidate == 'B4':
        need('knee-exp169-b174-v3-code' not in cell and
             'taeyangg4/knee-exp169-b174-v3-code' not in datasets and
             "_work174 = _work" in cell,
             'B4 v3 code removed without preserving EXP174 anchor')
    tolerance = read_json(root/TOL_REL)['default']
    write_json(folder/'parity-plan.json', dict(status='FROZEN_NOT_RUN', candidate=candidate,
        anchor={'S42': 'B172 authenticated own reference',
                'S43': 'new independent local own-only reference after average receipt',
                'A4_public_EXP155': 'A172 raw.npz arms' if candidate == 'A4' else None,
                'B4_EXP174': 'B174 authenticated own reference' if candidate == 'B4' else None},
        partner={'R_D': 'B172 original family raw member arrays',
                 'C4': 'new mixed-crop K36 flip-TTA member arrays',
                 'cross_gpu_tolerance_file': TOL_REL,
                 'cross_gpu_tolerance_sha256': sha(root/TOL_REL),
                 'cross_gpu_tolerance': tolerance},
        requirements=['exact inputs, IDs and targets', 'raw anchor probability and CSV parity',
                      'partner pilot/full and cross-GPU parity', 'recompute final by ID from raw arms']))
    estimate = runtime_estimate(root)
    write_json(folder/'timing-plan.json', dict(status='FROZEN_NOT_RUN', candidate=candidate,
        internal_time_limit_seconds=None, kaggle_seconds=KAGGLE_SECONDS,
        included_passes=['S42 OWN', 'S43 OWN', 'EXP155/public trio' if candidate == 'A4' else 'EXP174',
                         'C4/R/D partner pilot', 'C4/R/D full'],
        resource_limits=dict(ram_bytes=27*2**30, per_gpu_bytes=15*2**30),
        projection_policy='Pilot projection is recorded as a diagnostic only; no internal time limit; Kaggle 9 h.',
        hidden_runtime_estimate=dict(estimate, candidates={candidate: estimate['candidates'][candidate]}),
        requirement='Full-cohort elapsed seconds must include both OWN seed passes; intended cohort proof required'))


def verify_stage(path):
    path = Path(path)
    r = read_json(path/'build-receipt.json')
    need(r['version'] == VERSION and r['source_sha256'] == sha(__file__) and
         r['status'] in ('FAIL_CLOSED_S43_PLACEHOLDER', 'PLAN_ONLY_NOT_PUSHED'), 'V4 stage binding differs')
    for name, digest in r['files'].items():
        need(sha(path/name) == digest, 'V4 staged file differs: ' + name)
    need({p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()} ==
         set(r['files']) | {'build-receipt.json'}, 'V4 stage has unlisted/missing files')
    manifest_sha = sha(path/'code/assets/release-manifest.json')
    code_sha = sha(path/'code/dataset'/CODE_ARCHIVE)
    for candidate in ('A4', 'B4'):
        nb = read_json(path/candidate/'kernel/release.ipynb')
        source = ''.join(nb['cells'][-1]['source'])
        compile(source, candidate + '-notebook', 'exec')
        if r['status'] == 'FAIL_CLOSED_S43_PLACEHOLDER':
            need("raise RuntimeError('EXP182 S43 average checkpoint/receipt absent" in source,
                 'Fail-closed S43 gate absent')
        # 4.1.0 envelope: no unconditional deletion, v4 failure provenance,
        # wall-clock start handed to serving.
        failure = source[source.index(FAILURE_MARKER):]
        need(source.count('def _exp169_cleanup():') == 1 and
             source.count('_exp169_cleanup()\n') == 2 and  # inner and outer handlers
             "_fallback.unlink()" not in source and
             "Path('/kaggle/working/submission.csv').unlink" not in source and
             'VERIFICATION-v2.0.2' not in source and 'VERIFICATION-v3.0.0' not in source and
             f'manifest_sha256={manifest_sha!r}' in failure and
             repr(failure_note(manifest_sha, code_sha)) in failure and
             f'candidate={candidate!r}, generator_version={VERSION!r}' in failure and
             "release_manifest=_v4code/'release-manifest.json', started=_exp169_started" in source,
             'V4 notebook output/provenance envelope differs: ' + candidate)
        need('_exp169_signal' not in source and '_exp169_deadline' not in source and
             'SIGALRM' not in source and '.alarm(' not in source and
             '_exp169_started = _exp169_time.monotonic()' in source,
             'V4 notebook alarm removal differs: ' + candidate)
        code = (path/'code/assets/exp169_release_v4.py').read_text(encoding='utf-8')
        need("source = v2.replace_once(source, 'started + 3600', \"float('inf')\")" in code and
             '("deadline = start + remaining", "deadline = float(\'inf\')")' in code,
             'V4 parent deadline neutralisation differs')
        timing = read_json(path/candidate/'timing-plan.json')
        need(timing['internal_time_limit_seconds'] is None and
             timing['kaggle_seconds'] == KAGGLE_SECONDS and
             timing['projection_policy'].startswith('Pilot projection is recorded as a diagnostic only'),
             'V4 timing plan differs')
    if r['status'] == 'PLAN_ONLY_NOT_PUSHED':
        evidence = r['c4_provenance']
        need(all(v['unchanged'] for v in r['uploaded_assets'].values()) and
             set(r['uploads_needed']) == {CODE_DATASET} and
             evidence['status'] == 'TRAINING_POLICY_BOUND' and
             [f['fold'] for f in evidence['folds']] == list(range(5)) and
             all(f['crop_policy'] == 'mixed' and f['k_oof'] == 36 for f in evidence['folds']) and
             evidence['serving_window_parity']['plain_equal'] and
             evidence['serving_window_parity']['mirror_equal'],
             'V4 READY stage receipt differs')
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('stage', 'verify-stage'):
        q = sub.add_parser(command)
        q.add_argument('--out' if command == 'stage' else '--stage', type=Path, required=True)
    args = p.parse_args(argv)
    result = stage(args.out) if args.command == 'stage' else verify_stage(args.stage)
    summary = {k: v for k, v in result.items() if k not in ('files', 'c4_provenance')}
    summary['c4_provenance_status'] = (result.get('c4_provenance') or {}).get('status')
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
