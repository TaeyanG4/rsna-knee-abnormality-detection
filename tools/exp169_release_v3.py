"""EXP169 v3: offline B174 candidate staging and strict two-anchor serving.

B174 = .35 r(EXP143 v2 own raw) + .35 r(EXP174 own raw) + .30 r(Q), where Q is
the unchanged B172 partner bundle (20 members, same uploaded assets). No public
or OAI checkpoint. Rule: artifacts/selection/final-b-rule-v1.md.

The v2 wrapper (tools/exp169_release_v2.py) and its v2.0.2 stages are reused
read-only: the B172 partner archive and the B172 EXP143-v2 own-anchor archive
are bound by hash and NOT rebuilt, so B174 only adds two small datasets (the
EXP174 own anchor and this wrapper). No account operations; stages are plans.

Changelog v3.0.0 (Claude opus for B, 2026-10-10): first B174 release builder:
two EXP152-style own anchors (EXP143 v2 reused, EXP174 new), probability per
anchor -> rank -> fixed .35/.35/.30; stage, verify-stage, inspect-output,
compare-parity and timing-check subcommands.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
import time

try:
    from tools import exp169_release_v2 as v2
except ImportError:  # packaged Kaggle runtime: v2 is imported from the partner code folder
    import exp169_release_v2 as v2

VERSION = '3.0.0'
SCHEMA = 'exp169-b174-release-v3'
CANDIDATE = 'B174'
ROOT = Path(__file__).resolve().parents[1]
base = v2.base
need, sha, read_json, write_json = v2.need, v2.sha, v2.read_json, v2.write_json
TARGETS, UID, rank, aggregate, validate_csv = v2.TARGETS, v2.UID, v2.rank, v2.aggregate, v2.validate_csv

# Fixed recipe (final-b-rule-v1). Insertion order is the addition order.
RECIPE = {'EXP143V2': .35, 'EXP174': .35, 'bundle': .30}
ANCHOR_ARMS = ('EXP143V2', 'EXP174')
PARTNER_ARMS = ['L2_RADR50_10', 'EXP172_L2_DINO_BASE_5', 'EXP170_L2_CONVNEXT_T_5']
ANCHOR_DIRS = {'EXP143V2': 'exp169-trio', 'EXP174': 'exp169-anchor174'}
RULE_REL = 'artifacts/selection/final-b-rule-v1.md'

# Reused B172 v2.0.2 stage (read-only; uploaded datasets).
B172_STAGE_REL = 'artifacts/releases/EXP169_B172_v2_0_2_stage'
B172_NOTEBOOK_SHA = '95e404636389adcdf611bc5361dff3175d30bf15572ed4cb9611b3e6173978cc'
V2_SOURCE_SHA = 'f526eda7a7e5192c51ecdcaa35b1eff9b4c9f598d4e7bf2d2a5a31c58918d3d4'
PARTNER_DATASET = 'taeyangg4/knee-exp169-b172-v2-assets'
PARTNER_ARCHIVE_SHA = 'e22fdb960667a77a265c5ccfe3d82f436c36dd20414b9d000a24365c8cd720d6'
PARTNER_MANIFEST_SHA = '6c47cfc48f74d8422a1909ced52356826ccad7ad2e519fb3c57f4a44a10c61cd'
ANCHOR143_DATASET = 'taeyangg4/knee-exp169-b172-own-v2-assets'
ANCHOR143_ARCHIVE_SHA = 'c9a763e808375782c6b3ce99fa6895e99a7bd9da06c1b29cdbb83388974136c5'
ANCHOR143_MANIFEST_SHA = 'ff985fcd2e86638de39b81bdaea43c9757afe913874d1e29ee7980833d4f6cf1'

# New EXP174 own anchor (authenticated fixed SWA/BN average, copied byte-for-byte).
EXP174_AVERAGE_REL = 'artifacts/experiments/EXP174_OWN0949_JEV/average-v1'
EXP174_RECEIPT_SHA = 'cdd0fc7f7722388ac36ea6d2f10a8d96ebc62a75b0d6b1400994dae9907d6822'
EXP174_CHECKPOINT_SHA = 'df945d0d7843072a47d2fce2fa95f4d82dc0bb480cc049ff2e0bcef280734229'
EXP174_PROTOCOL_REL = 'artifacts/experiments/EXP174_OWN0949_JEV/protocol-v1/manifest.json'
EXP174_PROTOCOL_SHA = '8e3e82f41468274bb72c5853cd487003b729f60f7013a6b9c0697d84df2d0727'
EXP174_OWN_NAME = 'exp174-jev-e14-e11-e12-bn.pt'
EXP174_HELPER = 'exp174_own0949_jev_release.py'
EXP152_FILE = 'exp152_own0949_release.py'
ANCHOR174_ARCHIVE = 'exp174-assets.zip.bin'
ANCHOR174_DATASET = 'taeyangg4/knee-exp169-b174-exp174-v3-assets'
CODE_DATASET = 'taeyangg4/knee-exp169-b174-v3-code'
CODE_ARCHIVE = 'exp169-b174-v3.zip.bin'
KERNEL = 'taeyangg4/knee-exp169-b174-v3'


def expected_arms():
    return [*ANCHOR_ARMS, *PARTNER_ARMS]


def final_values(anchor143, anchor174, bundle):
    """.35 r(EXP143 v2) + .35 r(EXP174) + .30 r(Q); average-tie percentile ranks."""
    import numpy as np
    arrays = [np.asarray(x, np.float64) for x in (anchor143, anchor174, bundle)]
    shape = arrays[0].shape
    need(len(shape) == 2 and shape[0] > 0 and shape[1] == 12 and
         all(a.shape == shape and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
             for a in arrays), 'Invalid B174 anchor/bundle values')
    a143, a174, q = arrays
    final = (RECIPE['EXP143V2'] * rank(a143) + RECIPE['EXP174'] * rank(a174)) + RECIPE['bundle'] * rank(q)
    need(np.isfinite(final).all() and ((final >= 0) & (final <= 1)).all(), 'B174 final blend invalid')
    return final


def anchor_values(work143, work174, sample, manifest, partner_manifest):
    """Authenticated raw own probabilities of both anchors on one ID order.

    Each anchor goes through v2's B own-anchor path: receipt hashes, binding
    equality, CSV IDs/order against ``sample``, raw roster/targets/range.
    """
    anchors = manifest['anchors']
    need(anchors['EXP143V2']['binding'] == partner_manifest['anchor_binding'],
         'EXP143 v2 anchor binding differs from the reused B172 manifest')
    need(anchors['EXP174']['binding'] != anchors['EXP143V2']['binding'] and
         anchors['EXP174']['binding']['own_sha256'] != anchors['EXP143V2']['binding']['own_sha256'],
         'B174 anchors must be two distinct checkpoints')
    ids143, a143 = v2.anchor_values(work143, sample, {'candidate': 'B172',
                                                       'anchor_binding': anchors['EXP143V2']['binding']})
    ids174, a174 = v2.anchor_values(work174, sample, {'candidate': 'B172',
                                                       'anchor_binding': anchors['EXP174']['binding']})
    need(ids143 == ids174, 'B174 anchor rosters/order differ')
    return ids143, a143, a174


def verify_assets(b174, manifest, partner, partner_manifest):
    """Partner (reused B172) assets via v2, then this wrapper's own bindings."""
    b174, partner = Path(b174), Path(partner)
    v2.verify_assets(partner, partner_manifest)
    need(partner_manifest.get('candidate') == 'B172' and
         sha(partner/'release-manifest.json') == manifest['partner']['manifest_sha256'] and
         sha(partner/'code/exp169_release_v2.py') == manifest['partner']['v2_source_sha256'],
         'Reused B172 partner manifest/runtime differs')
    need(manifest.get('schema') == SCHEMA and manifest.get('generator_version') == VERSION and
         manifest.get('candidate') == CANDIDATE and manifest.get('weights') == RECIPE and
         list(manifest.get('weights', {})) == list(RECIPE) and
         manifest.get('anchor_order') == list(ANCHOR_ARMS) and
         manifest.get('expected_arms') == expected_arms(), 'B174 recipe drift')
    anchors = manifest['anchors']
    need(set(anchors) == set(ANCHOR_ARMS) and
         anchors['EXP143V2']['binding'] == partner_manifest['anchor_binding'], 'B174 anchor set differs')
    binding = anchors['EXP174']['binding']
    need(binding.get('variant') == 'own' and binding.get('public_sha256') is None and
         binding.get('own_sha256') == anchors['EXP174']['checkpoint_sha256'] and
         binding.get('own_sha256') != anchors['EXP143V2']['binding']['own_sha256'],
         'EXP174 anchor binding invalid')
    for name, info in manifest['files'].items():
        path = b174/name
        need(path.is_file() and path.stat().st_size == info['bytes'] and sha(path) == info['sha256'],
             'B174 asset differs: ' + name)
    need(sha(b174/'code/exp169_release_v3.py') == sha(__file__), 'B174 wrapper runtime drift')
    return True


def serve(partner, b174, work143, work174, output, competition, *, clock=time.monotonic,
          partner_scores=None):
    """Strict B174 adapter around the frozen serve; no fallback can be published."""
    partner, b174, work143, work174, output, competition = map(
        Path, (partner, b174, work143, work174, output, competition))
    started = False
    output.mkdir(parents=True, exist_ok=True)
    try:
        need(not (output/'exp169-executed-candidate.json').exists(), 'Stale candidate receipt')
        started = True
        need(not (output/'submission.csv').exists(), 'Stale root submission')
        partner_manifest = read_json(partner/'release-manifest.json')
        manifest = read_json(b174/'release-manifest.json')
        verify_assets(b174, manifest, partner, partner_manifest)
        ids, a143, a174 = anchor_values(work143, work174, competition/'sample_submission.csv',
                                        manifest, partner_manifest)
        measured = read_json(partner/'partner-pilot.json')
        need(measured.get('manifest_sha256') == sha(partner/'release-manifest.json') and
             measured.get('full_cohort') == len(ids) and
             0 < measured.get('peak_ram_bytes', 0) < 27*2**30, 'Pilot provenance/cohort/RAM differs')
        seconds = [read_json(w/'receipt.json')['seconds'] for w in (work143, work174)]
        replace_once = v2.replace_once
        source = inspect.getsource(base.serve)
        source = replace_once(source, "manifest.get('weights') == {'trio': .85, 'bundle': .15}",
                              "manifest.get('weights') == partner_weights")
        source = replace_once(source, "candidate='TRIO_PLUS_BUNDLE', executed_arms=['EXP155_BLEND3']",
                              "candidate=candidate, executed_arms=list(anchor_arms)")
        source = replace_once(source, "'VISIBLE_ONLY' if len(ids) < 30 else 'FULL_COHORT'",
                              "'VISIBLE_ONLY' if len(ids) < 30 else 'INTENDED_COHORT_PROOF_REQUIRED'")
        source = replace_once(source, "weights=manifest['weights']", "weights=dict(b174_weights)")
        source = replace_once(source, "trio_seconds = read_json(trio_work / 'receipt.json')['seconds']",
                              "trio_seconds = anchor_seconds")
        source = replace_once(source, "['L2_RADR50_10', 'DINO_BASE_5', 'EXP170_L2_CONVNEXT_T_5']",
                              "list(partner_arms)")
        source = replace_once(source, "receipt['candidate'] = 'TRIO_ONLY_FALLBACK'",
                              "receipt['candidate'] = candidate")
        source = replace_once(source,
            "        shutil.copyfile(trio_work / 'submission.csv', output / 'submission.csv')\n        validate_csv(output / 'submission.csv', sample)",
            '        raise')

        def csv_adapter(path, sample):
            if Path(path) == work143/'submission.csv':
                return ids, (a143, a174)
            return base.validate_csv(path, sample)

        def receipt_writer(path, receipt):
            receipt.update(generator_version=VERSION, verification_note=manifest['verification_note'],
                           manifest_sha256=sha(b174/'release-manifest.json'),
                           partner_manifest_sha256=sha(partner/'release-manifest.json'),
                           anchors={name: dict(binding=manifest['anchors'][name]['binding'],
                                               seconds=s, submission_sha256=sha(w/'submission.csv'),
                                               raw_sha256=sha(w/'raw.npz'))
                                    for name, w, s in zip(ANCHOR_ARMS, (work143, work174), seconds)},
                           backend_reset={'benchmark': False, 'deterministic': True})
            write_json(path, receipt)

        scope = {**base.__dict__, 'candidate': CANDIDATE, 'anchor_arms': ANCHOR_ARMS,
                 'partner_arms': PARTNER_ARMS,
                 # The reused partner manifest is B172's; its own weights are checked, not used.
                 'partner_weights': {'trio': v2.RECIPES['B172']['anchor_weight'],
                                     'bundle': v2.RECIPES['B172']['bundle_weight']},
                 'b174_weights': RECIPE, 'anchor_seconds': sum(seconds),
                 'validate_csv': csv_adapter, 'aggregate': aggregate, 'write_json': receipt_writer,
                 'final_values': lambda anchors, q: final_values(anchors[0], anchors[1], q)}
        exec(compile(source, '<exp169-v3-serve>', 'exec'), scope)
        result = scope['serve'](partner, work143, output, competition, clock=clock,
                                partner=partner_scores or v2.partner_scores)
        need(result['status'] == 'PASS_BUNDLE' and result['candidate'] == CANDIDATE and
             result['executed_arms'] == expected_arms(), 'Returned fallback/incomplete candidate')
        return result
    except BaseException as exc:
        (output/'submission.csv').unlink(missing_ok=True)
        failure = output/'exp169-executed-candidate.json'
        info = {'status': 'FAILED_BUNDLE', 'candidate': CANDIDATE, 'generator_version': VERSION,
                'error': type(exc).__name__ + ': ' + str(exc)}
        if 'manifest' in locals():
            info.update(verification_note=manifest.get('verification_note'),
                        manifest_sha256=sha(b174/'release-manifest.json'))
        if failure.exists() and started:
            existing = read_json(failure)
            existing.update(info)
            temporary = output/'exp169-failed-receipt.json'
            write_json(temporary, existing)
            temporary.replace(failure)
        elif not failure.exists():
            write_json(failure, info)
        raise


# ----------------------------------------------------------------- staging

def source_files():
    return [ROOT/'tools'/name for name in ('exp169_release_v3.py', 'exp169_release_v2.py',
            'exp174_own0949_jev_release.py', EXP152_FILE, 'exp169_v102_replay.py')] + [ROOT/RULE_REL]


verified_note = v2.clone(v2.verified_note, source_files=source_files)


def _unwrap(source):
    """Inverse of v2.candidate_notebook's failure envelope (checked by round trip)."""
    head, marker = 'from pathlib import Path\ntry:\n', '\nexcept BaseException as _failure_error:\n'
    need(source.startswith(head) and source.count(marker) == 1, 'B172 notebook envelope changed')
    body = source[len(head):source.index(marker) + 1]
    lines = body.splitlines()
    need(all(line.startswith('    ') for line in lines), 'B172 notebook body indentation changed')
    inner = ''.join(line[4:] + '\n' for line in lines)
    need(''.join('    ' + line + '\n' for line in inner.splitlines()) == body, 'Envelope round trip differs')
    return inner


def _wrap(inner, manifest_sha, verification_note):
    failure = f'''except BaseException as _failure_error:
    import json as _failure_json
    Path('/kaggle/working/submission.csv').unlink(missing_ok=True)
    _failure_path = Path('/kaggle/working/exp169-executed-candidate.json')
    _failure_info = _failure_json.loads(_failure_path.read_text()) if _failure_path.exists() else {{}}
    _failure_info.update(status='FAILED_BUNDLE', candidate={CANDIDATE!r}, generator_version={VERSION!r},
        manifest_sha256={manifest_sha!r}, verification_note={verification_note!r},
        error=type(_failure_error).__name__ + ': ' + str(_failure_error))
    _failure_path.write_text(_failure_json.dumps(_failure_info, indent=2, allow_nan=False) + '\\n')
    raise
'''
    return 'from pathlib import Path\ntry:\n' + ''.join('    ' + line + '\n' for line in inner.splitlines()) + failure


def b174_notebook(template, anchor174_code, code_archive_sha, manifest_sha, members, wrapper_sha,
                  verification_note=None):
    """B172 v2.0.2 notebook + EXP174 anchor block + B174 wrapper/serve; nothing else changes."""
    template = copy.deepcopy(template)
    need(len(template['cells']) == 3 and
         'Credits and required acknowledgement' in ''.join(template['cells'][1]['source']),
         'B172 notebook cell layout changed')
    inner = _unwrap(''.join(template['cells'][-1]['source']))
    for digest in (PARTNER_ARCHIVE_SHA, PARTNER_MANIFEST_SHA, ANCHOR143_ARCHIVE_SHA, ANCHOR143_MANIFEST_SHA):
        need(inner.count(digest) == 1, 'B172 template binding missing: ' + digest[:12])
    anchor = v2.replace_once(anchor174_code, "_work = Path('/kaggle/working')\n",
        f"_work = Path('/kaggle/working/{ANCHOR_DIRS['EXP174']}')\n_work.mkdir(exist_ok=False)\n")
    end143 = "_e152.bind(globals(), _manifest, _files, 'own', _work)\nmain()\n"
    inner = v2.replace_once(inner, end143,
        end143 + '_work143 = _work\n# B174 second own anchor: EXP174 (EXP152 serving, own-only).\n' +
        anchor + '_work174 = _work\n_work = _work143\n')
    inner = v2.replace_once(inner, "'seconds': _e169.read_json(_work/'receipt.json')['seconds'],",
        "'seconds': _e169.read_json(_work/'receipt.json')['seconds'] + _e169.read_json(_work174/'receipt.json')['seconds'],")
    slug = CODE_DATASET.split('/')[1]
    setup = f'''_b174_mounts = [Path('/kaggle/input/datasets/{CODE_DATASET}'), Path('/kaggle/input/{slug}')]
_b174_archive = next((p/{CODE_ARCHIVE!r} for p in _b174_mounts if (p/{CODE_ARCHIVE!r}).is_file()), None)
if _b174_archive is None: raise FileNotFoundError('Attach the private EXP169 B174 v3 code dataset')
if _hash(_b174_archive) != {code_archive_sha!r}: raise ValueError('EXP169 B174 archive SHA mismatch')
_b174 = Path('/kaggle/working/exp169-b174')
_b174.mkdir(exist_ok=False)
with zipfile.ZipFile(_b174_archive) as _z:
    if sorted(_z.namelist()) != {sorted(members)!r}: raise ValueError('EXP169 B174 archive members differ')
    _z.extractall(_b174)
if _hash(_b174/'release-manifest.json') != {manifest_sha!r}: raise ValueError('EXP169 B174 manifest SHA mismatch')
if _hash(_b174/'code/exp169_release_v3.py') != {wrapper_sha!r}: raise ValueError('EXP169 B174 wrapper SHA mismatch')
sys.path.insert(0, str(_b174/'code'))
import exp169_release_v3 as _e169v3
_e169v3.verify_assets(_b174, _e169v3.read_json(_b174/'release-manifest.json'), _partner, _e169.read_json(_partner/'release-manifest.json'))
'''
    verify_line = "    _e169.verify_assets(_partner, _e169.read_json(_partner/'release-manifest.json'))\n"
    inner = v2.replace_once(inner, verify_line,
                            verify_line + ''.join('    ' + line + '\n' for line in setup.splitlines()))
    inner = v2.replace_once(inner, "_e169.serve(_partner, _work, Path('/kaggle/working'),",
                            "_e169v3.serve(_partner, _b174, _work, _work174, Path('/kaggle/working'),")
    source = _wrap(inner, manifest_sha, verification_note)
    compile(source, '<b174-notebook>', 'exec')
    template['cells'][-1]['source'] = source
    template['cells'][0]['source'] = (f'# EXP169 {CANDIDATE} / generator {VERSION}\n\n'
        'PLAN ONLY / UNQUALIFIED. Fixed membership and weights; K36 partner; fail loudly. '
        '.35 r(EXP143v2 raw) + .35 r(EXP174 raw) + .30 Q; zero public/OAI checkpoints. '
        'Q is the unchanged B172 partner bundle (same 20 members and assets). '
        'Final-B choice follows final-b-rule-v1; this notebook never submits. '
        'Verification, parity and timing remain required.')
    return template


def stage(out, verification_note=None, root=ROOT):
    """Build a fresh B174 stage. Reuses B172 v2.0.2 artifacts by hash; never uploads."""
    from tools import exp174_own0949_jev_release as e174r
    from tools.exp169_v102_replay import regenerate
    begun_wall, begun_cpu = time.monotonic(), time.process_time()
    root, out = Path(root).resolve(), Path(out).resolve()
    need(root == ROOT.resolve() and out.is_relative_to(root) and out != root, 'Stage must be inside project')
    need(not out.exists(), 'Fresh B174 stage required')
    verification = verified_note(verification_note) if verification_note is not None else None
    # 1. Reused B172 stage: authenticated by its own v2 verifier plus pinned digests.
    b172_dir = root/B172_STAGE_REL
    b172 = v2.verify_stage(b172_dir)
    own_build = read_json(b172_dir/'own-anchor/build-receipt.json')
    need(b172['candidate'] == 'B172' and b172['archive_sha256'] == PARTNER_ARCHIVE_SHA and
         b172['manifest_sha256'] == PARTNER_MANIFEST_SHA and b172['notebook_sha256'] == B172_NOTEBOOK_SHA and
         b172['sources']['tools/exp169_release_v2.py'] == V2_SOURCE_SHA and
         own_build['dataset'] == ANCHOR143_DATASET and own_build['archive_sha256'] == ANCHOR143_ARCHIVE_SHA and
         own_build['manifest_sha256'] == ANCHOR143_MANIFEST_SHA, 'Reused B172 stage identity differs')
    partner_manifest = read_json(b172_dir/'assets/release-manifest.json')
    template = read_json(b172_dir/'kernel/release.ipynb')
    b172_meta = read_json(b172_dir/'kernel/kernel-metadata.json')
    need(b172_meta['dataset_sources'] == [PARTNER_DATASET, ANCHOR143_DATASET], 'B172 kernel sources differ')
    regenerated = regenerate(root)
    need(sha(root/v2.BASE_STAGE_REL/'kernel/release.ipynb') == v2.QUALIFIED_SHA and
         regenerated == (root/v2.BASE_STAGE_REL/'kernel/release.ipynb').read_bytes(), 'Qualified notebook drift')
    # 2. EXP174 own anchor (byte-for-byte authenticated average, like the B172 EXP143 v2 anchor).
    average = root/EXP174_AVERAGE_REL
    need(sha(average/'average-receipt.json') == EXP174_RECEIPT_SHA, 'EXP174 average receipt differs')
    receipt = read_json(average/'average-receipt.json')
    shared = e174r._shared
    need(receipt.get('schema') == 'exp174-average-v1' and receipt.get('status') == 'VERIFIED_FIXED_SWA_BN' and
         receipt.get('oai') is False and receipt.get('epochs_1based') == e174r.EPOCHS and
         receipt.get('run_dir') == e174r.RUN and receipt.get('checkpoint') == EXP174_OWN_NAME == e174r.OWN_NAME and
         receipt.get('checkpoint_sha256') == EXP174_CHECKPOINT_SHA and
         receipt.get('protocol_sha256') == EXP174_PROTOCOL_SHA == sha(root/EXP174_PROTOCOL_REL) and
         receipt.get('shared_serving_sha256') == e174r.BASE_SHA256 and
         isinstance(receipt.get('state_schema'), dict) and receipt['state_schema'],
         'EXP174 average identity differs')
    need(sha(average/EXP174_OWN_NAME) == EXP174_CHECKPOINT_SHA, 'EXP174 checkpoint differs')
    out.mkdir(parents=True)
    a_assets, a_dataset = out/'anchor174/assets', out/'anchor174/dataset'
    a_assets.mkdir(parents=True)
    a_dataset.mkdir(parents=True)
    import shutil
    shutil.copyfile(average/EXP174_OWN_NAME, a_assets/EXP174_OWN_NAME)
    shutil.copyfile(root/'tools'/EXP174_HELPER, a_assets/EXP174_HELPER)
    shutil.copyfile(root/'tools'/EXP152_FILE, a_assets/EXP152_FILE)
    source, credits = shared.public_source()
    shared.write_text(a_assets/'public-serving.py', shared.serving_source(source))
    need(sha(a_assets/EXP174_OWN_NAME) == EXP174_CHECKPOINT_SHA and sha(a_assets/EXP152_FILE) == e174r.BASE_SHA256,
         'Copied EXP174 anchor asset differs')
    need(sha(a_assets/'public-serving.py') == partner_manifest['anchor_binding']['serving_sha256'],
         'EXP174 serving source differs from the EXP143 v2 anchor serving')
    anchor_manifest = dict(schema='exp174-release-v1', version=e174r.VERSION,
        protocol_sha256=EXP174_PROTOCOL_SHA, epochs_1based=e174r.EPOCHS,
        average_receipt_sha256=EXP174_RECEIPT_SHA, source_final_sha256=receipt['source_final_sha256'],
        source_average_sha256=EXP174_CHECKPOINT_SHA, public_source_sha256=e174r.SOURCE_SHA,
        serving_sha256=sha(a_assets/'public-serving.py'),
        checkpoints={EXP174_OWN_NAME: dict(sha256=EXP174_CHECKPOINT_SHA, oai=False)},
        own_state_schema=receipt['state_schema'],
        asset_sha256={name: sha(a_assets/name) for name in (EXP174_OWN_NAME, EXP174_HELPER, EXP152_FILE, 'public-serving.py')},
        ranks=dict(own='public numpy double argsort / (N-1)'), policy=e174r.POLICY, tolerance=e174r.TOL,
        dataset=ANCHOR174_DATASET, source_bindings_sha256=shared.jhash(receipt['source_bindings']),
        approval_required={'own': 'submission per the project selection policy'},
        public_reference_role='none; zero public/OAI checkpoints in B174', registered_release_variants=['own'],
        release_role='B174 second own anchor (EXP169 v3); raw.npz own probabilities are ranked by the wrapper',
        qualification='UNQUALIFIED_NOT_PUSHED')
    e174r.write_json(a_assets/'release-manifest.json', anchor_manifest)
    e174r.verify_assets(a_assets, anchor_manifest)
    anchor_archive = a_dataset/ANCHOR174_ARCHIVE
    v2.deterministic_zip(anchor_archive, {p.name: p for p in a_assets.iterdir()})
    write_json(a_dataset/'dataset-metadata.json', dict(title=ANCHOR174_DATASET.split('/')[1], id=ANCHOR174_DATASET,
        isPrivate=True, licenses=[{'name': 'other'}],
        description='Private EXP174 OAI-free own checkpoint and frozen EXP152 serving runtime for EXP169 B174. No public/OAI checkpoint, training data, labels or reports.'))
    anchor_nb, _, _ = e174r.notebook('own', ANCHOR174_DATASET, sha(anchor_archive),
                                     sha(a_assets/'release-manifest.json'), sha(a_assets/EXP174_HELPER), credits)
    binding174 = dict(manifest_sha256=shared.jhash(read_json(a_assets/'release-manifest.json')), variant='own',
                      own_sha256=EXP174_CHECKPOINT_SHA, public_sha256=None, source_sha256=e174r.SOURCE_SHA,
                      serving_sha256=anchor_manifest['serving_sha256'], tolerance=e174r.TOL)
    # 3. B174 wrapper code dataset (small; re-uploaded alone after a verification restage).
    c_assets, c_dataset = out/'code/assets', out/'code/dataset'
    for directory in (c_assets/'code', c_assets/'protocols', c_dataset):
        directory.mkdir(parents=True)
    shutil.copyfile(Path(__file__), c_assets/'code/exp169_release_v3.py')
    shutil.copyfile(root/RULE_REL, c_assets/'protocols/final-b-rule-v1.md')
    if verification is not None:
        shutil.copyfile(verification['path'], c_assets/'protocols/verification.md')
    files = {p.relative_to(c_assets).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(c_assets.rglob('*')) if p.is_file()}
    manifest = dict(schema=SCHEMA, generator_version=VERSION, candidate=CANDIDATE, weights=RECIPE,
        anchor_order=list(ANCHOR_ARMS), expected_arms=expected_arms(), kernel=KERNEL,
        qualification='PLAN_ONLY_UNQUALIFIED' if verification else 'PLAN_ONLY_UNVERIFIED',
        verification_note=verification, rule=dict(path=RULE_REL, sha256=sha(root/RULE_REL)),
        formula='.35 r(EXP143V2 raw own) + .35 r(EXP174 raw own) + .30 r(Q); Q = B172 partner bundle',
        partner=dict(dataset=PARTNER_DATASET, archive_sha256=PARTNER_ARCHIVE_SHA,
                     manifest_sha256=PARTNER_MANIFEST_SHA, v2_source_sha256=V2_SOURCE_SHA,
                     b172_build_receipt_sha256=sha(b172_dir/'build-receipt.json'),
                     notebook_template_sha256=B172_NOTEBOOK_SHA, weights_in_reused_manifest=partner_manifest['weights']),
        anchors={'EXP143V2': dict(dataset=ANCHOR143_DATASET, archive_sha256=ANCHOR143_ARCHIVE_SHA,
                                  manifest_sha256=ANCHOR143_MANIFEST_SHA, work=ANCHOR_DIRS['EXP143V2'],
                                  checkpoint_sha256=partner_manifest['anchor_binding']['own_sha256'],
                                  binding=partner_manifest['anchor_binding']),
                 'EXP174': dict(dataset=ANCHOR174_DATASET, archive_sha256=sha(anchor_archive),
                                manifest_sha256=sha(a_assets/'release-manifest.json'), work=ANCHOR_DIRS['EXP174'],
                                checkpoint_sha256=EXP174_CHECKPOINT_SHA, average_receipt_sha256=EXP174_RECEIPT_SHA,
                                binding=binding174)},
        files=files,
        deferred_checks=['EXP174 checked_model own_meta/state schema/strict load on target T4',
                         'independent release verification bound to the v3 sources',
                         'visible T4 parity: EXP143V2 local ref reusable from B172; NEW local EXP174 own-only ref',
                         'representative T4 timing with two anchors; intended-cohort proof',
                         'final-b-rule-v1 item 2 Gold58 guard (EXP174 >= EXP143 v2 - 0.02)'])
    write_json(c_assets/'release-manifest.json', manifest)
    members = sorted([*files, 'release-manifest.json'])
    code_archive = c_dataset/CODE_ARCHIVE
    v2.deterministic_zip(code_archive, {name: c_assets/name for name in members})
    write_json(c_dataset/'dataset-metadata.json', dict(title=CODE_DATASET.split('/')[1], id=CODE_DATASET,
        isPrivate=True, licenses=[{'name': 'other'}],
        description='Private EXP169 B174 v3 wrapper code and manifest. No model weights, training data, labels or reports.'))
    # 4. Kernel and reused-manifest copy (offline inspection only; never uploaded).
    (out/'reused').mkdir()
    shutil.copyfile(b172_dir/'assets/release-manifest.json', out/'reused/b172-release-manifest.json')
    nb = b174_notebook(template, ''.join(anchor_nb['cells'][-1]['source']), sha(code_archive),
                       sha(c_assets/'release-manifest.json'), members, sha(Path(__file__)), verification)
    metadata = dict(b172_meta, id=KERNEL, title=KERNEL.split('/')[1],
                    dataset_sources=[PARTNER_DATASET, ANCHOR143_DATASET, ANCHOR174_DATASET, CODE_DATASET])
    need(metadata['is_private'] is True and metadata['enable_internet'] is False, 'Kernel must be private/offline')
    (out/'kernel').mkdir()
    write_json(out/'kernel/release.ipynb', nb)
    write_json(out/'kernel/kernel-metadata.json', metadata)
    (out/'regenerated-v1.0.2.ipynb').write_bytes(regenerated)
    plans(out, manifest)
    bound = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    build = dict(status='PLAN_ONLY_NOT_PUSHED' if verification else 'PLAN_ONLY_UNVERIFIED_NOT_PUSHABLE',
        version=VERSION, candidate=CANDIDATE, verification_note=verification, files=bound,
        sources={p.relative_to(root).as_posix(): sha(p) for p in source_files()},
        reused=dict(b172_stage=B172_STAGE_REL, b172_build_receipt_sha256=sha(b172_dir/'build-receipt.json'),
                    partner_archive_sha256=PARTNER_ARCHIVE_SHA, anchor143_archive_sha256=ANCHOR143_ARCHIVE_SHA),
        uploads_needed={ANCHOR174_DATASET: anchor_archive.stat().st_size, CODE_DATASET: code_archive.stat().st_size},
        gpu_hours=0, uploads=0, submissions=0,
        cpu_seconds=time.process_time()-begun_cpu, wall_seconds=time.monotonic()-begun_wall,
        notebook_sha256=sha(out/'kernel/release.ipynb'), manifest_sha256=sha(c_assets/'release-manifest.json'),
        anchor174_archive_sha256=sha(anchor_archive), code_archive_sha256=sha(code_archive))
    write_json(out/'build-receipt.json', build)
    return {k: build[k] for k in ('status', 'candidate', 'notebook_sha256', 'manifest_sha256',
                                  'anchor174_archive_sha256', 'code_archive_sha256', 'uploads_needed',
                                  'cpu_seconds', 'wall_seconds')}


def plans(out, manifest):
    """Frozen checks before target runs; plans are not PASS receipts."""
    out = Path(out)
    common = dict(version=VERSION, candidate=CANDIDATE, status='FROZEN_NOT_RUN',
        verification_note=manifest['verification_note'], manifest_sha256=sha(out/'code/assets/release-manifest.json'),
        notebook_sha256=sha(out/'kernel/release.ipynb'), expected_arms=expected_arms())
    write_json(out/'parity-plan.json', dict(**common, schema='exp169-v3-parity-plan-v1',
        anchor_references={'EXP143V2': 'B172 v2.0.2 local own-anchor reference is reusable unchanged (same archive '
                                        + ANCHOR143_ARCHIVE_SHA[:12] + ', same binding)',
                           'EXP174': 'NEW local EXP174 own-only run of the staged anchor174 archive with the staged binding'},
        anchor_tolerance=dict(max_abs=.002, mean_abs=.0005, inputs='exact', ids='exact', targets=TARGETS),
        partner_tolerance=2e-5, partner_reference='B172 local partner reference reusable (same 20-member archive '
                                                  + PARTNER_ARCHIVE_SHA[:12] + ')',
        cases=['both anchors: receipt, binding, inputs, raw own and CSV by study ID/order/value',
               'anchors share one ID order', 'partner raw per family/member',
               'final CSV recalculated from authenticated raw anchors and partner arrays']))
    write_json(out/'timing-plan.json', dict(**common, schema='exp169-v3-timing-plan-v1',
        guard_seconds=base.LIMIT_SECONDS-base.RESERVE_SECONDS, kaggle_seconds=9*3600,
        ram_limit_bytes=27*2**30, per_gpu_limit_bytes=15*2**30,
        added_vs_b172='one more EXP152 own pass (K94 + mirror, 2xT4); EXP152 planning 1.52 s/study + 180 s, '
                      'conservative 1.5x + 300 s; not measured',
        qualification='Full-size rehearsal and explicit intended-cohort proof remain required; n>=30 is not proof'))
    py = "& '" + sys.executable.replace("'", "''") + "'"
    lines = ['LOCAL ONLY; these commands do not contact Kaggle:',
             "$env:PYTHONPATH = '.venv/Lib/site-packages'",
             f'{py} -m tools.exp169_release_v3 verify-stage --stage "{out}"',
             f'{py} -m tools.exp169_release_v3 inspect-output --stage "{out}" --output "<kernel-output>"',
             f'{py} -m tools.exp169_release_v3 compare-parity --stage "{out}" --output "<kernel-output>" '
             '--anchor143-local "<B172 local own-anchor ref>" --anchor174-local "<NEW EXP174 local ref>" '
             '--partner-local "<local partner-raw.npz>" --out "<fresh-parity-result.json>"',
             f'{py} -m tools.exp169_release_v3 timing-check --stage "{out}" --output "<kernel-output>"', '',
             'Upload plan (NOT executed here): reuse ' + PARTNER_DATASET + ' and ' + ANCHOR143_DATASET + ' as-is;',
             'create ' + ANCHOR174_DATASET + ' from anchor174/dataset and ' + CODE_DATASET + ' from code/dataset;',
             'then push kernel/ privately. Verification, quota check and the selection policy come first.']
    (out/'COMMANDS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ----------------------------------------------------------------- checks

def verify_stage(stage_dir):
    folder = Path(stage_dir)
    receipt = read_json(folder/'build-receipt.json')
    need(receipt['version'] == VERSION and receipt['candidate'] == CANDIDATE and
         receipt['status'] in ('PLAN_ONLY_NOT_PUSHED', 'PLAN_ONLY_UNVERIFIED_NOT_PUSHABLE'), 'Wrong stage receipt')
    for name, digest in receipt['files'].items():
        need(sha(folder/name) == digest, 'Stale staged artifact: ' + name)
    need(receipt['manifest_sha256'] == sha(folder/'code/assets/release-manifest.json') and
         receipt['notebook_sha256'] == sha(folder/'kernel/release.ipynb') and
         receipt['anchor174_archive_sha256'] == sha(folder/'anchor174/dataset'/ANCHOR174_ARCHIVE) and
         receipt['code_archive_sha256'] == sha(folder/'code/dataset'/CODE_ARCHIVE), 'Stage identity/digest binding differs')
    for name, digest in receipt['sources'].items():
        need(sha(ROOT/name) == digest, 'Stale source: ' + name)
    note = receipt['verification_note']
    need(note is None or sha(note['path']) == note['sha256'], 'Verification note drift')
    return receipt


def _stage_manifests(stage_dir):
    folder = Path(stage_dir)
    manifest = read_json(folder/'code/assets/release-manifest.json')
    partner_manifest = read_json(folder/'reused/b172-release-manifest.json')
    need(sha(folder/'reused/b172-release-manifest.json') == manifest['partner']['manifest_sha256'],
         'Reused B172 manifest copy differs')
    return manifest, partner_manifest


def inspect_output(stage_dir, output):
    """Bind a downloaded run to this stage and recompute the final CSV from raw arms."""
    import numpy as np
    build = verify_stage(stage_dir)
    output = Path(output)
    manifest, partner_manifest = _stage_manifests(stage_dir)
    receipt = read_json(output/'exp169-executed-candidate.json')
    need(receipt['status'] == 'PASS_BUNDLE' and receipt['candidate'] == CANDIDATE and
         receipt['generator_version'] == VERSION and receipt['manifest_sha256'] == build['manifest_sha256'] and
         receipt['executed_arms'] == expected_arms() and receipt['weights'] == RECIPE, 'Wrong executed candidate')
    need(sha(output/'submission.csv') == receipt['submission_sha256'] and
         sha(output/'partner-raw.npz') == receipt['partner_raw_sha256'], 'Output hashes differ')
    trio = output/ANCHOR_DIRS['EXP143V2']
    ids, a143, a174 = anchor_values(trio, output/ANCHOR_DIRS['EXP174'], trio/'submission.csv',
                                    manifest, partner_manifest)
    with np.load(output/'partner-raw.npz', allow_pickle=False) as raw:
        need(set(raw.files) == {'ids', 'l2', 'dino', 'convnext'} and raw['ids'].tolist() == ids,
             'Partner roster/keys differ')
        bundle, _ = aggregate({name: raw[name] for name in ('l2', 'dino', 'convnext')})
    expected = final_values(a143, a174, bundle)
    expected = np.asarray([[float(format(float(v), '.10g')) for v in row] for row in expected])
    output_ids, actual = base.validate_csv(output/'submission.csv', trio/'submission.csv')
    need(output_ids == ids and np.array_equal(expected, actual), 'Final blend does not match raw arms')
    stages = receipt['stages']
    need(all(stages[k]['status'] == 'PASS' for k in ('trio', 'pilot', 'partner', 'parity', 'aggregate_output'))
         and all(v <= 2e-5 for v in stages['parity']['max_abs'].values())
         and receipt['elapsed_seconds'] <= base.LIMIT_SECONDS-base.RESERVE_SECONDS
         and max(stages['partner']['gpu_peak_bytes']) < 15*2**30
         and 0 < (stages['partner']['ram']['maxrss_kib'] or 0)*1024 < 27*2**30,
         'Timing/parity/memory gate failed')
    return dict(status='EXECUTION_CHECKED_NOT_QUALIFIED', version=VERSION, candidate=CANDIDATE,
                verification_note=build['verification_note'], seconds=receipt['elapsed_seconds'],
                remaining=['exact Kaggle version binding', 'independent visible parity',
                           'full-size rehearsal and intended cohort proof'])


def _anchor_diff(ref, remote, binding, label):
    import numpy as np
    for folder in (ref, remote):
        receipt = read_json(folder/'receipt.json')
        need(receipt['status'] == 'COMPLETE_INFERENCE', 'Anchor inference incomplete: ' + label)
        for name in ('raw.npz', 'inputs.json', 'runtime.json', 'binding.json', 'submission.csv'):
            need(sha(folder/name) == receipt['files'][name], f'Anchor file hash differs ({label}): {name}')
        need(read_json(folder/'binding.json') == binding, 'Anchor checkpoint/source binding differs: ' + label)
    same_inputs = read_json(ref/'inputs.json') == read_json(remote/'inputs.json')
    with np.load(ref/'raw.npz', allow_pickle=False) as local, np.load(remote/'raw.npz', allow_pickle=False) as kernel:
        need(set(local.files) == set(kernel.files) == {'ids', 'targets', 'own'} and
             np.array_equal(local['ids'], kernel['ids']) and np.array_equal(local['targets'], kernel['targets']) and
             local['targets'].tolist() == TARGETS and 0 < len(local['ids']) < 30,
             'Anchor visible roster/targets/arms differ: ' + label)
        a, b = local['own'], kernel['own']
        n, ids = len(local['ids']), local['ids'].tolist()
        need(a.shape == b.shape == (n, 12) and np.isfinite(a).all() and np.isfinite(b).all() and
             ((a >= 0) & (a <= 1)).all() and ((b >= 0) & (b <= 1)).all(), 'Anchor probabilities invalid: ' + label)
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    raw = dict(max_abs=float(d.max()), mean_abs=float(d.mean()))
    return ids, same_inputs, raw, v2.compare_anchor_csv(ref/'submission.csv', remote/'submission.csv', ids)


def compare_parity(stage_dir, output, partner_local, anchor143_local, anchor174_local, out):
    """Compare a visible run against independent local anchor and partner references."""
    import numpy as np
    stage_dir, output, out = Path(stage_dir), Path(output), Path(out)
    need(not out.exists(), 'Fresh parity result required')
    build = verify_stage(stage_dir)
    plan = read_json(stage_dir/'parity-plan.json')
    need(plan.get('schema') == 'exp169-v3-parity-plan-v1' and plan['version'] == VERSION and
         plan['candidate'] == CANDIDATE and plan['manifest_sha256'] == build['manifest_sha256'] and
         plan['notebook_sha256'] == build['notebook_sha256'] and plan['expected_arms'] == expected_arms(),
         'V3 parity plan binding differs')
    manifest, _ = _stage_manifests(stage_dir)
    anchors, rosters = {}, []
    for name, ref in zip(ANCHOR_ARMS, (anchor143_local, anchor174_local)):
        ids, same, raw, csv_diff = _anchor_diff(Path(ref), output/ANCHOR_DIRS[name],
                                                manifest['anchors'][name]['binding'], name)
        anchors[name] = dict(same_inputs=same, raw=raw, csv=csv_diff, reference_raw_sha256=sha(Path(ref)/'raw.npz'))
        rosters.append(ids)
    need(rosters[0] == rosters[1], 'B174 anchor visible rosters differ')
    n = len(rosters[0])
    with np.load(partner_local, allow_pickle=False) as local, \
         np.load(output/'partner-raw.npz', allow_pickle=False) as remote:
        need(set(local.files) == set(remote.files) == {'ids', 'l2', 'dino', 'convnext'} and
             np.array_equal(local['ids'], remote['ids']) and local['ids'].tolist() == rosters[0],
             'Partner local/kernel roster differs')
        partner_diffs = {}
        for arm in ('l2', 'dino', 'convnext'):
            a, b = local[arm], remote[arm]
            need(a.shape == b.shape and a.shape[0] == (10 if arm == 'l2' else 5) and a.shape[1] == n and
                 a.shape[2] == 12 and np.isfinite(a).all() and np.isfinite(b).all() and
                 ((a >= 0) & (a <= 1)).all() and ((b >= 0) & (b <= 1)).all(), 'Partner probabilities invalid: ' + arm)
            partner_diffs[arm] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    checked = inspect_output(stage_dir, output)
    tol = plan['anchor_tolerance']
    passed = all(a['same_inputs'] and all(d['max_abs'] <= tol['max_abs'] and d['mean_abs'] <= tol['mean_abs']
                                          for d in (a['raw'], a['csv'])) for a in anchors.values()) and \
        all(d <= plan['partner_tolerance'] for d in partner_diffs.values())
    result = dict(status='PASS_VISIBLE_PARITY_UNQUALIFIED' if passed else 'HOLD_PARITY', version=VERSION,
                  candidate=CANDIDATE, studies=n, anchors=anchors, partner_max_abs=partner_diffs,
                  partner_reference_sha256=sha(partner_local), plan_sha256=sha(stage_dir/'parity-plan.json'),
                  output_check=checked['status'], remaining=checked['remaining'])
    write_json(out, result)
    need(passed, 'V3 visible parity failed; inspect result')
    return result


def timing_check(stage_dir, output, require_full=False):
    stage_dir, output = Path(stage_dir), Path(output)
    build = verify_stage(stage_dir)
    plan = read_json(stage_dir/'timing-plan.json')
    need(plan.get('schema') == 'exp169-v3-timing-plan-v1' and plan['version'] == VERSION and
         plan['candidate'] == CANDIDATE and plan['manifest_sha256'] == build['manifest_sha256'] and
         plan['notebook_sha256'] == build['notebook_sha256'], 'V3 timing plan binding differs')
    checked = inspect_output(stage_dir, output)
    receipt = read_json(output/'exp169-executed-candidate.json')
    anchor_seconds = sum(a['seconds'] for a in receipt['anchors'].values())
    need(receipt['schema'] == 'exp169-executed-candidate-v1' and abs(receipt['trio_seconds'] - anchor_seconds) < 1e-6 and
         receipt['pilot']['projected_full_seconds'] <= plan['guard_seconds'] - anchor_seconds -
         receipt['pilot']['elapsed_seconds'] and receipt['elapsed_seconds'] <= plan['guard_seconds'] and
         max(receipt['stages']['partner']['gpu_peak_bytes']) < plan['per_gpu_limit_bytes'] and
         0 < receipt['stages']['partner']['ram']['maxrss_kib']*1024 < plan['ram_limit_bytes'],
         'V3 timing/resource gate failed')
    full_size = receipt['studies'] >= 30 and receipt['qualification'] == 'INTENDED_COHORT_PROOF_REQUIRED'
    need(not require_full or full_size, 'Visible run cannot satisfy full-size timing check')
    return dict(status='PASS_TIMING_RECEIPT_UNQUALIFIED', version=VERSION, candidate=CANDIDATE,
                studies=receipt['studies'], elapsed_seconds=checked['seconds'], anchor_seconds=anchor_seconds,
                full_size=full_size, intended_cohort_proof='PENDING', plan_sha256=sha(stage_dir/'timing-plan.json'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('stage')
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--verification-note', type=Path)
    a = sub.add_parser('verify-stage'); a.add_argument('--stage', type=Path, required=True)
    a = sub.add_parser('inspect-output')
    a.add_argument('--stage', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('compare-parity')
    for name in ('stage', 'output', 'partner-local', 'anchor143-local', 'anchor174-local', 'out'):
        a.add_argument('--' + name, type=Path, required=True)
    a = sub.add_parser('timing-check')
    a.add_argument('--stage', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a.add_argument('--require-full', action='store_true')
    args = p.parse_args(argv)
    if args.command == 'stage':
        result = stage(args.out, args.verification_note)
    elif args.command == 'verify-stage':
        r = verify_stage(args.stage)
        result = dict(status='VERIFIED_PLAN_ONLY', candidate=r['candidate'], version=VERSION,
                      build_status=r['status'], verification_note=r['verification_note'],
                      notebook_sha256=r['notebook_sha256'])
    elif args.command == 'inspect-output':
        result = inspect_output(args.stage, args.output)
    elif args.command == 'compare-parity':
        result = compare_parity(args.stage, args.output, args.partner_local, args.anchor143_local,
                                args.anchor174_local, args.out)
    else:
        result = timing_check(args.stage, args.output, args.require_full)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
