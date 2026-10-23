"""EXP169 v2: offline candidate staging and strict serving around the frozen v1.

No account operations. Stages are plans, never qualification or permission to push.

Changelog v2.0.2 (Claude opus for B, 2026-10-10): compare-parity now compares the
anchor submission CSV (A172 trio values / B172 own-anchor CSV) against the local
reference by study ID, roster order and value under the anchor tolerance; adds
serving_family_rank (probability mean over seeds/folds, then one rank) for the
pre-reading exact serving-policy check. v2.0.1 stages are stale for this version.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
import types
import zipfile

try:
    from tools import exp169_partner_bundle_release as base
except ImportError:  # packaged Kaggle runtime, including a foreign tools package
    import exp169_partner_bundle_release as base

VERSION = '2.0.2'
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_REL = 'artifacts/experiments/EXP169_RELEASE_V2/PROTOCOL.md'
BASE_STAGE_REL = 'artifacts/releases/EXP169_PARTNER_BUNDLE_A1_stage'
TRIO_STAGE_REL = 'artifacts/releases/EXP155_S42v1_1_stage_v12'
EXP172_REL = 'artifacts/experiments/EXP172_DINO_BASE_L2'
BASE_SHA = 'c48aac04128f524116097077a7f1ffd1365f421e3af123c91bb228da09d51665'
BASE_MANIFEST_SHA = 'dea8d3d636884cdc9be5763aa39c81ef470c7d474d80bdcce01e3ed81aa7f433'
BASE_ARCHIVE_SHA = '8a96ac91534144dd68444fc57d308b8c94a4d902fa439aa288992bf25d3dbb95'
QUALIFIED_SHA = 'e96a6c854751efd1684c25553a61ed49a1c70217a814fb0df3c25dddc8d39f75'
RECIPES = {
    'A172': {'anchor': 'EXP155_BLEND3', 'anchor_weight': .85, 'bundle_weight': .15},
    'B172': {'anchor': 'EXP143V2', 'anchor_weight': .70, 'bundle_weight': .30},
}
need, sha, read_json, write_json = base.need, base.sha, base.read_json, base.write_json
TARGETS, UID, _memory = base.TARGETS, base.UID, base._memory
validate_csv = base.validate_csv


def replace_once(source, old, new):
    need(source.count(old) == 1, 'Frozen source anchor changed: ' + old[:90])
    return source.replace(old, new)


def clone(fn, **overrides):
    """Private function globals; never mutate the shared frozen module."""
    return types.FunctionType(fn.__code__, {**fn.__globals__, **overrides},
                              fn.__name__, fn.__defaults__, fn.__closure__)


def rank(values):
    # Same registered average-tie percentile helper as EXP143/EXP152, without
    # requiring SciPy on a packaging-only CPU host. IEEE arithmetic is identical.
    import numpy as np
    try:
        from tools.exp152_own0949_release import percentile_average
    except ImportError:
        from exp152_own0949_release import percentile_average
    a = np.asarray(values, np.float64)
    need(a.ndim == 2 and a.shape[1] == 12 and len(a) > 0 and np.isfinite(a).all(),
         'Invalid family rank input')
    return np.asarray([percentile_average(a[:, j].tolist()) for j in range(12)]).T


aggregate = clone(base.aggregate, rank=rank)


def serving_family_rank(members, family):
    """Serving family aggregation for one family on one study set.

    Mirrors frozen base.aggregate: average PROBABILITIES over the family's
    seeds/folds first (L2 pairs its seed-42 and seed-43 halves, (a[:k]+a[k:])/2,
    before the mean), then take one average-tie rank. Member ranks are never
    averaged. Flip TTA (RadR50, ConvNeXt-T) or plain (DINO) is already inside
    each member's probabilities. For pooled OOF use one fold's held-out tables
    (L2: [S42, S43]); folds are then concatenated and globally reranked.
    """
    import numpy as np
    a = np.asarray(members, np.float64)
    need(family in ('l2', 'dino', 'convnext') and a.ndim == 3 and len(a) > 0 and
         a.shape[1] > 0 and a.shape[2] == 12 and np.isfinite(a).all() and
         ((a >= 0) & (a <= 1)).all() and (family != 'l2' or len(a) % 2 == 0),
         'Invalid serving family members: ' + str(family))
    if family == 'l2':
        half = len(a) // 2
        a = (a[:half] + a[half:]) / 2
    return rank(a.mean(0))


def final_values(anchor, bundle, candidate):
    import numpy as np
    need(candidate in RECIPES, 'Unknown candidate')
    # Reuse complete validation and original A arithmetic; B substitutes only
    # the two registered constants in that same function.
    source = inspect.getsource(base.final_values)
    if candidate == 'B172':
        source = replace_once(source, '.85 * rank(trio) + .15 * rank(bundle)',
                              '.70 * rank(trio) + .30 * rank(bundle)')
    scope = {**base.__dict__, 'rank': rank}
    exec(compile(source, '<exp169-v2-final>', 'exec'), scope)
    return scope['final_values'](np.asarray(anchor), np.asarray(bundle))


def reset_backend(torch_module=None):
    if torch_module is None:
        import torch as torch_module
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True


def _load_models(work, entries, device):
    # EXP172 originals are hash-authenticated training checkpoints. Their model
    # metadata and strict state_dict are checked by the existing loader on T4;
    # they have no export-only source_sha256 field. Other exports are unchanged.
    source = inspect.getsource(base._load_models)
    source = replace_once(source,
        "need(ck.get('source_sha256') == entry['source_sha256'],",
        "need((entry.get('format') == 'authenticated-original' and sha(path) == entry['source_sha256']) or ck.get('source_sha256') == entry['source_sha256'],")
    scope = dict(base.__dict__)
    exec(compile(source, '<exp169-v2-loader>', 'exec'), scope)
    return scope['_load_models'](work, entries, device)


def partner_scores(*args, **kwargs):
    reset_backend()
    return clone(base.partner_scores, _load_models=_load_models)(*args, **kwargs)


def pilot(*args, **kwargs):
    reset_backend()
    work = Path(args[0])
    def writer(path, receipt):
        need(0 < receipt['peak_ram_bytes'] < 27*2**30, 'Pilot RAM measurement unavailable/exceeded')
        receipt.update(generator_version=VERSION, manifest_sha256=sha(work/'release-manifest.json'),
                       verification_note=read_json(work/'release-manifest.json')['verification_note'])
        write_json(path, receipt)
    return clone(base.pilot, partner_scores=partner_scores, write_json=writer)(*args, **kwargs)


def expected_arms(candidate):
    return [RECIPES[candidate]['anchor'], 'L2_RADR50_10',
            'EXP172_L2_DINO_BASE_5', 'EXP170_L2_CONVNEXT_T_5']


def verify_assets(work, manifest):
    base.verify_assets(work, manifest)
    need(manifest.get('generator_version') == VERSION and manifest.get('candidate') in RECIPES,
         'Candidate/version missing')
    recipe = RECIPES[manifest['candidate']]
    need(manifest['weights'] == {'trio': recipe['anchor_weight'], 'bundle': recipe['bundle_weight']}
         and manifest['k_eval'] == 36 and
         manifest['family_weights'] == dict(l2=1/3, dino=1/3, convnext=1/3), 'Recipe drift')
    actual = [(e['family'], e['seed'], e['fold']) for e in manifest['entries']]
    expected = [('l2', s, f) for s in (42, 43) for f in range(5)] + [
        (family, 42, f) for family in ('dino', 'convnext') for f in range(5)]
    need(actual == expected and len({e['source_sha256'] for e in manifest['entries']}) == 20,
         'Partner identity matrix differs')
    for e in manifest['entries']:
        need(manifest['files'][e['path']]['sha256'] == e['sha256'], 'Entry/file binding differs')
        if e['family'] == 'dino':
            need(e['tag'] == f"EXP172_DINO_BASE_L2_S42_F{e['fold']}" and
                 e.get('format') == 'authenticated-original' and e['sha256'] == e['source_sha256'],
                 'Wrong DINO label/fold member')
    need(sha(Path(work)/'code/exp169_partner_bundle_release.py') == BASE_SHA,
         'Frozen runtime drift')
    need(sha(Path(work)/'code/exp169_release_v2.py') == sha(__file__), 'Wrapper runtime drift')
    return True


def anchor_values(work, sample, manifest):
    """Authenticate anchor raw values; B must preserve average ties."""
    import numpy as np
    work = Path(work)
    receipt = read_json(work/'receipt.json')
    need(receipt['status'] == 'COMPLETE_INFERENCE', 'Anchor incomplete')
    for name in ('raw.npz', 'inputs.json', 'runtime.json', 'binding.json', 'submission.csv'):
        need(sha(work/name) == receipt['files'][name], 'Anchor output drift: ' + name)
    ids, csv_values = base.validate_csv(work/'submission.csv', sample)
    binding = read_json(work/'binding.json')
    need(binding == manifest['anchor_binding'], 'Anchor checkpoint/recipe binding differs')
    with np.load(work/'raw.npz', allow_pickle=False) as raw:
        keys = {'ids', 'targets', 'own'} if manifest['candidate'] == 'B172' else {
            'ids', 'targets', 'own', 'public', 'exp143'}
        need(set(raw.files) == keys and raw['ids'].tolist() == ids and
             raw['targets'].tolist() == TARGETS, 'Anchor raw roster/targets differ')
        values = raw['own'].copy() if manifest['candidate'] == 'B172' else csv_values
        for name in keys - {'ids', 'targets'}:
            a = raw[name]
            need(a.shape == (len(ids), 12) and np.isfinite(a).all() and
                 ((a >= 0) & (a <= 1)).all(), 'Anchor raw probabilities invalid')
    return ids, values


def serve(work, trio_work, output, competition, *, clock=time.monotonic, partner=None):
    """Strict adapter: no code path can publish a fallback as a candidate."""
    work, trio_work, output, competition = map(Path, (work, trio_work, output, competition))
    candidate = 'UNKNOWN'
    started = False
    output.mkdir(parents=True, exist_ok=True)
    try:
        need(not (output/'exp169-executed-candidate.json').exists(), 'Stale candidate receipt')
        started = True
        need(not (output/'submission.csv').exists(), 'Stale root submission')
        manifest = read_json(work/'release-manifest.json')
        candidate = manifest['candidate']
        verify_assets(work, manifest)
        ids, values = anchor_values(trio_work, competition/'sample_submission.csv', manifest)
        measured = read_json(work/'partner-pilot.json')
        need(measured.get('manifest_sha256') == sha(work/'release-manifest.json') and
             measured.get('full_cohort') == len(ids) and
             0 < measured.get('peak_ram_bytes', 0) < 27*2**30,
             'Pilot provenance/cohort/RAM differs')
        source = inspect.getsource(base.serve)
        source = replace_once(source, "manifest.get('weights') == {'trio': .85, 'bundle': .15}",
                              "manifest.get('weights') == expected_weights")
        source = replace_once(source, "candidate='TRIO_PLUS_BUNDLE', executed_arms=['EXP155_BLEND3']",
                              "candidate=candidate, executed_arms=[anchor_name]")
        source = replace_once(source, "'VISIBLE_ONLY' if len(ids) < 30 else 'FULL_COHORT'",
                              "'VISIBLE_ONLY' if len(ids) < 30 else 'INTENDED_COHORT_PROOF_REQUIRED'")
        source = replace_once(source, "['L2_RADR50_10', 'DINO_BASE_5', 'EXP170_L2_CONVNEXT_T_5']",
                              "['L2_RADR50_10', 'EXP172_L2_DINO_BASE_5', 'EXP170_L2_CONVNEXT_T_5']")
        source = replace_once(source, "receipt['candidate'] = 'TRIO_ONLY_FALLBACK'",
                              "receipt['candidate'] = candidate")
        source = replace_once(source,
            "        shutil.copyfile(trio_work / 'submission.csv', output / 'submission.csv')\n        validate_csv(output / 'submission.csv', sample)",
            '        raise')

        def csv_adapter(path, sample):
            if Path(path) == trio_work/'submission.csv':
                return ids, values
            return base.validate_csv(path, sample)

        def receipt_writer(path, receipt):
            receipt.update(generator_version=VERSION, verification_note=manifest['verification_note'],
                           protocol_sha256=manifest['engineering_protocol_sha256'],
                           manifest_sha256=sha(work/'release-manifest.json'),
                           anchor=RECIPES[candidate]['anchor'], anchor_binding=manifest['anchor_binding'],
                           backend_reset={'benchmark': False, 'deterministic': True})
            write_json(path, receipt)

        scope = {**base.__dict__, 'candidate': candidate,
                 'anchor_name': RECIPES[candidate]['anchor'], 'expected_weights': manifest['weights'],
                 'validate_csv': csv_adapter, 'aggregate': aggregate,
                 'write_json': receipt_writer,
                 'final_values': lambda a, q: final_values(a, q, candidate)}
        exec(compile(source, '<exp169-v2-serve>', 'exec'), scope)
        result = scope['serve'](work, trio_work, output, competition, clock=clock,
                                partner=partner or partner_scores)
        need(result['status'] == 'PASS_BUNDLE' and result['candidate'] == candidate and
             result['executed_arms'] == expected_arms(candidate), 'Returned fallback/incomplete candidate')
        return result
    except BaseException as exc:
        (output/'submission.csv').unlink(missing_ok=True)
        failure = output/'exp169-executed-candidate.json'
        info = {'status': 'FAILED_BUNDLE', 'candidate': candidate,
                'generator_version': VERSION, 'error': type(exc).__name__ + ': ' + str(exc)}
        if 'manifest' in locals():
            info.update(verification_note=manifest.get('verification_note'),
                        manifest_sha256=sha(work/'release-manifest.json'))
        if failure.exists() and started:
            existing = read_json(failure)
            existing.update(info)
            temporary = output/'exp169-failed-receipt.json'
            write_json(temporary, existing)
            temporary.replace(failure)
        elif not failure.exists():
            write_json(failure, info)
        raise


def exp172_entries(root=ROOT):
    root = Path(root)
    folder = root/EXP172_REL
    collection = read_json(folder/'outputs/collection.json')
    need(collection['status'] == 'VERIFIED' and collection['completed_folds'] == list(range(5))
         and sha(folder/'outputs/manifest.json') == collection['manifest_sha256'], 'EXP172 collection drift')
    training_manifest = read_json(folder/'outputs/manifest.json')
    need(training_manifest['protocol_sha256'] == sha(folder/'PROTOCOL.md') and
         training_manifest['targets'] == TARGETS, 'EXP172 training manifest protocol/targets drift')
    screen = read_json(folder/'screen/result.json')
    pair = screen['new_bundle_minus_d_b_098']
    need(screen['status']['status'] == 'ADMIT' and screen['studies'] == 4343 and
         screen['folds'] == list(range(5)) and screen['protocol_sha256'] == sha(folder/'PROTOCOL.md')
         and pair['primary_mean_delta'] > 0 and pair['primary_mean_ci95'][0] > 0
         and pair['references']['dread']['delta'] >= -.003, 'EXP172 registered gate not admitted')
    entries, proofs = [], {}
    for fold in range(5):
        tag = f'EXP172_DINO_BASE_L2_S42_F{fold}'
        run = folder/'outputs/artifacts/experiments'/tag
        validated_path = folder/f'outputs/VALIDATED_SUCCESS-fold{fold}.json'
        validated, receipt = read_json(validated_path), read_json(run/'receipt.json')
        full = [p for p in validated['phases'] if p['phase'] == 'full']
        need(validated['status'] == 'VALIDATED_SUCCESS' and validated['fold'] == fold and
             validated['manifest_sha256'] == collection['manifest_sha256'] and len(full) == 1 and
             full[0]['receipt_sha256'] == sha(run/'receipt.json') and receipt['status'] == 'PASS'
             and receipt['phase'] == 'full' and receipt['fold'] == fold and
             receipt['protocol_sha256'] == validated['protocol_sha256'] ==
             training_manifest['files'][f'protocols/fold{fold}.json']['sha256'],
             'EXP172 fold receipt drift')
        for name in ('model_best.pt', 'oof_best.csv', 'provenance.json', 'summary.json'):
            info = receipt['outputs'][name]
            need(sha(run/name) == info['sha256'] and (run/name).stat().st_size == info['bytes'],
                 'EXP172 output drift: ' + name)
        need(screen['input_sha256'][f'dino_l2_f{fold}'] == sha(run/'oof_best.csv'),
             'Screened EXP172 fold differs')
        checkpoint = run/'model_best.pt'
        entries.append(dict(family='dino', seed=42, fold=fold, tag=tag,
            path=f'models/dino_s42_f{fold}.pt', source=str(checkpoint.relative_to(root)),
            source_sha256=sha(checkpoint), sha256=sha(checkpoint), bytes=checkpoint.stat().st_size,
            format='authenticated-original', receipt_sha256=sha(run/'receipt.json')))
        proofs[str(validated_path.relative_to(root))] = sha(validated_path)
        proofs[str((run/'receipt.json').relative_to(root))] = sha(run/'receipt.json')
    # Bind every screened prediction by hash, without parsing labels or rows.
    for key, relative in screen['input_files'].items():
        need(sha(root/relative) == screen['input_sha256'][key], 'Screen input drift: ' + key)
    for relative in ('PROTOCOL.md', 'outputs/collection.json', 'outputs/manifest.json', 'screen/result.json'):
        proofs[f'{EXP172_REL}/{relative}'] = sha(folder/relative)
    return entries, proofs


def verified_note(path):
    path = Path(path).resolve()
    note = path.read_text(encoding='utf-8')
    need(bool(note.strip()) and note.rstrip().splitlines()[-1] == 'Verdict: PASS',
         'Independent verification PASS required')
    # A reviewer must bind all the implemented sources and this exact protocol.
    for file in source_files():
        need(sha(file) in note, 'Verification did not bind current file: ' + file.name)
    return dict(path=str(path), sha256=sha(path))


def source_files():
    return [ROOT/'tools'/name for name in ('exp169_release_v2.py',
            'exp169_v102_replay.py', 'exp169_own_anchor.py')] + [ROOT/PROTOCOL_REL]


def deterministic_zip(path, files):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED) as z:
        for name, source in sorted(files.items()):
            need(not name.startswith('/') and '..' not in Path(name).parts, 'Unsafe member')
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            info.external_attr = 0o644 << 16
            with z.open(info, 'w', force_zip64=True) as dst, Path(source).open('rb') as src:
                shutil.copyfileobj(src, dst, 8 << 20)


def candidate_notebook(candidate, dataset, archive_sha, manifest_sha, anchor_notebook=None,
                       verification_note=None):
    from tools.exp169_v102_replay import regenerate
    nb = json.loads(regenerate(ROOT))
    source = ''.join(nb['cells'][-1]['source'])
    source = source.replace(BASE_ARCHIVE_SHA, archive_sha).replace(BASE_MANIFEST_SHA, manifest_sha)
    source = source.replace('taeyangg4/knee-exp169-partner-bundle-assets', dataset)
    source = source.replace("/kaggle/input/knee-exp169-partner-bundle-assets'",
                            f"/kaggle/input/{dataset.split('/')[1]}'")
    source = replace_once(source, 'import exp169_partner_bundle_release as _e169',
                          'import exp169_release_v2 as _e169')
    if candidate == 'B172':
        anchor = ''.join(anchor_notebook['cells'][-1]['source'])
        anchor = replace_once(anchor, "_work = Path('/kaggle/working')",
            "_work = Path('/kaggle/working/exp169-trio')\n_work.mkdir(exist_ok=False)")
        begin = source.index('# No pip, network, training data or credentials.')
        end = source.index('\ntry:\n', begin)
        source = source[:begin] + anchor + source[end:]
        # The frozen acknowledgement applies to both finals; the public
        # checkpoint setup markdown does not describe B's own-only anchor.
        credit = [cell for cell in nb['cells'][1:-1]
                  if cell['cell_type'] == 'markdown' and
                  'Credits and required acknowledgement' in ''.join(cell['source'])]
        need(len(credit) == 1, 'Frozen credit cell changed')
        nb['cells'] = [nb['cells'][0], credit[0], nb['cells'][-1]]
    # Remove any stale root CSV on failure including anchor setup/main, and
    # preserve the original alarm finally block inside this outer envelope.
    failure = f'''except BaseException as _failure_error:
    import json as _failure_json
    Path('/kaggle/working/submission.csv').unlink(missing_ok=True)
    _failure_path = Path('/kaggle/working/exp169-executed-candidate.json')
    _failure_info = _failure_json.loads(_failure_path.read_text()) if _failure_path.exists() else {{}}
    _failure_info.update(status='FAILED_BUNDLE', candidate={candidate!r}, generator_version={VERSION!r},
        manifest_sha256={manifest_sha!r}, verification_note={verification_note!r},
        error=type(_failure_error).__name__ + ': ' + str(_failure_error))
    _failure_path.write_text(_failure_json.dumps(_failure_info, indent=2, allow_nan=False) + '\\n')
    raise
'''
    source = ('from pathlib import Path\ntry:\n' +
              ''.join('    '+line+'\n' for line in source.splitlines()) +
              failure)
    compile(source, '<candidate-notebook>', 'exec')
    nb['cells'][-1]['source'] = source
    nb['cells'][0]['source'] = (f'# EXP169 {candidate} / generator {VERSION}\n\n'
        'PLAN ONLY / UNQUALIFIED. Fixed membership and weights; K36; fail loudly. '
        + ('.85 r(trio) + .15 Q.' if candidate == 'A172' else '.70 r(EXP143v2 raw) + .30 Q; zero public/OAI checkpoints.')
        + ' Q: equal family ranks, probability means within families, EXP172 L2 DINO. '
        'No submission without Taeyang instruction. Verification, parity and timing remain required.')
    return nb


def stage(out, candidate, verification_note, root=ROOT):
    from tools.exp169_v102_replay import regenerate
    from tools.exp169_own_anchor import stage as stage_anchor
    begun_wall, begun_cpu = time.monotonic(), time.process_time()
    root, out = Path(root), Path(out)
    need(root.resolve() == ROOT.resolve() and candidate in RECIPES, 'Invalid root/candidate')
    need(out.resolve().is_relative_to(root.resolve()) and out.resolve() != root.resolve(),
         'Candidate stage must be inside project')
    need(not out.exists(), 'Fresh candidate stage required')
    verification = verified_note(verification_note)
    original = root/BASE_STAGE_REL
    need(sha(original/'kernel/release.ipynb') == QUALIFIED_SHA and
         regenerate(root) == (original/'kernel/release.ipynb').read_bytes(), 'Qualified notebook drift')
    need(sha(original/'assets/release-manifest.json') == BASE_MANIFEST_SHA and
         sha(original/'dataset/exp169-assets.zip.bin') == BASE_ARCHIVE_SHA, 'Qualified asset binding drift')
    manifest = read_json(original/'assets/release-manifest.json')
    base.verify_assets(original/'assets', manifest)
    need(sha(root/'tools/exp169_partner_bundle_release.py') == BASE_SHA, 'Frozen generator drift')
    dino, proofs = exp172_entries(root)
    out.mkdir(parents=True)
    dataset = f'taeyangg4/knee-exp169-{candidate.lower()}-v2-assets'
    kernel_id = f'taeyangg4/knee-exp169-{candidate.lower()}-v2'
    assets = out/'assets'
    assets.mkdir()
    entries = copy.deepcopy(manifest['entries'])
    entries[10:15] = dino
    files = {}
    for name, info in manifest['files'].items():
        source = original/'assets'/name
        if name.startswith('models/dino_'):
            source = root/next(e['source'] for e in dino if e['path'] == name)
        dest = assets/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        files[name] = dest
    for name, source in {'code/exp169_release_v2.py': Path(__file__),
                         'protocols/PROTOCOL.md': root/PROTOCOL_REL,
                         'protocols/verification.md': Path(verification_note)}.items():
        dest = assets/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        files[name] = dest
    trio = root/TRIO_STAGE_REL
    anchor_notebook = None
    if candidate == 'B172':
        anchor_dir = out/'own-anchor'
        stage_anchor(anchor_dir, f'taeyangg4/knee-exp169-b172-own-v2-assets', verification_note, root=root)
        anchor_notebook = read_json(anchor_dir/'own/release.ipynb')
        anchor_meta = read_json(anchor_dir/'own/kernel-metadata.json')
        anchor_manifest = read_json(anchor_dir/'assets/release-manifest.json')
        from tools import exp152_own0949_release as own
        anchor_binding = dict(manifest_sha256=own.jhash(anchor_manifest), variant='own',
            own_sha256=anchor_manifest['checkpoints'][own.OWN_NAME]['sha256'], public_sha256=None,
            source_sha256=own.SOURCE_SHA, serving_sha256=anchor_manifest['serving_sha256'], tolerance=own.TOL)
        metadata = anchor_meta
    else:
        # Verify immutable upstream stage identity; no expensive training state reads.
        build = read_json(trio/'build-receipt.json')
        need(sha(trio/'assets/release-manifest.json') == manifest['trio_manifest_sha256'] == build['manifest_sha256'],
             'Trio manifest drift')
        need(sha(trio/'dataset/exp155-assets.zip.bin') == build['archive_sha256'], 'Trio archive drift')
        anchor_binding = read_json(trio/'blend3-t4-output/binding.json')
        metadata = read_json(trio/'blend3/kernel-metadata.json')
    recipe = RECIPES[candidate]
    manifest.update(generator_version=VERSION, candidate=candidate, dataset=dataset, kernel=kernel_id,
        qualification='PLAN_ONLY_UNQUALIFIED', verification_note=verification,
        engineering_protocol_sha256=sha(root/PROTOCOL_REL), entries=entries,
        anchor_binding=anchor_binding, weights={'trio': recipe['anchor_weight'], 'bundle': recipe['bundle_weight']},
        baseline_notebook_sha256=QUALIFIED_SHA, exp172_proof_files=proofs,
        screen_sha256=sha(root/EXP172_REL/'screen/result.json'),
        screen_protocol_sha256=sha(root/EXP172_REL/'PROTOCOL.md'),
        files={name: dict(sha256=sha(p), bytes=p.stat().st_size) for name, p in files.items()},
        deferred_checks=['EXP172 checkpoint tensor metadata/state schema on target T4',
            'independent release-run astra and fable verification', 'same-model partner parity',
            'representative T4 timing and full-size intended-cohort proof',
            'marginal EXP172 vs R+C gate and missing RadR50 flip OOF reconciliation',
            'B membership freeze before first reading; no B174 included'])
    manifest['source_code']['exp169_release_v2.py'] = sha(__file__)
    if candidate == 'B172':
        manifest.pop('trio_dataset', None)
        manifest.pop('trio_manifest_sha256', None)
        manifest['anchor_dataset'] = metadata['dataset_sources'][0]
    write_json(assets/'release-manifest.json', manifest)
    files['release-manifest.json'] = assets/'release-manifest.json'
    (out/'dataset').mkdir()
    archive = out/'dataset/exp169-assets.zip.bin'
    deterministic_zip(archive, files)
    write_json(out/'dataset/dataset-metadata.json', dict(title=dataset.split('/')[1], id=dataset,
        isPrivate=True, licenses=[{'name': 'other'}], description='Private inference assets. PLAN_ONLY. No training data.'))
    nb = candidate_notebook(candidate, dataset, sha(archive), sha(assets/'release-manifest.json'),
                            anchor_notebook, verification)
    metadata.update(id=kernel_id, title=kernel_id.split('/')[1], code_file='release.ipynb',
                    is_private=True, enable_internet=False,
                    dataset_sources=[dataset, *metadata['dataset_sources']])
    write_json(out/'kernel/release.ipynb', nb)
    write_json(out/'kernel/kernel-metadata.json', metadata)
    (out/'regenerated-v1.0.2.ipynb').write_bytes(regenerate(root))
    plans(out, candidate, manifest)
    bound = {str(p.relative_to(out)).replace('\\', '/'): sha(p)
             for p in out.rglob('*') if p.is_file()}
    receipt = dict(status='PLAN_ONLY_NOT_PUSHED', version=VERSION, candidate=candidate,
        verification_note=verification, files=bound,
        sources={str(p.relative_to(root)).replace('\\', '/'): sha(p) for p in source_files()},
        exp172_proof_files=proofs, gpu_hours=0, uploads=0, submissions=0,
        cpu_seconds=time.process_time()-begun_cpu, wall_seconds=time.monotonic()-begun_wall,
        notebook_sha256=sha(out/'kernel/release.ipynb'), archive_sha256=sha(archive),
        manifest_sha256=sha(assets/'release-manifest.json'))
    write_json(out/'build-receipt.json', receipt)
    return dict(status=receipt['status'], candidate=candidate, notebook_sha256=receipt['notebook_sha256'],
                archive_sha256=receipt['archive_sha256'], manifest_sha256=receipt['manifest_sha256'],
                cpu_seconds=receipt['cpu_seconds'], wall_seconds=receipt['wall_seconds'])


def plans(out, candidate, manifest):
    """Freeze checks before target runs; plans are not PASS receipts."""
    out = Path(out)
    common = dict(version=VERSION, candidate=candidate, status='FROZEN_NOT_RUN',
        verification_note=manifest['verification_note'], manifest_sha256=sha(out/'assets/release-manifest.json'),
        notebook_sha256=sha(out/'kernel/release.ipynb'), expected_arms=expected_arms(candidate))
    reference = ROOT/TRIO_STAGE_REL/'blend3-t4-output'
    write_json(out/'parity-plan.json', dict(**common, schema='exp169-v2-parity-plan-v1',
        anchor_reference=None if candidate == 'B172' else str(reference),
        anchor_reference_files={} if candidate == 'B172' else
            {name: sha(reference/name) for name in
             ('raw.npz', 'inputs.json', 'binding.json', 'submission.csv', 'receipt.json')},
        anchor_reference_arm='own' if candidate == 'B172' else ['own', 'public', 'exp143'],
        anchor_reference_requirement='B172 needs a fresh own-only local run with the staged binding; the old blend3 exp143 arm has a different checkpoint digest' if candidate == 'B172' else 'Frozen blend3 visible output',
        anchor_tolerance=dict(max_abs=.002, mean_abs=.0005, inputs='exact', ids='exact', targets=TARGETS),
        partner_tolerance=2e-5, partner_reference='NEW exact EXP172 L2 five-fold run required; old Dread predictions forbidden',
        cases=['pilot/full on identical IDs', 'raw probabilities per family/member',
               'anchor CSV by study ID, raw-roster order and value against the local reference',
               'global ranks after shard reunion', 'reordered IDs, ties, nonfinite/missing arms',
               'final CSV recalculated from authenticated anchor raw and partner arrays'],
        release_binding='Record actual Kaggle kernel version, dataset versions, downloaded hashes; cannot infer from slug or latest output'))
    write_json(out/'timing-plan.json', dict(**common, schema='exp169-v2-timing-plan-v1',
        hooks=['anchor receipt seconds/RAM/VRAM', 'partner pilot metrics',
               'full partner decode and family inference metrics', 'parity/aggregate stage seconds',
               'strict deadline and output receipt'],
        reuse='tools/exp176_release_timing.py observer/projection; its frozen stage() remains pinned to OLD notebook and MUST NOT be used as a new-candidate qualifier',
        guard_seconds=base.LIMIT_SECONDS-base.RESERVE_SECONDS, kaggle_seconds=9*3600,
        ram_limit_bytes=27*2**30, per_gpu_limit_bytes=15*2**30,
        old_sample_only_hours=dict(linear=2.491465, conservative=3.881635),
        future_budget='A: <=4 h wall / <=8 allocated T4 GPU h planning; B unmeasured, <=A planning only',
        qualification='Full-size rehearsal and explicit intended-cohort hash proof remain required; n>=30 is not proof'))
    py = "& '" + sys.executable.replace("'", "''") + "'"
    anchor_arg = ' --anchor-local "<exact-B172-own-only-local-output>"' if candidate == 'B172' else ''
    commands = [
        'LOCAL ONLY; these commands do not contact Kaggle:',
        "$env:PYTHONPATH = '.venv/Lib/site-packages'",
        f'{py} -m tools.exp169_release_v2 verify-stage --stage "{out}"',
        f'{py} -m tools.exp169_release_v2 inspect-output --stage "{out}" --output "{out / "kernel-output"}"',
        f'{py} -m tools.exp169_release_v2 compare-parity --stage "{out}" --output "{out / "kernel-output"}"{anchor_arg} --partner-local "<exact-local-partner-raw.npz>" --out "<fresh-parity-result.json>"',
        f'{py} -m tools.exp169_release_v2 timing-check --stage "{out}" --output "{out / "kernel-output"}"',
        '', 'Remote push/run/submit commands intentionally absent: separate astra + fable run verification, live quota/lane check,',
        'candidate freeze, and Taeyang submission instruction are still required.',
        'No new timing benchmark was run. Use timing-plan.json to register the next target-T4 run.',
    ]
    (out/'COMMANDS.txt').write_text('\n'.join(commands)+'\n', encoding='utf-8')


def verify_stage(stage_dir):
    folder = Path(stage_dir)
    receipt = read_json(folder/'build-receipt.json')
    need(receipt['version'] == VERSION and receipt['status'] == 'PLAN_ONLY_NOT_PUSHED', 'Wrong stage receipt')
    manifest = read_json(folder/'assets/release-manifest.json')
    need(receipt['candidate'] == manifest['candidate'] and receipt['candidate'] in RECIPES and
         receipt['manifest_sha256'] == sha(folder/'assets/release-manifest.json') and
         receipt['notebook_sha256'] == sha(folder/'kernel/release.ipynb') and
         receipt['archive_sha256'] == sha(folder/'dataset/exp169-assets.zip.bin'),
         'Stage identity/digest binding differs')
    for name, digest in receipt['files'].items():
        need(sha(folder/name) == digest, 'Stale staged artifact: ' + name)
    for name, digest in {**receipt['sources'], **receipt['exp172_proof_files']}.items():
        need(sha(ROOT/name) == digest, 'Stale source: ' + name)
    note = receipt['verification_note']
    need(sha(note['path']) == note['sha256'], 'Verification note drift')
    return receipt


def inspect_output(stage_dir, output):
    """Bind downloaded execution and timing to this stage; no qualification shortcut."""
    import numpy as np
    build = verify_stage(stage_dir)
    output = Path(output)
    receipt = read_json(output/'exp169-executed-candidate.json')
    need(receipt['status'] == 'PASS_BUNDLE' and receipt['candidate'] == build['candidate'] and
         receipt['manifest_sha256'] == build['manifest_sha256'] and
         receipt['executed_arms'] == expected_arms(build['candidate']), 'Wrong executed candidate')
    need(sha(output/'submission.csv') == receipt['submission_sha256'] and
         sha(output/'partner-raw.npz') == receipt['partner_raw_sha256'], 'Output hashes differ')
    manifest = read_json(Path(stage_dir)/'assets/release-manifest.json')
    ids, anchor = anchor_values(output/'exp169-trio', output/'exp169-trio/submission.csv', manifest)
    with np.load(output/'partner-raw.npz', allow_pickle=False) as raw:
        need(set(raw.files) == {'ids','l2','dino','convnext'} and raw['ids'].tolist() == ids,
             'Partner roster/keys differ')
        bundle, _ = aggregate({name: raw[name] for name in ('l2','dino','convnext')})
    expected = final_values(anchor, bundle, build['candidate'])
    expected = np.asarray([[float(format(float(v), '.10g')) for v in row] for row in expected])
    output_ids, actual = base.validate_csv(output/'submission.csv', output/'exp169-trio/submission.csv')
    need(output_ids == ids and np.array_equal(expected, actual), 'Final blend does not match raw arms')
    stages = receipt['stages']
    need(all(stages[k]['status'] == 'PASS' for k in ('trio','pilot','partner','parity','aggregate_output'))
         and all(v <= 2e-5 for v in stages['parity']['max_abs'].values())
         and receipt['elapsed_seconds'] <= base.LIMIT_SECONDS-base.RESERVE_SECONDS
         and max(stages['partner']['gpu_peak_bytes']) < 15*2**30
         and 0 < (stages['partner']['ram']['maxrss_kib'] or 0)*1024 < 27*2**30,
         'Timing/parity/memory gate failed')
    return dict(status='EXECUTION_CHECKED_NOT_QUALIFIED', version=VERSION,
        verification_note=build['verification_note'], candidate=build['candidate'],
        seconds=receipt['elapsed_seconds'], remaining=['exact Kaggle version binding',
        'independent visible parity', 'full-size rehearsal and intended cohort proof'])


def anchor_csv_by_id(path, ids, label):
    """Anchor CSV keyed by study ID: exact header, unique IDs, raw-roster order."""
    row_ids, values = base.validate_csv(path, path)
    need(len(row_ids) == len(ids) and set(row_ids) == set(ids),
         f'Anchor CSV study IDs differ from the raw roster ({label})')
    need(row_ids == list(ids), f'Anchor CSV rows permuted against the raw roster ({label})')
    return {uid: values[k] for k, uid in enumerate(row_ids)}


def compare_anchor_csv(local_csv, remote_csv, ids):
    """Per-study value differences between local and kernel anchor CSVs."""
    import numpy as np
    local = anchor_csv_by_id(local_csv, ids, 'local reference')
    remote = anchor_csv_by_id(remote_csv, ids, 'kernel output')
    d = np.abs(np.stack([local[uid] for uid in ids]) - np.stack([remote[uid] for uid in ids]))
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()))


def compare_parity(stage_dir, output, partner_local, out, anchor_local=None):
    """Compare a v2 visible run against independent local anchor and partner arms."""
    import numpy as np
    stage_dir, output, out = Path(stage_dir), Path(output), Path(out)
    need(not out.exists(), 'Fresh parity result required')
    build = verify_stage(stage_dir)
    plan = read_json(stage_dir/'parity-plan.json')
    candidate = build['candidate']
    need(plan.get('schema') == 'exp169-v2-parity-plan-v1' and
         plan['version'] == VERSION and plan['candidate'] == candidate and
         plan['manifest_sha256'] == build['manifest_sha256'] and
         plan['notebook_sha256'] == build['notebook_sha256'] and
         plan['expected_arms'] == expected_arms(candidate), 'V2 parity plan binding differs')
    manifest = read_json(stage_dir/'assets/release-manifest.json')
    ref = Path(anchor_local) if anchor_local is not None else Path(plan['anchor_reference'] or '')
    need(ref.is_dir() and (candidate != 'B172' or anchor_local is not None),
         'B172 requires an exact own-only local anchor reference')
    if anchor_local is None:
        for name, digest in plan['anchor_reference_files'].items():
            need(sha(ref/name) == digest, 'Frozen anchor reference changed: ' + name)
    # Both references and remote output must be internally authenticated. For B,
    # the binding must be the staged own-only checkpoint, not blend3's exp143.
    for folder in (ref, output/'exp169-trio'):
        receipt = read_json(folder/'receipt.json')
        need(receipt['status'] == 'COMPLETE_INFERENCE', 'Anchor inference incomplete')
        for name in ('raw.npz', 'inputs.json', 'runtime.json', 'binding.json', 'submission.csv'):
            need(sha(folder/name) == receipt['files'][name], 'Anchor file hash differs: '+name)
    need(read_json(ref/'binding.json') == manifest['anchor_binding'] if candidate == 'B172'
         else read_json(ref/'binding.json') == read_json(output/'exp169-trio/binding.json'),
         'Local anchor checkpoint/source binding differs')
    need(read_json(output/'exp169-trio/binding.json') == manifest['anchor_binding'],
         'Kernel anchor binding differs')
    same_inputs = read_json(ref/'inputs.json') == read_json(output/'exp169-trio/inputs.json')
    arms = ['own'] if candidate == 'B172' else ['own', 'public', 'exp143']
    with np.load(ref/'raw.npz', allow_pickle=False) as local, \
         np.load(output/'exp169-trio/raw.npz', allow_pickle=False) as remote:
        need(set(local.files) == set(remote.files) == {'ids', 'targets', *arms} and
             np.array_equal(local['ids'], remote['ids']) and
             np.array_equal(local['targets'], remote['targets']) and
             local['targets'].tolist() == TARGETS and 0 < len(local['ids']) < 30,
             'Anchor visible roster/targets/arms differ')
        n, ids = len(local['ids']), local['ids'].tolist()
        diffs = {}
        for arm in arms:
            a, b = local[arm], remote[arm]
            need(a.shape == b.shape == (n, 12) and np.isfinite(a).all() and
                 np.isfinite(b).all() and ((a >= 0) & (a <= 1)).all() and
                 ((b >= 0) & (b <= 1)).all(), 'Anchor probabilities invalid: '+arm)
            d = np.abs(a.astype(np.float64)-b.astype(np.float64))
            diffs[arm] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()))
    # A172 blends the anchor CSV values (B172: the own-anchor's standalone CSV);
    # raw arms alone cannot detect permuted or misassigned CSV rows.
    anchor_csv = compare_anchor_csv(ref/'submission.csv', output/'exp169-trio/submission.csv', ids)
    with np.load(partner_local, allow_pickle=False) as local, \
         np.load(output/'partner-raw.npz', allow_pickle=False) as remote:
        need(set(local.files) == set(remote.files) == {'ids', 'l2', 'dino', 'convnext'} and
             np.array_equal(local['ids'], remote['ids']), 'Partner local/kernel roster differs')
        partner_diffs = {}
        for arm in ('l2', 'dino', 'convnext'):
            a, b = local[arm], remote[arm]
            need(a.shape == b.shape and a.shape[0] == (10 if arm == 'l2' else 5) and
                 a.shape[2] == 12 and a.shape[1] == n and np.isfinite(a).all() and
                 np.isfinite(b).all() and ((a >= 0) & (a <= 1)).all() and
                 ((b >= 0) & (b <= 1)).all(), 'Partner probabilities invalid: '+arm)
            partner_diffs[arm] = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
    checked = inspect_output(stage_dir, output)
    tol = plan['anchor_tolerance']
    passed = same_inputs and all(d['max_abs'] <= tol['max_abs'] and
                                 d['mean_abs'] <= tol['mean_abs']
                                 for d in [*diffs.values(), anchor_csv]) and \
             all(d <= plan['partner_tolerance'] for d in partner_diffs.values())
    result = dict(status='PASS_VISIBLE_PARITY_UNQUALIFIED' if passed else 'HOLD_PARITY',
        version=VERSION, candidate=candidate, studies=n, same_inputs=same_inputs,
        anchor_raw=diffs, anchor_csv=anchor_csv, partner_max_abs=partner_diffs,
        anchor_reference_sha256=sha(ref/'raw.npz'),
        anchor_reference_csv_sha256=sha(ref/'submission.csv'),
        partner_reference_sha256=sha(partner_local), plan_sha256=sha(stage_dir/'parity-plan.json'),
        output_check=checked['status'], remaining=checked['remaining'])
    write_json(out, result)
    need(passed, 'V2 visible parity failed; inspect result')
    return result


def timing_check(stage_dir, output, require_full=False):
    """Apply v2 receipt bounds; an intended-cohort proof remains separate."""
    stage_dir, output = Path(stage_dir), Path(output)
    build = verify_stage(stage_dir)
    plan = read_json(stage_dir/'timing-plan.json')
    need(plan.get('schema') == 'exp169-v2-timing-plan-v1' and
         plan['version'] == VERSION and plan['candidate'] == build['candidate'] and
         plan['manifest_sha256'] == build['manifest_sha256'] and
         plan['notebook_sha256'] == build['notebook_sha256'] and
         plan['expected_arms'] == expected_arms(build['candidate']), 'V2 timing plan binding differs')
    checked = inspect_output(stage_dir, output)
    receipt = read_json(output/'exp169-executed-candidate.json')
    need(receipt['generator_version'] == VERSION and receipt['schema'] == 'exp169-executed-candidate-v1' and
         receipt['pilot']['projected_full_seconds'] <= plan['guard_seconds'] -
         receipt['trio_seconds'] - receipt['pilot']['elapsed_seconds'] and
         receipt['elapsed_seconds'] <= plan['guard_seconds'] and
         max(receipt['stages']['partner']['gpu_peak_bytes']) < plan['per_gpu_limit_bytes'] and
         0 < receipt['stages']['partner']['ram']['maxrss_kib']*1024 < plan['ram_limit_bytes'],
         'V2 timing/resource gate failed')
    full_size = receipt['studies'] >= 30 and receipt['qualification'] == 'INTENDED_COHORT_PROOF_REQUIRED'
    need(not require_full or full_size, 'Visible run cannot satisfy full-size timing check')
    return dict(status='PASS_TIMING_RECEIPT_UNQUALIFIED', version=VERSION,
        candidate=build['candidate'], studies=receipt['studies'],
        elapsed_seconds=checked['seconds'], full_size=full_size,
        intended_cohort_proof='PENDING', plan_sha256=sha(stage_dir/'timing-plan.json'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('stage')
    a.add_argument('--candidate', choices=RECIPES, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--verification-note', type=Path, required=True)
    a = sub.add_parser('replay')
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--verification-note', type=Path, required=True)
    a = sub.add_parser('verify-stage'); a.add_argument('--stage', type=Path, required=True)
    a = sub.add_parser('inspect-output')
    a.add_argument('--stage', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('compare-parity')
    a.add_argument('--stage', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a.add_argument('--partner-local', type=Path, required=True); a.add_argument('--anchor-local', type=Path)
    a.add_argument('--out', type=Path, required=True)
    a = sub.add_parser('timing-check')
    a.add_argument('--stage', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a.add_argument('--require-full', action='store_true')
    args = p.parse_args(argv)
    if args.command == 'stage':
        result = stage(args.out, args.candidate, args.verification_note)
    elif args.command == 'replay':
        from tools.exp169_v102_replay import regenerate
        note = verified_note(args.verification_note)
        need(not args.out.exists(), 'Fresh replay output required')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(regenerate(ROOT))
        result = dict(status='REGENERATED_BYTE_IDENTICAL', version=VERSION,
                      verification_note=note, sha256=sha(args.out))
    elif args.command == 'verify-stage':
        r = verify_stage(args.stage)
        result = dict(status='VERIFIED_PLAN_ONLY', candidate=r['candidate'], version=VERSION,
                      verification_note=r['verification_note'], notebook_sha256=r['notebook_sha256'])
    elif args.command == 'inspect-output':
        result = inspect_output(args.stage, args.output)
    elif args.command == 'compare-parity':
        result = compare_parity(args.stage, args.output, args.partner_local, args.out, args.anchor_local)
    else:
        result = timing_check(args.stage, args.output, args.require_full)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
