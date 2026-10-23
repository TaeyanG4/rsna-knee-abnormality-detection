"""CPU-only stage builder and fail-closed D01 production adapter.

No Kaggle upload/submission commands. Runtime is explicitly Kaggle two-T4 only.
Frozen original30 anchor -> process cleanup -> frozen v3 R2 -> all-N rank90/10.
"""
from __future__ import annotations
import argparse
import ast
import base64
import csv
import hashlib
import importlib.util
import io
import json
import lzma
import math
import os
from pathlib import Path
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
UID = 'StudyInstanceUID'
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
V3_SHA = '3800aec2662bf8f0d8a0219cd0adb8d50ad97f1943c63c45cadfcdf55c434ba9'
WRAPPER_SHA = 'e0bbc49731e35c36c5cc7e4f6ce4f7badd8c6090ea0c9b7acbca181548dbbd75'
ANCHOR_SHA = 'cd5b8b69f89bcfaa6b6ee065f7b18f7134dc71968be3827c87ca3e5672007cac'
RESULT_SHA = '7f330b74592258efdee34a1da96842d60649db7c628cc0cbafa513075793ba5d'
PROTOCOL_SHA = '5ef7bc3383ed8628c62b0ba9405b7d5e73cf4b30932e90d8013e6a06a70de0ea'
WHOLE_SECONDS, RESERVE_SECONDS = 32400, 300
SMALL_COHORT_MAX_SECONDS = 1800


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def resolve_competition(candidates=None):
    """One actual complete competition root, no proxy or existence patch."""
    if candidates is None:
        candidates = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                      Path('/kaggle/input/rsna-knee-abnormality-detection')]
    found = {}
    for p in map(Path, candidates):
        if all((p / n).is_file() for n in ('test.csv', 'test_series.csv', 'sample_submission.csv', 'train.csv')) and (p / 'test_series').is_dir():
            found[str(p.resolve())] = p
    need(len(found) == 1, f'exactly one complete actual test root required; got {len(found)}')
    return next(iter(found.values()))


def test_ids(competition):
    with (Path(competition) / 'test.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = csv.DictReader(f); need(UID in rows.fieldnames, 'test UID column missing')
        ids = [r[UID] for r in rows]
    need(ids and len(ids) == len(set(ids)) and all(u and '/' not in u and '\\' not in u and '..' not in u for u in ids), 'nonempty unique safe test IDs required')
    return ids


def budget(n, result):
    need(type(n) is int and n > 0, 'positive actual N required')
    neff = max(n, 133)
    core = result['production_core_seconds']
    need(math.isfinite(core) and core > 0, 'finite measured production time required')
    estimate = max(core * neff / 133, 30938.468180919 * neff / 1300)
    phases = {name: result['phases'][name]['whole_phase_seconds'] * neff / 133 for name in ('anchor', 'r2')}
    return {'actual_n': n, 'effective_n': neff, 'projected_seconds': estimate,
            'phase_estimates_seconds': phases, 'phase_estimates_are_not_caps': True,
            'whole_seconds': WHOLE_SECONDS, 'minimum_reserve_seconds': RESERVE_SECONDS,
            'internal_deadline_seconds': WHOLE_SECONDS - RESERVE_SECONDS,
            'minimum_reserve_pass': estimate + RESERVE_SECONDS <= WHOLE_SECONDS,
            'remaining_seconds_before_reserve': WHOLE_SECONDS - estimate,
            'scope': 'actualN/Neff conservative engineering estimate; not hidden-runtime or memory guarantee'}


def execution_seconds(n):
    """Bound small visible cohorts without changing hidden production budget."""
    need(type(n) is int and n > 0, 'positive actual N required for execution deadline')
    return SMALL_COHORT_MAX_SECONDS if n <= 3 else WHOLE_SECONDS - RESERVE_SECONDS


def verify_trial(stage, result_path):
    """Exact observed v2 result, downloaded dependencies, source/policy identity."""
    stage, result_path = Path(stage), Path(result_path)
    need(sha(result_path) == RESULT_SHA, 'exact integrated version2 PASS bytes required')
    result = read(result_path); remote = result_path.parent
    need(result['schema'] == 'd01-integrated133-trial-v2-path' and result['status'] == 'PASS'
         and result['readiness'] == 'READY_FOR_EXPLORATORY_SUBMISSION' and not result['failures'], 'integrated readiness gate')
    dependencies = {'production-core-receipt.json': result['production_core_receipt_sha256'],
                    'anchor/anchor-audit.json': result['anchor_audit_sha256'],
                    'anchor-process/phase-receipt.json': result['core_receipts']['anchor_phase'],
                    'r2/receipt.json': result['core_receipts']['r2'],
                    'r2-process/phase-receipt.json': result['core_receipts']['r2_phase'],
                    'public3/t4-gate.json': result['public3_gate_sha256'],
                    'public3-process/phase-receipt.json': None,
                    'integrated-trial-result.json': RESULT_SHA}
    bindings = {}
    for name, digest in dependencies.items():
        p = remote / name; actual = sha(p); d = read(Path(str(p) + '.download.json'))
        need(d['kernel'] == 'taeyangg4/knee-d01-integrated133' and d['version'] == 2
             and d['file'] == 'd01-integrated133-v2/results/' + name and d['status'] == 'DOWNLOADED'
             and d['sha256'] == actual and d['bytes'] == p.stat().st_size, 'exact-version dependency download: ' + name)
        need(digest is None or digest == actual, 'dependency hash mismatch: ' + name)
        bindings[name] = {'sha256': actual, 'result_byte_sha_binding': digest is not None}
    for phase in ('anchor', 'r2', 'public3'):
        need(read(remote / f'{phase}-process/phase-receipt.json') == result['phases'][phase], 'phase JSON identity: ' + phase)
    public = read(remote / 'public3/t4-gate.json')
    need(public['status'] == 'PASS' and all(public['checks'].values()) and public['release_helper_sha256'] == V3_SHA, 'public3 gate')
    payload = stage / 'payload'
    for name, digest in read(payload / 'trial-manifest.json')['files'].items():
        need(sha(payload / name) == digest, 'trial payload changed: ' + name)
    for name, digest in [('helper/d01_release_v3.py', V3_SHA), ('d01_integrated_trial_v2.py', WRAPPER_SHA),
                         ('anchor-original.ipynb', ANCHOR_SHA), ('trial-protocol.json', PROTOCOL_SHA)]:
        need(sha(payload / name) == digest, 'frozen source changed: ' + name)
    need(result['wrapper_sha256'] == WRAPPER_SHA and result['helper_sha256'] == V3_SHA and result['protocol_sha256'] == PROTOCOL_SHA, 'result source identities')
    helper = module(payload / 'helper/d01_release_v3.py', 'd01_candidate_frozen_gate')
    helper.validate_bundle(payload / 'helper')
    need(budget(1300, result)['minimum_reserve_pass'], 'frozen projected reserve gate')
    return result, bindings


def validate_frame(frame, ids):
    import numpy as np
    need(list(frame.columns) == [UID, *LABELS], 'exact twelve-target schema/order required')
    need(frame[UID].tolist() == ids and len(set(ids)) == len(ids), 'exact final study order/coverage required')
    values = frame[LABELS].to_numpy(dtype=float)
    need(values.shape == (len(ids), 12) and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'finite bounded predictions required')


def verify_r2_output(base, work, ids, helper):
    import numpy as np
    import pandas as pd
    receipt = read(work / 'receipt.json')
    need(receipt['status'] == 'COMPLETE' and receipt['split'] == 'test' and receipt['ids'] == ids
         and receipt['studies'] == len(ids) and receipt['shape'] == [5, len(ids), 12]
         and receipt['member_count'] == 5 and receipt['all_finite'] and receipt['mean_before_rank']
         and receipt['fp32_fallbacks'] == 0 and receipt['neutral_fills'] == 0 and not receipt['failures'], 'R2 complete all-five actual-test contract')
    need(receipt['release_helper_sha256'] == V3_SHA and receipt['policy_sha256'] == helper.POLICY_SHA, 'R2 frozen helper/policy')
    reference = read(base / 'helper/reference-receipt.json')
    identities = [{'name': r['name'], 'sha256': r['checkpoint_sha256']} for r in reference['inference']['member_results']]
    need(receipt['member_identities'] == identities, 'R2 checkpoint identities changed')
    with np.load(work / 'members.npz', allow_pickle=False) as arr:
        raw = arr['member_probabilities']; need(arr['ids'].tolist() == ids and raw.dtype == np.float32 and raw.shape == (5, len(ids), 12), 'R2 NPZ identity/shape')
        need(np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all(), 'R2 raw probabilities')
        mean = raw.mean(axis=0, dtype=np.float32)
    frame = pd.read_csv(work / 'mean.csv', dtype={UID: str}); validate_frame(frame, ids)
    need(np.array_equal(frame[LABELS].to_numpy(dtype=np.float32), mean), 'five-member float32 mean before rank')
    return frame


def publish(frame, ids, work, final, receipt, deadline):
    """Only final validated all-N blend can claim the public output filename."""
    final = Path(final); validate_frame(frame, ids)
    need(not final.exists(), 'unexpected public output before final publication')
    need(time.monotonic() < deadline, 'shared deadline exhausted before publication')
    temp = work / 'submission.validated.csv'; frame.to_csv(temp, index=False)
    receipt.update(status='COMPLETE', submission_sha256=sha(temp), submission_rows=len(ids),
                   scope='fixed exploratory90/10 candidate; no independent quality gain or hidden-full guarantee')
    dump(work / 'candidate-receipt.json', receipt)
    need(time.monotonic() < deadline, 'shared deadline exhausted at publication')
    os.replace(temp, final)


def execute(base, started):
    base = Path(base); work = base / 'results'; work.mkdir(exist_ok=False)
    final = Path('/kaggle/working/submission.csv')
    need(not final.exists(), 'preexisting public submission; do not overwrite')
    receipt = {'schema': 'd01-candidate-v3', 'status': 'RUNNING', 'phases': {}, 'failures': [],
               'adapter_sha256': sha(__file__), 'integrated_result_sha256': RESULT_SHA}
    deadline = float(started) + WHOLE_SECONDS - RESERVE_SECONDS
    try:
        need(os.name == 'posix' and base.resolve().is_relative_to(Path('/kaggle/working')), 'explicit Kaggle Linux runtime required')
        manifest = read(base / 'candidate-manifest.json')
        for name, digest in manifest['files'].items():
            need(sha(base / name) == digest, 'candidate payload changed: ' + name)
        need(sha(base / 'helper/d01_release_v3.py') == V3_SHA and sha(base / 'd01_integrated_trial_v2.py') == WRAPPER_SHA
             and sha(base / 'anchor-original.ipynb') == ANCHOR_SHA, 'frozen runtime source identities')
        result = read(base / 'evidence/integrated-trial-result.json'); need(sha(base / 'evidence/integrated-trial-result.json') == RESULT_SHA, 'trial result changed')
        helper = module(base / 'helper/d01_release_v3.py', 'd01_candidate_helper'); helper.validate_bundle(base / 'helper')
        control = module(base / 'd01_integrated_trial_v2.py', 'd01_candidate_process_control')
        v2 = module(base / 'helper/d01_release_v2.py', 'd01_candidate_anchor_suffix')
        supervisor = module(base / 'helper/d01_r2_shard_benchmark.py', 'd01_candidate_supervisor')
        competition = resolve_competition(); ids = test_ids(competition)
        b = budget(len(ids), result); need(b['minimum_reserve_pass'], 'actualN projection lacks fixed5min reserve')
        phase_cap = execution_seconds(len(ids))
        deadline = float(started) + phase_cap
        receipt.update(competition_root=str(competition), actual_n=len(ids), budget=b,
                       deadline_seconds=phase_cap, deadline_policy='actualN<=3:1800s; actualN>3:32100s; both from first notebook cell',
                       provenance=manifest['provenance'])
        dump(work / 'test-ids.json', {'ids': ids})
        roots = [Path('/kaggle/input/knee-d01-r2-assets'), Path('/kaggle/input/datasets/taeyangg4/knee-d01-r2-assets')]
        assets = {str(p.parent.resolve()): p.parent for root in roots if root.is_dir()
                  for p in root.rglob('d01-assets-manifest.json') if sha(p) == helper.ASSETS_SHA}
        need(len(assets) == 1, 'one exact R2 asset mount required'); assets = next(iter(assets.values()))
        codecs = {str(p.resolve()): p for p in [Path('/kaggle/input/knee-d01-codec-assets'), Path('/kaggle/input/datasets/taeyangg4/knee-d01-codec-assets')] if p.is_dir()}
        need(len(codecs) == 1, 'one codec mount required'); codec = next(iter(codecs.values()))
        anchor = work / 'anchor'; anchor.mkdir()
        original = read(base / 'anchor-original.ipynb'); nb = v2.anchor_worker_notebook(original)
        need(len(original['cells']) == 30 and nb['cells'][:30] == original['cells'], 'original30 cells changed')
        dump(anchor / 'worker.ipynb', nb)
        env = dict(os.environ, RSNA_COMP_ROOT=str(competition), D01_ANCHOR_OUTPUT=str(anchor / 'anchor.csv'))
        for key in ('RSNA_PARALLEL_COAT_READERS', 'RSNA_CACHE_RAM_GIB', 'RSNA_PIXEL_SCRATCH', 'CUDA_VISIBLE_DEVICES'):
            env.pop(key, None)
        command = [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', str(anchor / 'worker.ipynb'),
                   '--output', 'executed.ipynb', '--output-dir', str(anchor), '--ExecutePreprocessor.timeout=-1']
        receipt['phases']['anchor'] = control.run_phase('anchor', command, env, work / 'anchor-process', phase_cap, deadline, supervisor)
        need(receipt['phases']['anchor']['all_owned_processes_reaped'], 'anchor descendants remain')
        audit = read(anchor / 'anchor.json')
        need(audit['status'] == 'COMPLETE' and audit['cohort'] == len(ids) and audit['dino_members'] == 20
             and audit['a5_folds'] == 5 and audit['raptor_views'] == 4 and audit['coat_family_members'] == 4, 'complete original anchor cohort/member graph')
        need(not final.exists(), 'anchor public output not quarantined by suffix')
        receipt['anchor_audit_sha256'] = sha(anchor / 'anchor.json')
        command = [sys.executable, str(base / 'd01_candidate_v3.py'), 'r2', '--base', str(base), '--assets', str(assets),
                   '--codec', str(codec), '--competition', str(competition), '--work', str(work / 'r2'),
                   '--ids', str(work / 'test-ids.json'), '--deadline', str(deadline)]
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'); env.pop('CUDA_VISIBLE_DEVICES', None)
        receipt['phases']['r2'] = control.run_phase('r2', command, env, work / 'r2-process', phase_cap, deadline, supervisor)
        need(receipt['phases']['r2']['all_owned_processes_reaped'], 'R2 descendants remain')
        import numpy as np
        import pandas as pd
        r2 = verify_r2_output(base, work / 'r2', ids, helper)
        anc = pd.read_csv(anchor / 'anchor.csv', dtype={UID: str})
        frame = helper.blend(assets, anc, r2, ids); validate_frame(frame, ids)
        expected = .9 * anc.set_index(UID).loc[ids, LABELS].rank(method='average', pct=True) + .1 * r2.set_index(UID).loc[ids, LABELS].rank(method='average', pct=True)
        need(np.array_equal(frame[LABELS].to_numpy(), expected.to_numpy()), 'whole-cohort fixed90/10 rank differs')
        receipt['r2_receipt_sha256'] = sha(work / 'r2/receipt.json')
        receipt['elapsed_seconds_before_publish'] = time.monotonic() - float(started)
        publish(frame, ids, work, final, receipt, deadline)
    except BaseException as exc:
        # This invocation established absence before launching either owned phase.
        # The original anchor writes its absolute final filename before its suffix;
        # any controlled failure removes that transient file, never submits anchor alone.
        if final.exists():
            final.unlink()
        receipt.update(status='FAILED'); receipt['failures'].append({'error': str(exc), 'traceback': traceback.format_exc()})
        dump(work / 'candidate-receipt.json', receipt)
        raise
    return receipt


def worker(base, assets, codec, competition, work, ids, deadline):
    base = Path(base); need(sha(base / 'helper/d01_release_v3.py') == V3_SHA, 'worker frozen helper identity')
    helper = module(base / 'helper/d01_release_v3.py', 'd01_candidate_r2')
    ordered = read(ids)['ids']; need(ordered == test_ids(competition), 'worker must use exact actual test order')
    return helper.run_r2(base / 'helper', assets, codec, competition, work, ordered, 'test', float(deadline), False)


def stage(trial_stage, trial_result, out):
    trial_stage, out = Path(trial_stage), Path(out)
    need(not out.exists(), 'new immutable stage required')
    result, bindings = verify_trial(trial_stage, trial_result)
    raw = {name: (trial_stage / 'payload' / name).read_bytes()
           for name in read(trial_stage / 'payload/trial-manifest.json')['files']}
    # Keep original helper evidence (including protocol) byte-for-byte; trial
    # controller is reused only for process control, never execute/worker/proxy.
    raw['d01_candidate_v3.py'] = Path(__file__).read_bytes()
    raw['evidence/integrated-trial-result.json'] = Path(trial_result).read_bytes()
    raw['evidence/dependency-bindings.json'] = (json.dumps(bindings, indent=2) + '\n').encode()
    metadata = read(trial_stage / 'kernel-metadata.json')
    raw['evidence/tested-stage-metadata.json'] = (trial_stage / 'kernel-metadata.json').read_bytes()
    provenance = {'trial_kernel': 'taeyangg4/knee-d01-integrated133', 'trial_version': 2,
                  'trial_result_sha256': RESULT_SHA,
                  'submitted_local_metadata_sha256': sha(trial_stage / 'kernel-metadata.json'),
                  'dataset_refs': metadata['dataset_sources'], 'kernel_refs': metadata['kernel_sources'],
                  'resolved_dataset_revision_ids': 'UNKNOWN; local submitted refs remain unversioned',
                  'resolved_kernel_input_revision_ids': 'UNKNOWN; local submitted refs remain unversioned',
                  'limitation': 'Same recorded input refs/docker/model as trial. Fingerprints/checkpoint subsets are checked by original code; not independent SHA of every mounted DINO file or every server-resolved revision.',
                  'engineering_only': '133 source/input/numerical/runtime PASS; no independent accuracy or hidden1300 memory/runtime guarantee'}
    manifest = {'schema': 'd01-candidate-v3-manifest', 'files': {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
                'provenance': provenance, 'minimum_reserve_seconds': RESERVE_SECONDS, 'whole_seconds': WHOLE_SECONDS,
                'execution_seconds': {'actual_n_le_3': SMALL_COHORT_MAX_SECONDS, 'actual_n_gt_3': WHOLE_SECONDS-RESERVE_SECONDS},
                'blend': {'anchor_rank': .9, 'r2_rank_mean5': .1, 'ties': 'average', 'final_rerank': False}}
    raw['candidate-manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, value in sorted(raw.items()):
            entry = tarfile.TarInfo(name); entry.size = len(value); entry.mtime = 0; entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(value))
    compressed = lzma.compress(buffer.getvalue(), preset=9); packed = base64.b85encode(compressed).decode('ascii')
    expected = {k: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for k, v in raw.items()}
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(base64.b85decode(packed))), mode='r:') as archive:
        need(set(archive.getnames()) == set(raw), 'archive file set changed')
        for entry in archive.getmembers():
            need(entry.isfile() and archive.extractfile(entry).read() == raw[entry.name], 'archive roundtrip failed')
    code = f'''from pathlib import Path
import ast,base64,hashlib,importlib.util,io,json,lzma,sys,tarfile,time
started=time.monotonic()
base=Path('/kaggle/working/d01-candidate-v3');base.mkdir(exist_ok=False)
expected={expected!r}
compressed=base64.b85decode({packed!r})
assert hashlib.sha256(compressed).hexdigest()=={hashlib.sha256(compressed).hexdigest()!r}
with tarfile.open(fileobj=io.BytesIO(lzma.decompress(compressed)),mode='r:') as archive:
    entries=archive.getmembers()
    assert len(entries)==len(expected) and {{e.name for e in entries}}==set(expected)
    for entry in entries:
        assert entry.isfile() and not Path(entry.name).is_absolute() and '..' not in Path(entry.name).parts
        value=archive.extractfile(entry).read();item=expected[entry.name]
        assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
        if entry.name.endswith('.py'):ast.parse(value.decode('utf-8'))
        target=base/entry.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
spec=importlib.util.spec_from_file_location('d01_candidate_v3',base/'d01_candidate_v3.py')
candidate=importlib.util.module_from_spec(spec);sys.modules['d01_candidate_v3']=candidate;spec.loader.exec_module(candidate)
result=candidate.execute(base,started)
print(json.dumps({{k:result[k] for k in ('status','actual_n','budget','submission_sha256','scope')}},indent=2))
'''
    ast.parse(code)
    for name, value in raw.items():
        if name.endswith('.py'): ast.parse(value.decode('utf-8'))
    notebook = {'nbformat': 4, 'nbformat_minor': 4, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# D01 fixed exploratory90/10 candidate\nPrivate CPU stage; execution and competition submission are separate decisions.\n']},
                          {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)}]}
    nb_bytes = (json.dumps(notebook, ensure_ascii=False, indent=2) + '\n').encode()
    need(len(nb_bytes) <= 1_000_000, 'candidate exceeds notebook transport limit')
    metadata.update(id='taeyangg4/knee-d01-candidate-v3', title='knee-d01-candidate-v3', code_file='d01-candidate-v3.ipynb',
                    is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False)
    out.mkdir(parents=True)
    for name, value in raw.items():
        p = out / 'payload' / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(value)
    (out / metadata['code_file']).write_bytes(nb_bytes); dump(out / 'kernel-metadata.json', metadata)
    review = {'status': 'CPU_STAGED_NO_REMOTE_OR_GPU', 'adapter_sha256': sha(__file__), 'notebook_sha256': sha(out / metadata['code_file']),
              'notebook_bytes': len(nb_bytes), 'payload_files': len(raw), 'integrated_result_sha256': RESULT_SHA,
              'trial_dependent_receipts_verified': bindings, 'original30cells': 'unchanged via frozen v2 suffix only',
              'production_input': 'single actual test root; no train proxy or existence monkeypatch',
              'budget1300': budget(1300, result), 'provenance': provenance,
              'execution_seconds': manifest['execution_seconds'], 'planned_kaggle_session_timeout_seconds': WHOLE_SECONDS,
              'small_cohort_cost_bound': 'actualN<=3 internal1800s shared deadline; two-T4 planning bound1GPUh plus process/platform cleanup overhead; no automatic retry',
              'failure_scope': 'controlled failure quarantines/removes anchor final; abrupt external SIGKILL/VM termination cannot guarantee Python cleanup',
              'next_action': 'root independent CPU review and private launch decision; competition submission requires separate approval'}
    dump(out / 'stage-review.json', review)
    return review


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('stage')
    for name in ('trial-stage', 'trial-result', 'out'): p.add_argument('--' + name, required=True)
    p = sub.add_parser('r2')
    for name in ('base', 'assets', 'codec', 'competition', 'work', 'ids'): p.add_argument('--' + name, required=True)
    p.add_argument('--deadline', type=float, required=True)
    args = vars(parser.parse_args()); action = args.pop('action')
    if action == 'stage': print(json.dumps(stage(**args), indent=2))
    else: worker(**args)
