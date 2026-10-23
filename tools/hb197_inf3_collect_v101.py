"""INF3 completed-run collection / complete-wall runtime v1.0.1, offline only."""
from __future__ import annotations
import argparse
import ast
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hb197_inf3_qualification_v101 as b
from tools import strict_variant_gate_inf3_v103 as old
need = b.need
ROLES = {'own': 'own', 'q': 'Q_A5q', 'raptor': 'R', 'r224': 'R224', 'r6': 'S_R7'}


def literal(filename, name):
    files = b.payload()[2]
    tree = ast.parse(files[filename])
    values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    need(len(values) == 1, 'unique frozen literal'); return values[0]


def positive(value):
    need(type(value) in (int, float) and math.isfinite(value) and value > 0, 'positive finite measured value')
    return float(value)


def table(inputs, item, targets):
    with inputs.file(item).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); rows = list(reader)
    need(reader.fieldnames == ['StudyInstanceUID', *targets], 'exact12 target order')
    ids = [r['StudyInstanceUID'] for r in rows]
    need(len(ids) == len(set(ids)) and all(ids), 'unique nonempty output IDs')
    for row in rows:
        values = [float(row[k]) for k in targets]
        need(len(values) == 12 and all(math.isfinite(x) and 0 <= x <= 1 for x in values), 'finite output probabilities')
    return ids


def journal(inputs, item, ids, members):
    rows = [json.loads(line) for line in inputs.file(item).read_text().splitlines() if line.strip()]
    seen = set(); errors = 0
    for row in rows:
        key = (row['uid'], row['member'])
        need(key not in seen and row['uid'] in ids and row['member'] in members, 'journal cohort/member/duplicate')
        seen.add(key)
        if 'error' in row: errors += 1
        else:
            values = row['values']
            need(len(values) == 12 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values), 'journal nonfinite/shape')
    need(seen == {(u, m) for u in ids for m in members}, 'incomplete full branch journal')
    return errors


def q_output(inputs, item, ids):
    import numpy as np
    with np.load(inputs.file(item), allow_pickle=False) as values:
        need(values['ids'].tolist() == ids, 'Q exact ordered cohort')
        for family in ('l2', 'dino', 'convnext'):
            a = values[family]
            need(a.dtype == np.float32 and a.shape == (10 if family == 'l2' else 5, len(ids), 12) and
                 np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), 'all20 Q leaves shape/precision/finite')


def sampled_memory(inputs, item):
    observed = inputs.json(item)
    need(observed['schema'] == 'h197-observer/v1.0.1' and observed['version'] == '1.0.1' and observed['status'] == 'STOPPED' and
         'Linux' in observed['platform'] and observed['owned_children_alive'] == [] and observed['watchdog_abort'] is None,
         'stopped Linux observer and dead owned children')
    need(observed['cleanup_complete'] is True and observed['cleanup_errors'] == [] and
         observed['observer_started'] is True and observed['observer_thread_stopped'] is True and
         observed['original_failure'] is None, 'successful complete diagnostic lifecycle')
    need(observed['source_sha256'] == b.sha(ROOT / 'tools/hb197_inf3_observer_v101.py'), 'actual observer source')
    need(observed['telemetry_errors'] == [] and len(observed['samples']) >= 2 and observed['process_inventory'],
         'complete measured hardware/process evidence')
    devices = {}
    for sample in observed['samples']:
        need(sample['gpu'], 'missing GPU sample')
        for gpu in sample['gpu']:
            total = positive(gpu['total_mib']); used = gpu['used_mib']
            need(type(used) in (int, float) and math.isfinite(used) and 0 <= used <= total and gpu['uuid'] and gpu['name'], 'actual GPU shape/occupancy')
            identity = (gpu['name'], total)
            if gpu['uuid'] in devices: need(devices[gpu['uuid']]['identity'] == identity, 'GPU identity drift')
            else: devices[gpu['uuid']] = dict(identity=identity, max_sampled_mib=0.)
            devices[gpu['uuid']]['max_sampled_mib'] = max(devices[gpu['uuid']]['max_sampled_mib'], used)
    return dict(environment='Kaggle Linux CUDA', devices=devices,
                metric='sampled device occupancy; not peak_reserved or upper bound',
                local_4070_policy_satisfied=False), observed


def execution(inputs, manifest_ref, build, *, fixture=False):
    spec = inputs.json(manifest_ref)
    need(spec['schema'] == 'h197-completed-run/v1' and spec['fixture_only'] is fixture and
         spec['candidate'] == b.science.KEY and spec['kind'] in ('FULL_VISIBLE', 'N32'), 'exact completed run identity')
    kind = spec['kind']; n = 3 if kind == 'FULL_VISIBLE' else 32
    remote = inputs.json(spec['remote'])
    expected_slug = b.science.SLUG if kind == 'FULL_VISIBLE' else b.SLUG
    expected_hash = b.KERNEL if kind == 'FULL_VISIBLE' else build['diagnostic_kernel_sha256']
    metadata_hash = b.METADATA if kind == 'FULL_VISIBLE' else build['diagnostic_metadata_sha256']
    need(remote['schema'] == 'h197-remote-completion/v1' and remote['recorded_by'] == 'B' and
         remote['status'] == 'COMPLETE' and remote['slug'] == expected_slug and
         type(remote['version']) is int and remote['version'] > 0 and remote['fixture_only'] is fixture and
         remote['owned_children_dead'] is True and remote['fast'] is False, 'actual completed remote/owned lifecycle')
    start, end = [datetime.fromisoformat(remote[k]) for k in ('executing_started_at', 'completed_at')]
    need(start.tzinfo is not None and end.tzinfo is not None and end > start, 'actual executing wall chronology')
    wall = (end - start).total_seconds()
    downloaded = inputs.file(spec['notebook']); metadata_path = inputs.file(spec['metadata'])
    need(b.sha(downloaded) == expected_hash and b.sha(metadata_path) == metadata_hash, 'wrong release/harness source or metadata')
    metadata = inputs.json(spec['metadata']); need(metadata['id'] == expected_slug, 'exact downloaded slug')
    actual_meta = inputs.json(spec['remote_metadata'])
    need(actual_meta['id'] == expected_slug and type(actual_meta['versionNumber']) is int and
         actual_meta['versionNumber'] == remote['version'], 'actual remote metadata version')
    for item in remote['raw_status_evidence']: inputs.file(item)
    need(remote['raw_status_evidence'], 'raw executing/completion evidence required')
    console_path = inputs.file(spec['console']); console = console_path.read_text(encoding='utf-8-sig', errors='replace')
    need(console and 'FAST_PATH_VISIBLE' not in console, 'real full execution console')
    ids = inputs.json(spec['ids']); targets = spec['targets']
    frozen = inputs.json({'path': old.FROZEN, 'sha256': old.FROZEN_SHA}) if not fixture else inputs.json(spec['fixture_frozen'])
    registered = inputs.json(frozen['roles']['bridge_registration'])['targets']
    need(targets == registered and len(targets) == 12 and len(ids) == len(set(ids)) == n and all(ids), 'registered targets/exact cohort count')
    need(table(inputs, spec['output'], targets) == ids, 'whole final output cohort')
    receipt = inputs.json(spec['release_receipt'])
    need(receipt['status'] == 'COMPLETE_C85R7' and receipt['studies'] == n and
         receipt['submission_sha256'] == spec['output']['sha256'] and receipt['exposure'] == 'EXPOSED' and
         receipt['clean_oof'] is False and receipt['codecs']['status'] == 'ACTIVE' and
         receipt['codecs']['s_r7_gate'] == 'ACTIVE', 'exact completed formula/output/active codecs')
    need(receipt['formula'] == literal('exp169_r224.py', 'FORMULA') and
         receipt['version'] == b.science.VERSION and receipt['findings'] == 12, 'exact scientific formula/version')
    need(set(receipt['fallbacks']) == {'a172_prediction', 'r_prediction', 's_r7_prediction', 'r224_prediction'} and
         all(type(v) is int and 0 <= v <= n for v in receipt['fallbacks'].values()), 'actual original fallback counts')
    need(set(spec['branches']) == set(ROLES), 'all five logical branches required')
    phase_records = []
    for line in console.splitlines():
        if line.startswith('C85_PHASE '):
            phase_records.append(json.loads(line.split(' ', 1)[1]))
    for role, phase_name in ROLES.items():
        part = spec['branches'][role]; phase = inputs.json(part['exit'])
        need(phase['phase'] == phase_name and phase['exit_code'] == 0 and phase in phase_records and
             positive(phase['seconds']) > 0, 'actual phase log/exit missing or nonzero')
        need(part['logs'], 'each branch actual child log required')
        for log in part['logs']: need(inputs.file(log).stat().st_size > 0, 'empty child log')
        if role == 'own': journal(inputs, part['journal'], ids, ['EXP143v2', 'EXP155'])
        elif role == 'q':
            q_output(inputs, part['leaves'], ids)
            timing = inputs.json(part['timings'])
            need(len(timing['completed']) == 2 and sum(timing['completed']) == n and len(timing['study_seconds']) == n and
                 all(positive(v) > 0 for v in timing['study_seconds']) and len(timing['gpu_peak_bytes']) == 2 and
                 all(positive(v) > 0 for v in timing['gpu_peak_bytes']), 'actual Q completed timings and allocator samples')
            manifest_path = inputs.file(part['manifest'])
            need(b.sha(manifest_path) == b.payload()[3]['original']['q-a5q-manifest.json'], 'exact20 Q manifest identity')
            journal(inputs, part['journal'], ids, [str(i) for i in range(20)])
        elif role == 'raptor': journal(inputs, part['journal'], ids, ['R'])
        elif role == 'r224':
            need(set(table(inputs, part['raw'], targets)) == set(ids) and set(table(inputs, part['ranked'], targets)) == set(ids), 'R224 silently used visible3')
            raw_receipt = inputs.json(part['receipt'])
            need(raw_receipt['status'] == 'COMPLETE_R224' and
                 raw_receipt['checkpoint_sha256'] == literal('exp169_r224.py', 'CHECKPOINT_SHA'), 'R224 actual completion/checkpoint')
            need(raw_receipt['files']['raptor224_raw.csv']['sha256'] == part['raw']['sha256'] and
                 raw_receipt['files']['raptor224_sub.csv']['sha256'] == part['ranked']['sha256'], 'R224 output identity')
        else:
            controller = inputs.json(part['controller']); workers = part['workers']
            need(controller['status'] == 'PASS' and controller['rows'] == n and workers, 'R6 actual controller/worker coverage')
            codes = controller['worker_exit_codes']; need(set(codes) == set(workers) and all(type(v) is int and v == 0 for v in codes.values()), 'every actual R6 child exit')
            seen = []
            for key, worker in workers.items():
                assigned = inputs.json(worker['ids']); output_ids = table(inputs, worker['output'], targets)
                wr = inputs.json(worker['receipt']); log = inputs.file(worker['log']).read_text(errors='replace')
                need(len(assigned) == len(set(assigned)) and set(output_ids) <= set(assigned) and
                     set(assigned) - set(output_ids) <= set(wr['fallback']) and
                     wr['status'] == 'PASS' and wr['studies'] == len(assigned) and
                     wr['device'] == 'cuda' and wr['cuda_initialized'] is True and 'WORKER_DONE' in log,
                     'actual full R6 worker cohort/device/log')
                seen.extend(assigned)
            need(len(seen) == len(set(seen)) == n and set(seen) == set(ids), 'R6 full same cohort')
            need(table(inputs, part['output'], targets) == ids, 'R6 controller output cohort')
            invocation = inputs.json(part['invocation']); provenance = inputs.json(part['provenance'])
            identity = inputs.json(part['identity'])
            need(invocation['device'] == 'cuda' and '--device' in invocation['args'] and
                 invocation['args'][invocation['args'].index('--device') + 1] == 'cuda', 'actual R6 CUDA invocation')
            for key in ('manifest_sha256', 'file_sha256', 'dataset_slug', 'bundle_id'):
                need(identity[key] == provenance[key], 'actual R6 model identity/provenance')
    need(set(spec['outer_exits']) == {'a172', 'public'}, 'actual parent child exits')
    for item in spec['outer_exits'].values():
        p = inputs.json(item); need(p['phase'] == 'child' and p['exit_code'] == 0 and p in phase_records, 'outer child exit/log')
    for item in spec['device_evidence']: inputs.file(item)
    need(spec['device_evidence'], 'actual CUDA/TTA/device evidence required for review')
    result = dict(kind=kind, slug=expected_slug, remote_version=remote['version'], studies=n,
        notebook_sha256=expected_hash, metadata_sha256=metadata_hash, executing_wall_seconds=wall,
        output=spec['output'], fallbacks=receipt['fallbacks'], formula=receipt['formula'],
        full_device_shape_requires_independent_review=True)
    if kind == 'N32':
        cohort = inputs.json(spec['cohort'])
        need(cohort['schema'] == 'h197-n32-cohort/v1' and cohort['studies'] == 32 and cohort['ids'] == ids, 'actual selected same32')
        need(set(spec['cohort_files']) == set(cohort['sha256']), 'all selection metadata required')
        for name, item in spec['cohort_files'].items():
            need(item['sha256'] == cohort['sha256'][name], 'selected metadata drift'); inputs.file(item)
        selection = inputs.json(spec['cohort_files']['selection.json'])
        need(selection['n'] == 32 and selection['method'] == 'round-robin max/min series/max/min slices' and
             [v['uid'] for v in selection['selection']] == ids, 'actual deterministic selector roster')
        for name in ('test.csv', 'test_series.csv'):
            with inputs.file(spec['cohort_files'][name]).open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            got = [r['StudyInstanceUID'] for r in rows]
            need(got == ids if name == 'test.csv' else set(got) == set(ids), 'actual selected metadata same32')
        need(table(inputs, spec['cohort_files']['sample_submission.csv'], targets) == ids, 'same32 sample template')
        result['memory'], observed = sampled_memory(inputs, spec['observer'])
        result['complete_wall_seconds'] = max(wall, positive(observed['monotonic_wall_seconds']))
        cost = inputs.json(spec['global_cost'])
        need(cost['schema'] == 'h197-global-cost/v1' and cost['status'] == 'COMPLETE_SYNTHETIC_COST_ONLY' and
             cost['model_forward'] is False and cost['real_predictions'] is False and cost['repeats'] >= 3 and
             len(cost['measured_seconds']) == cost['repeats'] and
             cost['seconds'] == max(positive(v) for v in cost['measured_seconds']), 'actual whole-global synthetic benchmark')
        proof = inputs.json(spec['diagnostic_diff'])
        need(proof == b.payload()[3], 'exact reviewed diagnostic diff')
        need(cost['source_sha256'] == proof['diagnostic'] and cost['observer_source_sha256'] == b.sha(ROOT / 'tools/hb197_inf3_observer_v101.py'), 'actual benchmark full source inventory')
        h = cost['hidden_studies']; need(type(h) is int and h >= 32, 'benchmark cohort size')
        need(cost['leaf_shapes'] == {f: [10 if f == 'l2' else 5, h, 12] for f in ('l2', 'dino', 'convnext')} and
             cost['leaf_dtype'] == 'float32' and cost['pool_dtype'] == 'float64', 'benchmark actual full shapes')
        result['global_cost'] = cost
    return result


def assemble(kind, output_root, remote_path, notebook, metadata, remote_metadata, console, destination):
    """Inventory completed downloaded output only. No polling/download or launch."""
    root = Path(output_root).resolve(); remote_path = Path(remote_path).resolve()
    remote = json.loads(remote_path.read_text())
    need(remote.get('status') == 'COMPLETE' and remote.get('owned_children_dead') is True and
         remote.get('fixture_only') is False, 'assemble only completed dead-child production run')
    def ref(name): return old.ref(root / name)
    adir = 'c85r7-run/a172/'; pdir = 'c85r7-run/pr7/'; rdir = pdir + 'pr7-run/'; sdir = rdir + 's_r7/'
    branches = {}
    console_ref = old.ref(console)
    for role, name in ROLES.items():
        folder = adir if role in ('own', 'q') else pdir
        branches[role] = dict(exit=ref(folder + name + '-exit.json'), logs=[console_ref])
    branches['own']['journal'] = ref(adir + 'exp169-trio/predictions.jsonl')
    branches['q'].update(journal=ref(adir + 'q-journal/predictions.jsonl'), leaves=ref(adir + 'partner-raw.npz'),
        timings=ref(adir + 'q-timings.json'), manifest=ref(adir + 'q-a5q-manifest.json'))
    branches['raptor']['journal'] = ref(rdir + 'r/predictions.jsonl')
    branches['r224'].update(raw=ref('raptor224_raw.csv'), ranked=ref('raptor224_sub.csv'), receipt=ref(rdir + 'r224-receipt.json'))
    controller = json.loads((root / sdir / 'controller/submission_receipt.json').read_text())
    workers = {}
    for key in controller['worker_exit_codes']:
        need(str(key).isdecimal(), 'actual numeric worker identity')
        prefix = sdir + 'controller/worker' + str(key) + '/'
        workers[str(key)] = dict(ids=ref(prefix + 'ids.json'), output=ref(prefix + 'predictions.csv'),
            receipt=ref(prefix + 'receipt.json'), log=ref(prefix + 'log.txt'))
    branches['r6'].update(controller=ref(sdir + 'controller/submission_receipt.json'), workers=workers,
        output=ref(sdir + 'controller/submission.csv'), invocation=ref(sdir + 'invocation.json'),
        provenance=ref(sdir + 'provenance.json'), identity=ref(sdir + 'identity.json'))
    frozen = json.loads((ROOT / old.FROZEN).read_text())
    targets = json.loads((ROOT / frozen['roles']['bridge_registration']['path']).read_text())['targets']
    spec = dict(schema='h197-completed-run/v1', fixture_only=False, candidate=b.science.KEY, kind=kind,
        remote=old.ref(remote_path), notebook=old.ref(notebook), metadata=old.ref(metadata), remote_metadata=old.ref(remote_metadata),
        console=console_ref, ids=ref(adir + 'ids.json'), targets=targets, output=ref('submission.csv'),
        release_receipt=ref('c85r7-receipt.json'), branches=branches,
        outer_exits={k: ref(folder + 'child-exit.json') for k, folder in [('a172', adir), ('public', pdir)]},
        device_evidence=[console_ref, ref(adir + 'q-timings.json'), ref(sdir + 'invocation.json'), ref(sdir + 'provenance.json'),
                         *[v['log'] for v in workers.values()]])
    if kind == 'N32':
        spec.update(cohort=ref('inf3-n32-data/h197-cohort.json'),
            cohort_files={n: ref('inf3-n32-data/' + n) for n in ('test.csv', 'test_series.csv', 'sample_submission.csv', 'selection.json')},
            observer=ref('h197-observer.json'), global_cost=ref('h197-global-cost.json'),
            diagnostic_diff=old.ref(b.WORK / 'EMBEDDED-DIFF.json'))
    # Preserve the inventory even on validation failure; it never grants admission.
    temp_inputs = old.Inputs()
    b.write(Path(destination), b.encoded(spec))
    execution(temp_inputs, old.ref(destination), b.validate())
    temp_inputs.recheck()
    return spec


def projection(n32, population):
    need(population['schema'] == 'h197-hidden-population/v1' and population['recorded_by'] == 'B' and
         population['applicable'] is True and type(population['studies']) is int and population['studies'] >= 32, 'applicable authenticated hidden workload')
    h = population['studies']; cost = n32['global_cost']
    need(n32['kind'] == 'N32' and n32['studies'] == 32 and cost['hidden_studies'] == h, 'full32/full-H timing identity')
    wall = positive(n32['complete_wall_seconds']); global_seconds = positive(cost['seconds'])
    value = h / 32 * wall + global_seconds
    return dict(schema='h197-complete-wall/v1', status='PASS' if value <= 32400 else 'HOLD', hidden_studies=h,
        n32_complete_wall_seconds=wall, full_global_seconds=global_seconds, projected_seconds=value, gate_seconds=32400,
        formula='H/32 * max(monotonic complete N32 wall, actual remote executing wall) + measured complete global H cost',
        conservative_overcount='fixed/diagnostic/32-study global work scaled; full global cost added without deductions',
        uncertainty='observed selected32/hardware only; hidden workload and node variance remain')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run'); p.add_argument('--out', required=True)
    p.add_argument('--assemble', choices=['FULL_VISIBLE', 'N32']); p.add_argument('--output-root'); p.add_argument('--remote')
    p.add_argument('--notebook'); p.add_argument('--metadata'); p.add_argument('--remote-metadata'); p.add_argument('--console')
    a = p.parse_args()
    if a.assemble:
        need(all(getattr(a, k) for k in ('output_root', 'remote', 'notebook', 'metadata', 'remote_metadata', 'console')), 'complete raw collection arguments')
        assemble(a.assemble, a.output_root, a.remote, a.notebook, a.metadata, a.remote_metadata, a.console, a.out)
        print(json.dumps(dict(status='COLLECTED_NOT_ADMITTED', kind=a.assemble))); return 0
    need(a.run, '--run required'); build = b.validate(); inputs = old.Inputs(); result = execution(inputs, old.ref(a.run), build)
    inputs.recheck(); result['consumed_sha256'] = inputs.consumed
    b.write(Path(a.out), b.encoded(result)); print(json.dumps({k: result[k] for k in ('kind', 'slug', 'remote_version', 'studies')})); return 0


if __name__ == '__main__': raise SystemExit(main())
