"""H-B-227 v1.0.0: actual embedded vector runtime over cached real predictions.

No inference, labels printed, GPU, network, new statistics, or qualification.
The 976 R7-f0 parent boundary is a proxy; this is not full INF3 execution.
"""
import json
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from scipy.stats import rankdata

from tools import hb227_alpha_confirm_v1 as confirmation
from tools import k1n_stage_v1 as builder
from tools import k1n_cpu_replay_v1 as csv_util

ROOT, OUT = confirmation.ROOT, confirmation.OUT
CONFIG = ROOT / 'configs/k1n/x3-stride2-alpha-vector-v2.json'


def main():
    config = json.loads(CONFIG.read_bytes())
    # Only replay predictions here; never rerun metrics or the frozen comparison.
    ids, primary, _refs, parent, take = confirmation.context.context()
    arrays = [confirmation.context.load(confirmation.PREDS / (m + '_s2.csv'), ids)[take]
              .astype(np.float32) for m in ('v2', 's0', 's1')]
    parent = parent.astype(np.float32)
    result = dict(scope='real cached 976/32 prediction boundary; R7-f0 proxy parent; no classifier or full INF3 inference',
                  cuda_requested=False, cases=[])
    for diagnostic in (False, True):
        stage = builder.stage(config, diagnostic)
        registration = json.loads((stage / 'registration.json').read_bytes())
        notebook = (stage / 'kernel/release.ipynb').read_bytes()
        assert builder.digest(notebook) == registration['kernel_sha256']
        nb = json.loads(notebook)
        files = builder.before.parent.archive(''.join(nb['cells'][2]['source']))[2]
        assert {n: builder.digest(v) for n, v in files.items()} == registration['sources']
        r = ModuleType('hb227_embedded_smoke')
        exec(compile(files['tools/k1sr_runtime_v7.py'], '<registered stage runtime>', 'exec'), r.__dict__)
        assert r.CONFIG_SHA == registration['config_sha256']
        size = 32 if diagnostic else 976
        roster, pp = primary[:size], parent[:size]
        aa = [a[:size] for a in arrays]
        parent_bytes = csv_util.csv_bytes(roster, pp)
        rows = [dict(uid=uid, ok=True, config_sha256=r.CONFIG_SHA, k1_models=list(r.MODEL_IDS),
                     finished_elapsed_seconds=10.,
                     predictions={m: a[i].tolist() for m, a in zip(r.MODEL_IDS, aa)})
                for i, uid in enumerate(roster)]
        rank = lambda x: rankdata(x, method='average', axis=0) / len(x)
        weights = np.array(config['alpha_vector'])
        expected = rank((1-weights)*rank(pp) + weights*rank(np.mean([rank(a) for a in aa], axis=0))).astype(np.float32)
        for fault in ('ok', 'missing_model', 'wrong_vector'):
            path = OUT / 'cpu-smoke' / ('n32' if diagnostic else 'release') / fault
            path.mkdir(parents=True, exist_ok=True)
            source_name = 'diagnostic-c85r7.csv' if diagnostic else 'submission.csv'
            (path / source_name).write_bytes(parent_bytes)
            worker = path / 'run/pr7/pr7-run/s_r7/controller/worker0'
            worker.mkdir(parents=True, exist_ok=True)
            replay_rows = json.loads(json.dumps(rows))
            if fault == 'missing_model':
                replay_rows[0]['predictions'].pop(r.MODEL_IDS[-1])
            if fault == 'wrong_vector':
                replay_rows[0]['config_sha256'] = '0'*64
            (worker / 'k1-studies.jsonl').write_text('\n'.join(map(json.dumps, reversed(replay_rows))), encoding='utf-8')
            # write_once remains immutable; use a verify-existing shim only for
            # this fixed scratch's already-identical parent artifact on reruns.
            original_write_once = r.util.write_once
            def same_parent_or_write(dest, payload):
                dest = Path(dest)
                if dest.exists():
                    assert dest == path / 'k1sr-parent.csv' and dest.read_bytes() == payload
                else:
                    original_write_once(dest, payload)
            with patch.object(r.util, 'write_once', same_parent_or_write):
                receipt = r.finish(path, path / 'run', builder.config(config, diagnostic), 12.)
            output = (path / ('diagnostic-k1n.csv' if diagnostic else 'submission.csv')).read_bytes()
            assert receipt['alpha_vector'] == config['alpha_vector']
            assert receipt['alpha_targets'] == r.util.LABELS
            assert receipt['parent_mode'] == 'INF3'
            assert receipt['status'] == ('COMPLETE_SHARED_K1N' if fault == 'ok' else 'DEGRADED_PARENT_ONLY')
            if fault == 'ok':
                np.testing.assert_array_equal(r.util.table(output)[1], expected)
            else:
                assert output == parent_bytes
            if diagnostic:
                assert not (path / 'submission.csv').exists()
                assert receipt['never_submit'] and receipt['submission_sha256'] is None
            result['cases'].append(dict(diagnostic=diagnostic, fault=fault, n=size,
                                        status=receipt['status'], output_sha256=builder.digest(output),
                                        runtime_sha256=registration['sources']['tools/k1sr_runtime_v7.py'],
                                        kernel_sha256=registration['kernel_sha256']))
    pins = json.loads((OUT / 'frozen-stage-pins.json').read_text(encoding='utf-8-sig'))
    assert all(builder.util.sha(Path(p)) == sha for p, sha in pins.items())
    result.update(status='PASS_REAL_PREDICTION_CPU_REPLAY', frozen_files_unchanged=len(pins))
    (OUT / 'cpu-smoke-result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
