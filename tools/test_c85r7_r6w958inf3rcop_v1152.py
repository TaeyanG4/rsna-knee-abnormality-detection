"""Focused CPU tests for H217 (tools/c85r7_r6w958inf3rcop_v1152.py). No GPU/model/network."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools import c85r7_r6w958inf3rcop_v1152 as h217

ROOT = h217.ROOT
ASSETS = ROOT / '.local/scratch/slot-b/k2-release/fix1/assets-v1'
IDS = ['1.2.3.1', '1.2.3.2', '1.2.3.3']


@pytest.fixture(scope='module')
def built():
    return h217.package()


def load_overlay(tmp_path, files):
    path = tmp_path / 'h217_overlay.py'
    path.write_bytes(files['c85r7_rco_overlay_v1.py'])
    spec = importlib.util.spec_from_file_location('h217_overlay_' + tmp_path.name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scope(built):
    nb, meta, proof, old, new = built
    assert set(new) == {'parent-run.py', 'c85r7_rco_overlay_v1.py', h217.NEW_WORKER_NAME}
    assert new['parent-run.py'] == old['parent-run.py']
    assert hashlib.sha256(new[h217.NEW_WORKER_NAME]).hexdigest() == h217.NEW_WORKER_SHA
    # Reversing every registered edit reproduces the submitted H208 overlay exactly.
    src = new['c85r7_rco_overlay_v1.py'].decode()
    for before, after in reversed(h217.edits(old['c85r7_rco_overlay_v1.py'])):
        assert src.count(after) == 1
        src = src.replace(after, before)
    assert src.encode() == old['c85r7_rco_overlay_v1.py']
    h208_meta = json.loads((h217.H208_STAGE / 'kernel/kernel-metadata.json').read_bytes())
    assert meta['dataset_sources'] == h208_meta['dataset_sources']
    assert meta['id'] == h217.SLUG and meta['enable_internet'] is False and meta['is_private'] is True
    assert {k: v for k, v in meta.items() if k not in ('id', 'title')} == \
           {k: v for k, v in h208_meta.items() if k not in ('id', 'title')}


def setup_run(tmp_path, mod, files, worker_mode, calls):
    import numpy as np
    runtime, output, data = tmp_path / 'rt', tmp_path / 'out', tmp_path / 'data'
    for d in (runtime, output, data):
        d.mkdir()
    for name, payload in files.items():
        (runtime / name).write_bytes(payload)
    header = 'StudyInstanceUID,' + ','.join('"%s"' % x if "'" in x else x for x in mod.LABELS)
    template = data / 'sample_submission.csv'
    import csv
    import io
    def csv_bytes(values):
        buf = io.StringIO(newline='')
        w = csv.writer(buf, lineterminator='\n')
        w.writerow([mod.UID] + mod.LABELS)
        for uid, row in zip(IDS, values):
            w.writerow([uid] + [format(float(v), '.9g') for v in row])
        return buf.getvalue().encode()
    template.write_bytes(csv_bytes(np.full((3, 12), .5)))
    rng = np.random.default_rng(0)
    parent = csv_bytes(rng.random((3, 12)))
    rco = csv_bytes(rng.random((3, 12)))
    manifest = json.loads((ASSETS / 'asset-manifest.json').read_bytes())

    def runner(command, *, env=None, cwd=None, timeout=None):
        calls.append(dict(command=command, timeout=timeout))
        if any(str(c).endswith('parent-run.py') for c in command):
            (output / 'submission.csv').write_bytes(parent)
            hook = output / 'c85r7-run/codecs/hooks'
            hook.mkdir(parents=True)
            (output / 'c85r7-run/codecs/install.json').write_text(json.dumps(dict(status='ACTIVE', hook_dir=str(hook))))
            (output / 'c85r7-receipt.json').write_text(json.dumps(dict(status='COMPLETE_C85R7',
                version=mod.PARENT_VERSION, submission_sha256=hashlib.sha256(parent).hexdigest(),
                studies=3, findings=12, fallbacks=dict(a172_prediction=0, r_prediction=0, s_r7_prediction=0),
                codecs=dict(status='ACTIVE', s_r7_gate='ACTIVE'))))
        elif '-m' in command and 'pip' in command:
            pass
        elif any(str(c).endswith(h217.NEW_WORKER_NAME) for c in command):
            if worker_mode == 'timeout':
                raise subprocess.TimeoutExpired(command, timeout)
            out = Path(command[command.index('--output') + 1])
            out.write_bytes(rco)
            version = '1.0.1-rcof0' if worker_mode == 'old_version' else h217.NEW_WORKER_VERSION
            out.with_name('worker-receipt.json').write_text(json.dumps(dict(status='COMPLETE', version=version,
                kernel_id='taeyangg4/knee-l1-rcof0-v1', contract=manifest['contract'],
                manifest_sha256=mod.MANIFEST_SHA, template_sha256=mod.sha(template),
                output_sha256=hashlib.sha256(rco).hexdigest(), study_count=3, prediction_count=3,
                fast_path=False, placeholder=False, fallbacks=dict(prediction=0), codecs=dict(status='ACTIVE'),
                studies=[dict(index=i, uid_sha256=hashlib.sha256(u.encode()).hexdigest(), error=None,
                              prediction_fallback=0) for i, u in enumerate(IDS)])))
        else:
            raise AssertionError('unexpected command')
    return runtime, output, data, runner, parent, rco


@pytest.mark.parametrize('mode', ['ok', 'timeout', 'old_version', 'deadline'])
def test_simulated_run(tmp_path, built, mode):
    *_, files = built
    mod = load_overlay(tmp_path, files)
    assert mod.MODEL_SHA == json.loads((ASSETS / 'asset-manifest.json').read_bytes())['contract']['model_sha256']
    if mode == 'deadline':
        mod.RCO_DEADLINE_SECONDS = 100
    calls = []
    runtime, output, data, runner, parent, rco = setup_run(tmp_path, mod, files, mode, calls)
    record = mod.run(runtime, output, runner=runner, assets=ASSETS, data=data)
    final = (output / 'submission.csv').read_bytes()
    worker_calls = [c for c in calls if any(str(x).endswith(h217.NEW_WORKER_NAME) for x in c['command'])]
    if mode == 'ok':
        assert record['status'] == 'COMPLETE_K3'
        assert record['executed_members'] == ['INF3', 'RCO_K1_seed0_f0']
        assert final == mod.blend(parent, rco, IDS)
        assert worker_calls and 0 < worker_calls[0]['timeout'] <= h217.DEADLINE_SECONDS
    else:
        assert record['status'] == 'DEGRADED_PARENT_ONLY' and final == parent
        assert json.loads((output / 'k3-receipt.json').read_bytes())['status'] == 'DEGRADED_PARENT_ONLY'
        if mode == 'deadline':
            assert not worker_calls and 'insufficient time' in record['error']


def test_run_child_real_timeout(tmp_path, built):
    *_, files = built
    mod = load_overlay(tmp_path, files)
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        mod.run_child([sys.executable, '-c', 'import time; time.sleep(120)'], timeout=2)
    assert time.monotonic() - t0 < 30
