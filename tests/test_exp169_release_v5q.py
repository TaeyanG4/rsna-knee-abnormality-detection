"""CPU gates for the local EXP169 v5q overlay.

v1.2.0 (astra, 2026-10-11): unlimited runtime including the authenticated A172
derivative; original parent parity/memory checks; A172 before S6 wheel install.

v1.1.0 (Claude opus for B, 2026-10-11): parent-sequence blend with float32 trio CSV
boundary, S6-neutralisation regression vs the scored A172 sequence (N=1/2/3/1300,
random / quantized ties / constant columns), mathematical-tie oracle, output lifecycle
(no deletion after commit, BrokenPipe-safe logging, failure injection before commit),
process-group timeout, anchor-time projection and anatomy-fallback logging.
"""
from __future__ import annotations

import builtins
import ast
import csv
import importlib.util
import inspect
import io
import json
import math
import runpy
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import exp169_release_v2 as a172
from tools import exp169_release_v5q as v5q
from tools import exp189_s6_port as s6

ROOT = Path(__file__).resolve().parents[1]
PARENT_ASSETS = ROOT / "artifacts/releases/EXP155_S42v1_1_stage_v12/assets"


def _rank(values):
    """A172 average-tie rank: (one-based average rank - 1)/max(1, N - 1)."""
    a = np.asarray(values, dtype=np.float64)
    out = np.empty_like(a)
    for col in range(a.shape[1]):
        for value in np.unique(a[:, col]):
            positions = np.flatnonzero(a[:, col] == value)
            lower = np.count_nonzero(a[:, col] < value)
            out[positions, col] = (lower + (len(positions) + 1) / 2 - 1) / max(1, len(a) - 1)
    return out


def _public_pct(values):
    """Public cell 6 uses the one-based average rank divided by N."""
    a = np.asarray(values, dtype=np.float64)
    out = np.empty_like(a)
    for col in range(a.shape[1]):
        for value in np.unique(a[:, col]):
            positions = np.flatnonzero(a[:, col] == value)
            lower = np.count_nonzero(a[:, col] < value)
            out[positions, col] = (lower + (len(positions) + 1) / 2) / len(a)
    return out


def _members():
    ids = [f"study-{i}" for i in range(5)]
    def columns(values):
        return np.column_stack([np.roll(values, col % 5) for col in range(12)])
    base = columns(np.array([.11, .11, .31, .86, .48]))
    s6_values = columns(np.array([.72, .25, .84, .25, .13]))
    e143 = columns(np.array([.4, .8, .1, .6, .2]))
    e155 = columns(np.array([.9, .5, .3, .1, .7]))
    q = columns(np.array([.3, .1, .7, .9, .5]))
    return ids, base, s6_values, e143, e155, q


def _f32(trio):
    """The float32 boundary as seen by a rank: float32 values in float64."""
    return np.asarray(trio, dtype=np.float32).astype(np.float64)


@pytest.fixture(scope="module")
def parent155():
    """The staged, scored EXP155 v1.2 release module (its own pinned EXP152 copy)."""
    spec = importlib.util.spec_from_file_location("parent_exp155_v12", PARENT_ASSETS / "exp155_own0949_oai_release_v12.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parent_final(parent155, ids, r, e143, e155, q):
    """A172 as scored: exp155 output_values([EXP155, R, EXP143], 'blend3') -> public
    serving float32 DataFrame.to_csv -> base.validate_csv float parse ->
    exp169_release_v2.final_values(values, Q, 'A172')."""
    trio = parent155.output_values([e155, r, e143], "blend3", None)
    sub = pd.DataFrame(trio.astype(np.float32), columns=list(v5q.LABELS))
    sub.insert(0, v5q.UID, ids)
    rows = list(csv.DictReader(io.StringIO(sub.to_csv(index=False))))
    values = np.asarray([[float(row[t]) for t in v5q.LABELS] for row in rows], np.float64)
    return a172.final_values(values, q, "A172")


def _inputs(kind, n, seed):
    rng = np.random.default_rng(seed)

    def one():
        if kind == "random":
            return rng.random((n, 12)).astype(np.float32)
        if kind == "quantized":
            return (np.round(rng.random((n, 12)) * 20) / 20).astype(np.float32)
        a = rng.random((n, 12)).astype(np.float32)
        a[:, ::2] = .25                       # alternating constant columns
        return a
    return one(), one(), one(), one()


def _partner(rng, n):
    return {"l2": rng.random((10, n, 12)), "dino": rng.random((5, n, 12)), "convnext": rng.random((5, n, 12))}


# ---------------------------------------------------------------- formulas

def test_rank_references_distinguish_public_and_a172_with_ties_and_columns():
    _, r, *_ = _members()
    assert r[0, 0] == r[1, 0] and r[0, 0] != r[0, 1]
    assert _rank(r)[0, 0] == .125
    assert _public_pct(r)[0, 0] == .3
    np.testing.assert_array_equal(_rank(r)[:, 1], np.roll(_rank(r)[:, 0], 1))
    np.testing.assert_array_equal(_rank(r[:1]), np.zeros((1, 12)))      # N=1 denominator guard


def test_a5_uses_public_p954_in_r_slot_through_parent_sequence():
    ids, r, s, e143, e155, q = _members()
    actual = v5q.blend_arrays("A5-172", ids, r, s, e143, e155, q, s6_port=s6)
    p954 = (.70 * _public_pct(r) + .30 * _public_pct(s)).astype(np.float32)
    trio = (_rank(e155) + _rank(p954) + _rank(e143)) / 3
    expected = .85 * _rank(_f32(trio)) + .15 * _rank(q)
    np.testing.assert_array_equal(actual, expected)


def test_a5q_four_members_through_parent_sequence_and_protocol_order():
    ids, r, s, e143, e155, q = _members()
    actual = v5q.blend_arrays("A5q-172", ids, r, s, e143, e155, q)
    trio = (_rank(e155) + _rank(r) + _rank(e143) + _rank(s)) / 4
    np.testing.assert_array_equal(actual, .85 * _rank(_f32(trio)) + .15 * _rank(q))
    listed = (_rank(r) + _rank(s) + _rank(e143) + _rank(e155)) / 4      # PROTOCOL listing order
    np.testing.assert_array_equal(actual, .85 * _rank(_f32(listed)) + .15 * _rank(q))


def test_trio_output_values_equals_scored_exp155_output_values(parent155):
    rng = np.random.default_rng(7)
    arms = [rng.random((40, 12)).astype(np.float32) for _ in range(3)]
    np.testing.assert_array_equal(v5q.trio_output_values(arms), parent155.output_values(arms, "blend3", None))


@pytest.mark.parametrize("n", [1, 2, 3, 1300])
@pytest.mark.parametrize("kind", ["random", "quantized", "constant"])
@pytest.mark.parametrize("candidate", sorted(v5q.CANDIDATES))
def test_neutral_s6_reproduces_scored_a172_exactly(parent155, candidate, kind, n):
    ids = [f"s{i:05d}" for i in range(n)]
    r, e143, e155, raw_q = _inputs(kind, n, 1000 * n + len(kind))
    q = a172.rank(raw_q)
    expected = _parent_final(parent155, ids, r, e143, e155, q)
    neutral = np.full((n, 12), .5)
    actual = v5q.blend_arrays(candidate, ids, r, neutral, e143, e155, q, a172=a172, s6_port=s6)
    assert np.array_equal(actual, expected)
    serialised = [[format(float(x), ".10g") for x in row] for row in actual]
    assert serialised == [[format(float(x), ".10g") for x in row] for row in expected]


@pytest.mark.parametrize("kind", ["random", "quantized", "constant"])
@pytest.mark.parametrize("candidate", sorted(v5q.CANDIDATES))
def test_real_s6_trio_ties_match_exact_integer_oracle_at_1300(candidate, kind):
    """Mathematically equal rank sums must tie exactly after the float32 boundary."""
    n = 1300
    ids = [f"s{i:05d}" for i in range(n)]
    r, e143, e155, s = _inputs(kind, n, 77 + len(kind))
    members = v5q.trio_members(candidate, ids, r, s, e143, e155, s6_port=s6)
    trio = v5q.trio_values(candidate, ids, r, s, e143, e155, s6_port=s6)
    numerators = []
    for member in members:
        doubled = np.empty((n, 12), np.int64)
        for j in range(12):
            column = member[:, j]
            order = np.argsort(column, kind="stable")
            at = 0
            while at < n:
                end = at + 1
                while end < n and column[order[end]] == column[order[at]]:
                    end += 1
                doubled[order[at:end], j] = at + end - 1
                at = end
        numerators.append(doubled)
    exact = sum(numerators)                      # exact integer rank sums (2x average 0-based ranks)
    np.testing.assert_array_equal(_rank(trio), _rank(exact.astype(np.float64)))


def test_id_mismatch_fails_before_blend():
    ids, r, s, e143, e155, q = _members()
    with pytest.raises(ValueError, match="order mismatch"):
        v5q.validate_ids(ids, ids[::-1])
    rows_r = [{s6.UID: uid, **dict(zip(s6.LABELS, row))} for uid, row in zip(ids, r)]
    rows_s = [{s6.UID: uid, **dict(zip(s6.LABELS, row))} for uid, row in zip(ids[::-1], s)]
    with pytest.raises(ValueError, match="study order"):
        s6.p954(rows_r, rows_s)


def test_trio_round_trip_file_matches_memory_and_refuses_overwrite(tmp_path):
    ids, r, s, e143, e155, _ = _members()
    trio = v5q.trio_output_values([e155, r, e143, s])
    in_memory = v5q.trio_round_trip(ids, trio)
    on_disk = v5q.trio_round_trip(ids, trio, tmp_path / "trio.csv")
    np.testing.assert_array_equal(in_memory, on_disk)
    np.testing.assert_array_equal(_rank(in_memory), _rank(_f32(trio)))
    np.testing.assert_allclose(in_memory, trio, rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match="already exists"):
        v5q.trio_round_trip(ids, trio, tmp_path / "trio.csv")


def test_s6_provenance_fails_closed_on_wrong_manifest_hash(tmp_path):
    (tmp_path / "manifest.json").write_text('{"bundle_id":"wrong"}', encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        s6.write_provenance(provenance, bundle_dir=tmp_path)
    assert not provenance.exists()


def test_v5q_stage_fails_closed_on_s6_source_drift(monkeypatch):
    real_sha = v5q.sha

    def drifted(path):
        if str(path).replace("\\", "/").endswith("tools/exp189_s6_port.py"):
            return "0" * 64
        return real_sha(path)

    monkeypatch.setattr(v5q, "sha", drifted)
    with pytest.raises(ValueError, match="S6 source drift"):
        v5q.stage()


def test_timing_is_informational_and_rejects_only_invalid_inputs():
    report = v5q.timing_projection(100_000.0)
    assert report['projected_conservative_seconds'] > 100_000_000
    assert report['informational'] is True and 'guard_seconds' not in report
    assert report['projected_linear_seconds'] < report['projected_conservative_seconds']
    for rate in (float('nan'), float('inf'), 0, -1):
        with pytest.raises(ValueError, match='invalid S6 timing'):
            v5q.timing_projection(rate)
    service = {'service_seconds_per_study': 1.0 + .5 + 2.0 + .25}
    assert v5q.s6_service_rate(service) == 3.75 + v5q.SERVICE_ALLOWANCE_SECONDS


def test_a172_runtime_cell_exact_anchors_and_no_alarm():
    source = ''.join(json.loads((v5q.STAGE / 'kernel/release.ipynb').read_text())['cells'][4]['source'])
    derived = v5q.a172_runtime_source(source)
    ast.parse(derived)
    assert '_exp169_signal' not in derived and 'TimeoutError' not in derived
    assert derived.index('_e169.verify_assets(') < derived.index('_v5q.install_a172_runtime(')
    assert derived.index('_v5q.install_a172_runtime(') < derived.index('_e169.pilot(')
    assert 'a172-runtime-functions.json' in derived
    with pytest.raises(ValueError, match='A172 cell drift'):
        v5q.a172_runtime_source(source + '\n')


def _unlimited_a172():
    isolated = types.SimpleNamespace(**a172.__dict__)
    record = v5q.install_a172_runtime(isolated)
    return isolated, record


def test_a172_runtime_derivative_is_private_authenticated_and_has_no_timers():
    original_partner, original_serve = a172.partner_scores, a172.serve
    original_hashes = (v5q.sha(a172.__file__), v5q.sha(a172.base.__file__))
    runtime, record = _unlimited_a172()
    private = runtime.serve.__globals__['base']
    assert private is not a172.base
    assert record['timing_policy'] == 'unlimited'
    assert set(record['functions']) == {'partner_scores', 'pilot', 'serve'}
    for name, digest in record['functions'].items():
        source = inspect.getsource(getattr(private, name))
        assert v5q.hashlib.sha256(source.encode()).hexdigest() == digest
        tree = ast.parse(source)
        assert not any(isinstance(n, ast.keyword) and n.arg == 'timeout' for n in ast.walk(tree))
        assert 'deadline exceeded' not in source.lower()
        assert '< deadline' not in source and 'LIMIT_SECONDS' not in source
    assert private._load_models is a172.base._load_models
    assert private.aggregate is a172.base.aggregate
    assert private.final_values is a172.base.final_values
    assert runtime.serve.__kwdefaults__ == original_serve.__kwdefaults__
    assert a172.partner_scores is original_partner and a172.serve is original_serve
    assert original_hashes == (v5q.sha(a172.__file__), v5q.sha(a172.base.__file__))
    with pytest.raises(ValueError, match='anchor changed'):
        v5q.a172_function_source('pilot', inspect.getsource(a172.base.pilot).replace('started + 3600', 'started + 1'))
    changed = types.SimpleNamespace(**a172.__dict__)
    changed.__file__ = v5q.__file__
    with pytest.raises(ValueError, match='runtime source drift'):
        v5q.install_a172_runtime(changed)


def _a172_cpu_fixture(tmp_path):
    helper = runpy.run_path(str(ROOT / 'tests/test_exp169_release_v2.py'))
    return helper['_build_serving'](tmp_path, 'A172')


def test_cpu_blend_replay_archived_arrays_and_fresh_output(tmp_path):
    # This unit-test input is an archived prediction array, not an S6 model run.
    source = v5q.STAGE / 'kernel-output/submission.csv'
    out = tmp_path / 'replay'
    receipt = v5q.cpu_replay(source, out)
    assert receipt['status'] == 'PASS_CPU_BLEND_REPLAY'
    assert all(item['neutral_parent_exact'] and item['rows'] == 3
               for item in receipt['candidates'].values())
    for candidate, folder in v5q.CANDIDATES.items():
        assert v5q.sha(out / (folder + '.csv')) == receipt['candidates'][candidate]['submission_sha256']
    with pytest.raises(ValueError, match='fresh CPU replay'):
        v5q.cpu_replay(source, out)


def test_cpu_blend_replay_rejects_id_order_mismatch(tmp_path):
    ids, values = v5q._read_csv(v5q.STAGE / 'kernel-output/submission.csv')
    source = tmp_path / 'reordered.csv'
    v5q._write_csv(source, ids[::-1], values[::-1])
    with pytest.raises(ValueError, match='study ID/order mismatch'):
        v5q.cpu_replay(source, tmp_path / 'replay')
    assert not (tmp_path / 'replay').exists()


def test_a172_unlimited_serve_preserves_exact_parent_output_after_arbitrary_elapsed_time(tmp_path):
    s = _a172_cpu_fixture(tmp_path)
    baseline = a172.serve(s.work, s.anchor, s.output, s.competition,
                          clock=lambda: 100., partner=s.partner)
    expected = (s.output / 'submission.csv').read_bytes()
    runtime, _ = _unlimited_a172()
    pilot_path = s.work / 'partner-pilot.json'
    measured = json.loads(pilot_path.read_text())
    measured.update(projected_full_seconds=1e12, elapsed_seconds=1e12)
    pilot_path.write_text(json.dumps(measured))
    ticks = iter(float(i) * 1e12 for i in range(100))
    observed_deadlines = []
    def partner(*args):
        observed_deadlines.append(args[-1])
        return s.partner(*args)
    derived_out = tmp_path / 'unlimited-output'
    result = runtime.serve(s.work, s.anchor, derived_out, s.competition,
                           clock=lambda: next(ticks), partner=partner)
    assert result['status'] == baseline['status'] == 'PASS_BUNDLE'
    assert result['elapsed_seconds'] > 1e12 and observed_deadlines == [None]
    assert (derived_out / 'submission.csv').read_bytes() == expected
    assert result['bundle_rank_sha256'] == baseline['bundle_rank_sha256']
    assert result['family_rank_sha256'] == baseline['family_rank_sha256']


def test_a172_unlimited_pilot_keeps_registered_selection_and_no_deadline(tmp_path):
    s = _a172_cpu_fixture(tmp_path)
    runtime, _ = _unlimited_a172()
    with (s.competition / 'test_series.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([v5q.UID, 'SeriesInstanceUID'])
        writer.writerows((uid, f'series-{i}') for i, uid in enumerate(s.ids))
    calls = []
    def partner(work, competition, ids, manifest, deadline):
        calls.append((ids, deadline))
        ix = [s.ids.index(uid) for uid in ids]
        return ({name: array[:, ix] for name, array in s.scores.items()},
                {'gpu_peak_bytes': [1024, 1024], 'ram': {'maxrss_kib': 1024},
                 'p95_study_seconds': 1e12, 'p95_decode_study_seconds': 1e12})
    runtime.pilot.__globals__['partner_scores'] = partner
    result = runtime.pilot(s.work, s.competition, tmp_path / 'pilot', studies=len(s.ids))
    assert result['status'] == 'PASS' and result['studies'] == len(s.ids)
    assert len(calls) == 1 and set(calls[0][0]) == set(s.ids) and calls[0][1] is None
    assert result['manifest_sha256'] == v5q.sha(s.work / 'release-manifest.json')


@pytest.mark.parametrize('fault', ['ram', 'vram', 'pilot_coverage', 'pilot_hash', 'score_parity'])
def test_a172_unlimited_serve_keeps_non_timing_checks(tmp_path, fault):
    s = _a172_cpu_fixture(tmp_path)
    runtime, _ = _unlimited_a172()
    pilot_path = s.work / 'partner-pilot.json'
    measured = json.loads(pilot_path.read_text())
    if fault == 'ram':
        measured['peak_ram_bytes'] = 28 * 2**30
    elif fault == 'vram':
        measured['peak_vram_bytes'] = 16 * 2**30
    elif fault == 'pilot_coverage':
        measured['studies'] = 0
    elif fault == 'pilot_hash':
        measured['scores_sha256'] = '0' * 64
    pilot_path.write_text(json.dumps(measured))
    def partner(*args):
        scores, metrics = s.partner(*args)
        if fault == 'score_parity':
            scores['l2'][0, 0, 0] += .01
        return scores, metrics
    with pytest.raises(ValueError):
        runtime.serve(s.work, s.anchor, s.output, s.competition,
                      clock=lambda: 1e12, partner=partner)
    assert not (s.output / 'submission.csv').exists()
    assert json.loads((s.output / 'exp169-executed-candidate.json').read_text())['status'] == 'FAILED_BUNDLE'


def test_real_json_worker_exit_receipt_checks_values_and_exact_roster():
    receipt = {"status": "PASS", "rows": 3, "scored": 3, "fallback_studies": 0,
               "gpus": ["Tesla T4", "Tesla T4"], "worker_exit_codes": {"0": 0, "1": 0}}
    v5q.validate_s6_receipt(receipt, 3)
    for codes in ({"0": 1, "1": 0}, {"0": 0}, {"0": 0, "1": 0, "2": 0}):
        with pytest.raises(ValueError, match="worker exit-code"):
            v5q.validate_s6_receipt({**receipt, "worker_exit_codes": codes}, 3)


def test_output_schema_and_order(tmp_path):
    ids, r, *_ = _members()
    path = tmp_path / "submission.csv"
    v5q._write_csv(path, ids, r)
    got_ids, got = v5q._read_csv(path)
    assert got_ids == ids
    assert got.shape == (len(ids), 12)
    with path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert header == list(v5q.TARGETS)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([v5q.UID, *reversed(v5q.LABELS)])
        writer.writerow([ids[0], *r[0]])
    with pytest.raises(ValueError, match="schema/order"):
        v5q._read_csv(path)


# ---------------------------------------------------------------- run_after_a172 harness

class Kaggle:
    """Minimal /kaggle layout with completed A172 outputs and a fake S6 runtime."""

    def __init__(self, tmp_path, monkeypatch, *, anchor_s=.5, anchor_methods=None, n=6):
        self.work = tmp_path / "working"
        self.comp = tmp_path / "comp"
        self.n = n
        rng = np.random.default_rng(11)
        self.ids = [f"1.2.{i}" for i in range(n)]
        for folder in (self.work / "exp169-trio", self.work / "exp169-partner", self.comp / "test_series"):
            folder.mkdir(parents=True)
        for uid in self.ids:
            (self.comp / "test_series" / uid).mkdir()
        v5q._write_csv(self.comp / "sample_submission.csv", self.ids, np.full((n, 12), .5))
        self.members = {k: rng.random((n, 12)).astype(np.float32) for k in ("own", "public", "exp143")}
        np.savez(self.work / "exp169-trio/raw.npz", ids=np.asarray(self.ids), targets=np.asarray(v5q.LABELS),
                 **self.members)
        self.partner = _partner(rng, n)
        np.savez(self.work / "partner-raw.npz", ids=np.asarray(self.ids), **self.partner)
        (self.work / "exp169-partner/release-manifest.json").write_text("{}", encoding="utf-8")
        v5q._write_csv(self.work / "submission.csv", self.ids, rng.random((n, 12)))
        self.a172_sha = v5q.sha(self.work / "submission.csv")
        (self.work / "exp169-executed-candidate.json").write_text(json.dumps(
            {"status": "PASS_BUNDLE", "candidate": "A172", "submission_sha256": self.a172_sha}), encoding="utf-8")
        self.s6_values = rng.random((n, 12)).round(6)
        self.anchor_s = anchor_s
        self.anchor_methods = anchor_methods or {"skm_tea_joint": n}
        fake_a172 = types.ModuleType("exp169_release_v2")
        fake_a172.__file__ = a172.__file__
        fake_a172.verify_assets = lambda work, manifest: True
        fake_a172.aggregate = a172.aggregate
        fake_a172.final_values = a172.final_values
        fake_s6 = types.ModuleType("exp189_s6_port")
        fake_s6.__file__ = s6.__file__
        for name in ("UID", "LABELS", "DATASET_SLUG", "VERSION", "p954", "service_components", "anatomy_fallbacks"):
            setattr(fake_s6, name, getattr(s6, name))
        fake_s6.verify_bundle = fake_s6.verify_source = lambda *a, **k: None
        fake_s6.prepare_code = self.prepare_code
        fake_s6.infer_studies = self.infer_studies
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
        monkeypatch.setitem(sys.modules, "exp169_release_v2", fake_a172)
        monkeypatch.setitem(sys.modules, "exp189_s6_port", fake_s6)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(v5q, "WORK", self.work)
        monkeypatch.setattr(v5q, "_mount", lambda slug, required: self.comp)
        monkeypatch.setattr(v5q, "S6_PORT_SHA", v5q.sha(s6.__file__))
        self.prepare_calls = []

    def prepare_code(self, src, destination, install_wheels=True, *, code_tree_sha256=None):
        self.prepare_calls.append((install_wheels, code_tree_sha256))
        return Path(destination) / "mainline_code", Path(src) / "anatomy.pt"

    def infer_studies(self, dirs, *, bundle_dir, code_dir, anatomy_weights, output_dir, workers, amp,
                      max_fallback_fraction):
        out = Path(output_dir)
        (out / "controller").mkdir(parents=True)
        ids = [Path(d).name for d in dirs]
        with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([s6.UID, *s6.LABELS])
            for uid, row in zip(ids, self.s6_values):
                writer.writerow([uid, *["%.6f" % x for x in row]])
        half = self.n // 2
        stats = lambda rows: {"decode_s": .2 * rows, "render_s": .1 * rows, "anchor_s": self.anchor_s * rows,
                              "gpu_s": .3 * rows, "rows": rows}
        controller = {"status": "PASS", "rows": self.n, "scored": self.n, "fallback_studies": 0,
                      "gpus": ["Tesla T4", "Tesla T4"], "worker_exit_codes": {"0": 0, "1": 0},
                      "workers": [{"status": "PASS", "stats": stats(half), "anchor_methods": self.anchor_methods},
                                  {"status": "PASS", "stats": stats(self.n - half), "anchor_methods": {}}]}
        (out / "controller/submission_receipt.json").write_text(json.dumps(controller), encoding="utf-8")
        (out / "provenance.json").write_text(json.dumps({"code_tree_pinned_in_source": True}), encoding="utf-8")
        return s6.read_predictions(out / "predictions.csv", ids)

    def expected(self, candidate):
        q, _ = a172.aggregate(self.partner)
        return v5q.blend_arrays(candidate, self.ids, self.members["public"], self.s6_values,
                                self.members["exp143"], self.members["own"], q, a172=a172, s6_port=s6)

    def assert_failed_closed(self):
        assert not (self.work / "submission.csv").exists()
        assert not (self.work / "v5q-submission.tmp").exists()
        assert v5q.sha(self.work / "a172-submission.csv") == self.a172_sha
        receipt = json.loads((self.work / "v5q-receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "FAILED_V5Q"
        assert not v5q.committed(self.work)


@pytest.mark.parametrize("candidate", sorted(v5q.CANDIDATES))
def test_run_after_a172_commits_last_with_complete_receipt(tmp_path, monkeypatch, candidate):
    k = Kaggle(tmp_path, monkeypatch)
    result = v5q.run_after_a172(candidate, started_epoch=time.time())
    assert result["status"] == v5q.PASS_STATUS and v5q.committed(k.work)
    assert v5q.sha(k.work / "submission.csv") == result["submission_sha256"]
    assert json.loads((k.work / "v5q-receipt.json").read_text()) == json.loads(json.dumps(result))
    assert v5q.sha(k.work / "a172-submission.csv") == k.a172_sha == result["a172_submission_sha256"]
    assert not (k.work / "v5q-submission.tmp").exists() and not (k.work / "v5q-receipt.tmp").exists()
    _, written = v5q._read_csv(k.work / "submission.csv")
    expected = k.expected(candidate)
    assert np.array_equal(written, [[float(format(float(v), ".10g")) for v in row] for row in expected])
    assert result["trio_members"] == list(v5q.TRIO_MEMBERS[candidate])
    assert k.prepare_calls == [(True, None)]   # source pin, no time limit


@pytest.mark.parametrize("stage_name", ["hash", "receipt", "replace"])
def test_late_failures_before_commit_never_leave_a_submission(tmp_path, monkeypatch, stage_name):
    k = Kaggle(tmp_path, monkeypatch)
    real_sha, real_atomic, real_replace = v5q.sha, v5q._atomic_write_text, v5q.os.replace

    if stage_name == "hash":
        monkeypatch.setattr(v5q, "sha", lambda p: (_ for _ in ()).throw(OSError("hash failed"))
                            if str(p).endswith("v5q-submission.tmp") else real_sha(p))
    elif stage_name == "receipt":
        def failing_atomic(path, text):
            if json.loads(text).get("status") == v5q.PASS_STATUS:
                raise OSError("disk full")
            return real_atomic(path, text)
        monkeypatch.setattr(v5q, "_atomic_write_text", failing_atomic)
    else:
        def failing_replace(src, dst):
            if Path(dst).name == "submission.csv" and Path(src).name == "v5q-submission.tmp":
                raise OSError("replace failed")
            return real_replace(src, dst)
        monkeypatch.setattr(v5q.os, "replace", failing_replace)
    with pytest.raises((OSError, ValueError)):
        v5q.run_after_a172("A5-172", started_epoch=time.time())
    monkeypatch.undo()
    k.assert_failed_closed()


def test_interrupt_after_commit_and_wrapper_cleanup_never_delete(tmp_path, monkeypatch):
    k = Kaggle(tmp_path, monkeypatch)
    real_replace = v5q.os.replace

    def replace_then_interrupt(src, dst):
        real_replace(src, dst)
        if Path(dst).name == "submission.csv":
            raise KeyboardInterrupt("arrives just after the commit")
    monkeypatch.setattr(v5q.os, "replace", replace_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        v5q.run_after_a172("A5q-172", started_epoch=time.time())
    monkeypatch.setattr(v5q.os, "replace", real_replace)
    assert v5q.committed(k.work)
    digest = v5q.sha(k.work / "submission.csv")
    assert v5q.cleanup_uncommitted(candidate="A5q-172", error=RuntimeError("wrapper")) is False
    assert v5q.sha(k.work / "submission.csv") == digest
    assert json.loads((k.work / "v5q-receipt.json").read_text())["status"] == v5q.PASS_STATUS


def test_hidden_run_records_large_projection_and_anchor_fallbacks(tmp_path, monkeypatch):
    k = Kaggle(tmp_path, monkeypatch, anchor_s=50.0, anchor_methods={"skm_tea_joint": 1, "slab_prior": 2})
    result = v5q.run_after_a172("A5-172", started_epoch=time.time())
    timing = result["timing_projection"]
    assert timing["service_per_study"]["anchor_s"] == pytest.approx(50.0)
    assert timing["s6_seconds_per_study"] == pytest.approx(.2 + .1 + 50.0 + .3 + v5q.SERVICE_ALLOWANCE_SECONDS)
    assert timing["projected_conservative_seconds"] > 24_300      # projection only, not a rejection
    assert result["status"] == v5q.PASS_STATUS and v5q.committed(k.work)
    assert result["s6_anatomy_anchors"]["fallback"] == k.n - 1


def test_ancient_started_epoch_and_late_clock_do_not_reject(tmp_path, monkeypatch):
    k = Kaggle(tmp_path, monkeypatch)
    original_blend = v5q.blend_arrays
    def late_blend(*args, **kwargs):
        out = original_blend(*args, **kwargs)
        monkeypatch.setattr(time, 'time', lambda: 4e12)
        return out
    monkeypatch.setattr(v5q, 'blend_arrays', late_blend)
    result = v5q.run_after_a172('A5-172', started_epoch=1.0)
    assert result['status'] == v5q.PASS_STATUS and v5q.committed(k.work)
    assert 'deadline_epoch' not in result


# ---------------------------------------------------------------- S6 preflight (before A172)

def _fake_code_mount(root: Path) -> dict:
    training = root / "cloud_code" / "training"
    training.mkdir(parents=True)
    for rel in s6.DIRECT_CODE_MODULES + ("training/common.py",):
        (root / "cloud_code" / rel).write_text(f"# {rel}\n", encoding="utf-8")
    (root / "anatomy.pt").write_bytes(b"anatomy")
    (root / "dependencies/wheels").mkdir(parents=True)
    for name in s6.WHEEL_HASHES:
        (root / "dependencies/wheels" / name).write_bytes(name.encode())
    return {"code": {n: s6.sha256(training / n) for n in s6.CODE_HASHES},
            "anatomy": s6.sha256(root / "anatomy.pt"),
            "wheels": {n: s6.sha256(root / "dependencies/wheels" / n) for n in s6.WHEEL_HASHES}}


def _preflight_env(tmp_path, monkeypatch):
    mount, work = tmp_path / "mount", tmp_path / "working"
    work.mkdir()
    hashes = _fake_code_mount(mount)
    monkeypatch.setitem(sys.modules, "exp189_s6_port", s6)
    monkeypatch.setattr(v5q, "_mount", lambda slug, required: mount)
    monkeypatch.setattr(v5q, "S6_PORT_SHA", v5q.sha(s6.__file__))
    monkeypatch.setattr(s6, "verify_bundle", lambda bundle: None)
    monkeypatch.setattr(s6, "CODE_HASHES", hashes["code"])
    monkeypatch.setattr(s6, "ANATOMY_SHA256", hashes["anatomy"])
    monkeypatch.setattr(s6, "WHEEL_HASHES", hashes["wheels"])
    monkeypatch.setattr(s6, "install_wheels", lambda *a, **k: pytest.fail("preflight must not install"))
    return mount, work


def test_preflight_fails_closed_before_a172_while_code_tree_unpinned(tmp_path, monkeypatch):
    mount, work = _preflight_env(tmp_path, monkeypatch)
    monkeypatch.setattr(s6, "CODE_TREE_SHA256", None)       # simulate the unpinned source state
    monkeypatch.setattr(s6, "CODE_TREE_FILES", None)
    with pytest.raises(ValueError, match="not populated; fail closed"):
        v5q.preflight_s6(started_epoch=time.time(), root=work)
    assert not (work / "s6-preflight").exists() and not (work / "v5q-preflight.json").exists()


def test_preflight_passes_with_matching_pins_without_installing(tmp_path, monkeypatch):
    mount, work = _preflight_env(tmp_path, monkeypatch)
    observed = s6.observe_code(mount, tmp_path / "observed.json", work_dir=tmp_path)
    monkeypatch.setattr(s6, "CODE_TREE_SHA256", observed["tree_sha256"])
    monkeypatch.setattr(s6, "CODE_TREE_FILES", observed["files"])
    record = v5q.preflight_s6(started_epoch=time.time(), root=work)
    assert record["status"] == "PASS_S6_PREFLIGHT" and not (work / "s6-preflight").exists()
    (mount / "cloud_code/training/anatomy.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="code tree differs"):
        v5q.preflight_s6(started_epoch=time.time(), root=work)


# ---------------------------------------------------------------- generated cells

def _staged_cells(folder="a5_172"):
    nb = json.loads((v5q.OUT / "kernels" / folder / "release.ipynb").read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"]]


def _wrapper_code(source):
    """Cell code without the embedded source/archive literals (their content is pinned by hash)."""
    import ast
    drop = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                                                 {"V5Q_SOURCE", "A172_SOURCE", "PORT_ARCHIVE_B64"}
                                                 for t in node.targets):
            drop.update(range(node.lineno, node.end_lineno + 1))
    return chr(10).join(line for i, line in enumerate(source.splitlines(), 1) if i not in drop)


def test_staged_cells_order_preflight_before_a172_and_logging_outside_cleanup():
    for folder, candidate in (("a5_172", "A5-172"), ("a5q_172", "A5q-172")):
        cells = _staged_cells(folder)
        assert len(cells) == 5
        assert "V5Q_STARTED_EPOCH = time.time()" in cells[1] and "import exp169_release_v5q as v5q" in cells[1]
        assert "import exp189_s6_port as s6" in cells[2] and "v5q.cleanup_uncommitted" in cells[2]
        wrappers = [_wrapper_code(cell) for cell in cells[1:]]
        assert wrappers[2].index("v5q.preflight_s6(") < wrappers[2].index("v5q.run_a172(")
        assert all("subprocess.run" not in code and "unlink" not in code for code in wrappers)
        assert "start_new_session" not in wrappers[2]           # the group handling lives in v5q.run_a172
        overlay = cells[4]
        assert f"v5q.run_after_a172({candidate!r}" in overlay
        assert overlay.rstrip().splitlines()[-1].startswith("v5q.log_result(V5Q_RESULT)")


def test_a172_finishes_before_s6_wheel_install():
    """The public parent must run with its original timm/pydicom versions."""
    source = inspect.getsource(v5q.run_after_a172)
    assert source.index("s6.prepare_code(") < source.index("s6.infer_studies(")
    assert "install_wheels=True" in source
    for folder in ("a5_172", "a5q_172"):
        cells = _staged_cells(folder)
        assert "v5q.run_a172(" in _wrapper_code(cells[3])
        assert "v5q.run_after_a172(" in _wrapper_code(cells[4])
        assert all("install_wheels=True" not in _wrapper_code(cell) for cell in cells[1:4])


def test_staged_overlay_cell_broken_pipe_after_success_keeps_submission(tmp_path, monkeypatch):
    work = tmp_path / "working"
    work.mkdir()
    monkeypatch.setattr(v5q, "WORK", work)

    def successful_run(candidate, *, started_epoch):
        (work / "v5q-submission.tmp").write_text("final\n", encoding="utf-8")
        result = {"status": v5q.PASS_STATUS, "submission_sha256": v5q.sha(work / "v5q-submission.tmp")}
        v5q._atomic_write_text(work / "v5q-receipt.json", json.dumps(result))
        v5q.os.replace(work / "v5q-submission.tmp", work / "submission.csv")
        return result

    def broken_print(*args, **kwargs):
        raise BrokenPipeError("stdout closed")

    monkeypatch.setattr(v5q, "run_after_a172", successful_run)
    monkeypatch.setattr(builtins, "print", broken_print)
    exec(compile(_staged_cells()[4], "<overlay>", "exec"), {"v5q": v5q, "V5Q_STARTED_EPOCH": time.time()})
    monkeypatch.setattr(builtins, "print", print)
    assert (work / "submission.csv").read_text(encoding="utf-8") == "final\n" and v5q.committed(work)


def test_staged_overlay_cell_failure_removes_a172_submission(tmp_path, monkeypatch):
    work = tmp_path / "working"
    work.mkdir()
    (work / "submission.csv").write_text("a172\n", encoding="utf-8")
    monkeypatch.setattr(v5q, "WORK", work)
    monkeypatch.setattr(v5q, "run_after_a172", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("S6 failed")))
    with pytest.raises(RuntimeError, match="S6 failed"):
        exec(compile(_staged_cells()[4], "<overlay>", "exec"), {"v5q": v5q, "V5Q_STARTED_EPOCH": time.time()})
    assert not (work / "submission.csv").exists()
    assert json.loads((work / "v5q-receipt.json").read_text())["status"] == "FAILED_V5Q"


# ---------------------------------------------------------------- A172 process group

class _FakePopen:
    instances = []
    interrupt = False

    def __init__(self, args, **kwargs):
        self.args, self.kwargs, self.pid, self.returncode = args, kwargs, 31337, None
        _FakePopen.instances.append(self)

    def wait(self):
        if self.returncode is None and self.interrupt:
            raise KeyboardInterrupt('explicit interruption')
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_a172_runs_in_new_session_without_timeout(monkeypatch):
    _FakePopen.instances.clear()
    monkeypatch.setattr(v5q.os, 'name', 'posix')
    monkeypatch.setattr(v5q.subprocess, 'Popen', _FakePopen)
    v5q.run_process_group(['python', 'a172.py'], label='A172 v2.0.2')
    assert _FakePopen.instances[0].kwargs.get('start_new_session') is True
    assert _FakePopen.instances[0].returncode == 0


def test_a172_explicit_interrupt_kills_process_group(monkeypatch):
    _FakePopen.instances.clear()
    killed = []
    monkeypatch.setattr(v5q.os, 'name', 'posix')
    monkeypatch.setattr(v5q.subprocess, 'Popen', _FakePopen)
    monkeypatch.setattr(_FakePopen, 'interrupt', True)
    monkeypatch.setattr(v5q.os, 'killpg', lambda pid, sig: killed.append((pid, sig)), raising=False)
    with pytest.raises(KeyboardInterrupt, match='explicit interruption'):
        v5q.run_process_group(['python', 'a172.py'], label='A172 v2.0.2')
    assert killed == [(31337, 9)]
    assert _FakePopen.instances[0].returncode == -9


def test_run_a172_materialises_source_and_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(v5q, 'a172_runtime_source', lambda source: source)
    with pytest.raises(RuntimeError, match='exit code 3'):
        v5q.run_a172('import sys; sys.exit(3)\n', started_epoch=1.0, root=tmp_path)
    assert (tmp_path / 'a172-v2.0.2-v5q-1.2.0.py').read_text(encoding='utf-8') == 'import sys; sys.exit(3)\n'
    assert json.loads((tmp_path / 'a172-runtime-adapter.json').read_text())['timing_policy'] == 'unlimited'


# ---------------------------------------------------------------- local visible comparator (FIX-Q1)

from tools import exp169_release_v5q_compare as compare  # noqa: E402


def _visible_run(tmp_path, candidate, *, conservative=40_000.0, observed=41_000.0, sibling_delta=0.0):
    """Fake visible outputs: the scored A172 visible run plus a synthetic S6 written exactly as
    run_after_a172 writes it; timing deliberately above the guard (3-study setup bias)."""
    import shutil
    ref = v5q.STAGE / "kernel-output"
    out, sib = tmp_path / "output", tmp_path / "sibling"
    (out / "exp169-trio").mkdir(parents=True)
    (out / "s6-port").mkdir()
    sib.mkdir()
    for name in ("exp169-executed-candidate.json", "partner-raw.npz"):
        shutil.copyfile(ref / name, out / name)
    for name in ("raw.npz", "inputs.json", "binding.json", "submission.csv"):
        shutil.copyfile(ref / "exp169-trio" / name, out / "exp169-trio" / name)
    shutil.copyfile(ref / "submission.csv", out / "a172-submission.csv")
    with np.load(out / "exp169-trio/raw.npz", allow_pickle=False) as anchor:
        ids = anchor["ids"].tolist()
        r, e143, e155 = anchor["public"].copy(), anchor["exp143"].copy(), anchor["own"].copy()
    s6_values = np.random.default_rng(5).random((len(ids), 12)).round(6)
    v5q._write_csv(out / "s6-port-predictions.csv", ids, s6_values)
    v5q._write_csv(sib / "ours_sub.csv", ids, np.clip(s6_values + sibling_delta, 0, 1))
    v5q._write_csv(sib / "exp189_port_predictions.csv", ids, s6_values)
    (sib / "exp189_parity_receipt.json").write_text(json.dumps(
        {"bad_cells": 0, "rows": len(ids), "gpu_names": ["Tesla T4", "Tesla T4"]}), encoding="utf-8")
    (out / "s6-port/provenance.json").write_text(json.dumps({
        "dataset_slug": s6.DATASET_SLUG, "dataset_version": "10-equivalent", "bundle_id": s6.BUNDLE_ID,
        "licence": "other", "readme_permission_line": s6.README_PERMISSION, "gold_in_training": 0,
        "manifest_sha256": s6.MANIFEST_SHA256, "code_tree_pinned_in_source": True,
        "code_tree_sha256": s6.CODE_TREE_SHA256}), encoding="utf-8")
    with np.load(out / "partner-raw.npz", allow_pickle=False) as partner:
        q, _ = a172.aggregate({key: partner[key] for key in ("l2", "dino", "convnext")})
    v5q.trio_values(candidate, ids, r, s6_values, e143, e155, s6_port=s6, trio_csv=out / "v5q-trio.csv")
    final = v5q.blend_arrays(candidate, ids, r, s6_values, e143, e155, q, a172=a172, s6_port=s6)
    v5q._write_csv(out / "submission.csv", ids, final)
    receipt = {"status": v5q.PASS_STATUS, "candidate": candidate, "generator_version": v5q.VERSION,
               "source_sha256": v5q.S6_PORT_SHA, "a172_source_sha256": v5q.A172_SOURCE_SHA,
               "a172_submission_sha256": v5q.sha(out / "a172-submission.csv"),
               "s6_raw_sha256": v5q.sha(out / "s6-port-predictions.csv"),
               "submission_sha256": v5q.sha(out / "submission.csv"),
               "trio_csv_sha256": v5q.sha(out / "v5q-trio.csv"),
               "s6_provenance_sha256": v5q.sha(out / "s6-port/provenance.json"),
               "s6_anatomy_anchors": {"scored": len(ids), "model_anchor": len(ids), "fallback": 0, "methods": {}},
               "timing_projection": {"projected_conservative_seconds": conservative,
                                     "projected_from_observed_seconds": observed, "s6_seconds_per_study": 20.0,
                                     "service_per_study": {"decode_s": 12.0, "render_s": 1.0, "anchor_s": 4.0,
                                                           "gpu_s": 3.0}},
               "started_epoch": 1000.0, "timing_policy": "unlimited"}
    (out / "v5q-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return out, sib, receipt


@pytest.mark.parametrize("candidate", sorted(v5q.CANDIDATES))
def test_visible_timing_projection_is_informational_and_parity_result_is_written(tmp_path, candidate):
    out, sib, _ = _visible_run(tmp_path, candidate)
    result = tmp_path / "parity.json"
    data = compare.compare_visible(candidate, out, sib, result)
    assert data["status"] == "PASS_VISIBLE_PARITY_UNQUALIFIED" and result.is_file()
    assert data["timing_informational"]["informational"] is True
    assert "within_guard" not in data["timing_informational"]
    assert data["timing_projection"]["projected_conservative_seconds"] > 24_300
    second = v5q.compare_visible(candidate, out, sib, tmp_path / 'embedded-entry.json')
    assert second == data


def _rewrite_receipt(out, **changes):
    receipt = json.loads((out / "v5q-receipt.json").read_text(encoding="utf-8"))
    receipt.update(changes)
    (out / "v5q-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


@pytest.mark.parametrize("fault", ["formula", "hash", "timing_receipt", "invalid_time", "provenance", "ids", "port"])
def test_visible_comparator_keeps_every_other_check_blocking(tmp_path, fault):
    out, sib, receipt = _visible_run(tmp_path, "A5-172")
    if fault == "formula":
        ids, values = v5q._read_csv(out / "submission.csv")
        values[0, 0] = min(1.0, values[0, 0] + 1e-3)
        (out / "submission.csv").unlink()
        v5q._write_csv(out / "submission.csv", ids, values)
        _rewrite_receipt(out, submission_sha256=v5q.sha(out / "submission.csv"))
        message = "final CSV formula differs"
    elif fault == "hash":
        _rewrite_receipt(out, s6_raw_sha256="0" * 64)
        message = "output hashes differ"
    elif fault == "timing_receipt":
        timing = dict(receipt["timing_projection"], service_per_study={"decode_s": 1, "render_s": 1, "gpu_s": 1})
        _rewrite_receipt(out, timing_projection=timing)
        message = "timing receipt incomplete"
    elif fault == "invalid_time":
        timing = dict(receipt["timing_projection"], projected_from_observed_seconds=float("nan"))
        _rewrite_receipt(out, timing_projection=timing)
        message = "timing receipt incomplete"
    elif fault == "provenance":
        provenance = json.loads((out / "s6-port/provenance.json").read_text())
        provenance["code_tree_pinned_in_source"] = False
        (out / "s6-port/provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        _rewrite_receipt(out, s6_provenance_sha256=v5q.sha(out / "s6-port/provenance.json"))
        message = "S6 provenance differs"
    elif fault == "ids":
        ids, values = v5q._read_csv(sib / "ours_sub.csv")
        (sib / "ours_sub.csv").unlink()
        v5q._write_csv(sib / "ours_sub.csv", ids[::-1], values[::-1])
        message = "order mismatch"
    else:
        _rewrite_receipt(out, source_sha256="0" * 64)
        message = "execution receipt mismatch"
    result = tmp_path / "parity.json"
    with pytest.raises(ValueError, match=message):
        compare.compare_visible("A5-172", out, sib, result)
    assert not result.exists()


def test_visible_comparator_writes_hold_result_before_failing_s6_parity(tmp_path):
    out, sib, _ = _visible_run(tmp_path, "A5q-172", sibling_delta=1e-5)
    result = tmp_path / "parity.json"
    with pytest.raises(ValueError, match="visible parity failed"):
        compare.compare_visible("A5q-172", out, sib, result)
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["status"] == "HOLD_PARITY" and data["s6_public_max_abs"] > v5q.S6_ATOL


def test_visible_comparator_runs_as_a_script_for_a5(tmp_path):
    out, sib, _ = _visible_run(tmp_path, "A5-172")
    result = tmp_path / "parity.json"
    done = subprocess.run([sys.executable, str(ROOT / "tools/exp169_release_v5q_compare.py"), "--candidate", "A5-172",
                           "--output", str(out), "--sibling-output", str(sib), "--result", str(result)],
                          capture_output=True, text=True, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "PASS_VISIBLE_PARITY_UNQUALIFIED"
