"""Future-screen L4 extension v2.0.0; H-B-119, not production adoption.

Changelog 2.0.0: exact >=.003 on >=3/4 named rulers, aggregate and seed
delta tensors, whole-readout malformed-point blocking (H117). All other
statistics/CI telemetry come from the authenticated v1 API unchanged.
No CLI/run_spec adapter: caller owners must explicitly migrate after live jobs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

import numpy as np

VERSION = '2.0.0'
BASE_SHA256 = '77c49ce378f7a87e29287592dc7aba2cead217740ec494d275376b14b05de504'
BASE_PATH = Path(__file__).with_name('multi_lens_readout.py')


def authenticate_base():
    """Reject source drift; never silently bless a new canonical implementation."""
    raw = BASE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise RuntimeError('Authenticated base source mismatch; explicit version review required')
    return raw


# Execute the authenticated bytes in a private namespace: no stale pyc and no
# mutation/rebinding of functions in the live canonical module. No code copy.
_base = ModuleType('tools._multi_lens_l4_v2_authenticated_base')
_base.__file__ = str(BASE_PATH)
_base.__package__ = 'tools'
exec(compile(authenticate_base(), str(BASE_PATH), 'exec'), _base.__dict__)
RULERS, TARGETS = _base.RULERS, tuple(_base.TARGETS)
NOTE = ('Inherited L4 value/ci95, target value/ci95 and positive_rulers are legacy '
        'telemetry. They are NOT a point estimate or CI for the corrected 3-of-4 rule. '
        'Only named ruler point deltas determine PASS; no CI significance gate.')
RULE = {'rulers': list(RULERS), 'target_threshold': .003, 'minimum_rulers': 3,
        'all_four_finite_required': True, 'first_two_mean_gate': False}


def _tensor(delta):
    d = np.asarray(delta)
    if (d.ndim != 3 or d.shape[0] < 1 or d.shape[1:] != (4, 12)
            or d.dtype.kind not in 'fi'):
        raise ValueError('Expected numeric delta tensor [point+draws, 4 named rulers, 12 targets]')
    return d


def _correct(l4, delta):
    selected = []
    for j, target in enumerate(TARGETS):
        points = delta[0, :, j]
        valid = bool(np.isfinite(points).all())
        count = int(np.count_nonzero(points >= .003)) if valid else None
        passed = valid and count >= 3
        l4['targets'][target].update(
            PASS=passed, eligible=valid,
            correction_status=('PASS' if passed else 'FAIL') if valid else 'NOT_EVALUABLE',
            rulers_ge_0_003=count, ruler_point_deltas=_base.clean(dict(zip(RULERS, points))),
            legacy_telemetry_note=NOTE)
        if passed:
            selected.append(target)
    valid = bool(np.isfinite(delta[0]).all())
    l4.update(PASS=valid and bool(selected), eligible=valid,
              selected_targets=selected if valid else [],
              correction_status=('PASS' if selected else 'FAIL') if valid else 'NOT_EVALUABLE',
              legacy_telemetry_note=NOTE)


def lenses(delta, label_change=False):
    """Same tuple API as v1; axes must be RULERS/TARGETS in exported order.

    Structural/type errors raise ValueError for the whole call. Nonfinite point
    data block the entire L4/readout, not just that target. Missing bootstrap
    draws retain v1 finite-draw CI behavior; they are not missing point evidence.
    """
    authenticate_base()
    d = _tensor(delta)
    ls, guard, rulers = _base.lenses(d, label_change)
    _correct(ls['L4'], d)
    return ls, guard, rulers


def verdict(ls, guard):
    """Diagnostic mechanical verdict only; no release/admission authorization."""
    authenticate_base()
    if ls['L4'].get('correction_status') not in ('PASS', 'FAIL'):
        return {'verdict': 'NOT_EVALUABLE', 'blocked': True, 'passing_lenses': [],
                'decision_label': 'NOT_EVALUABLE', 'release_scope': 'none',
                'not_release_authorization': True, 'reason': 'L4 evidence malformed or missing'}
    return {**_base.verdict(ls, guard), 'not_release_authorization': True}


class _Capture:
    """Delegate scoring unchanged, retain the already computed seed deltas."""
    def __init__(self, ctx, seed_count):
        self.ctx, self.seed_count, self.seed_deltas = ctx, seed_count, []

    def __getattr__(self, name):
        return getattr(self.ctx, name)

    def delta(self, arm, control):
        d = _tensor(self.ctx.delta(arm, control))
        if len(self.seed_deltas) < self.seed_count:
            self.seed_deltas.append(d.copy())
        return d


def evaluate(ctx, arms, controls, label_change=False, bundle=None, image_reads=None):
    """v1 evaluate signature, using each seed's actual paired delta exactly once.

    Aggregate is the mean of seed AUC deltas, as in v1 (not the L5 AUC of
    seed-mean predictions). Existing L1-L3/L5-L7, rulers, guard, warnings and
    numeric telemetry are untouched. release_scope is diagnostic, never approval.
    """
    authenticate_base()
    capture = _Capture(ctx, len(arms))
    result = _base.evaluate(capture, arms, controls, label_change, bundle, image_reads)
    _correct(result['lenses']['L4'], np.mean(capture.seed_deltas, axis=0))
    for seed, delta in zip(arms, capture.seed_deltas):
        _correct(result['per_seed'][seed]['L4'], delta)
    result.update(verdict(result['lenses'], result['harm_guard']))
    result.update(version=VERSION, base_version=_base.VERSION,
                  base_source_sha256=BASE_SHA256, l4_rule={**RULE, 'rulers': list(RULERS)})
    authenticate_base()
    return result
