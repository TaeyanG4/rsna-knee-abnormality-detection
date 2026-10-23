"""v1.0.0: strict runtime11 per-member evaluation strides; legacy schema unchanged.

Changelog: validate a complete model-id -> stride map before delegating all
unchanged fields to the frozen runtime10 schema. No interpreter/device access.
"""
import copy
from tools import k1n_config_v1 as legacy

VERSION = '1.0.0'


def validate(value):
    legacy.util.need(isinstance(value, dict), 'config must be an object')
    c = copy.deepcopy(value)
    legacy.util.need(c.get('k1_runtime_version') == 11, 'runtime11 required')
    strides = c.pop('k1_member_eval_strides', None)
    c['k1_runtime_version'] = 10
    c = legacy.validate(c)
    ids = {m['id'] for m in c['models']}
    legacy.util.need(isinstance(strides, dict) and set(strides) == ids,
                     'complete per-member stride identity required')
    legacy.util.need(all(type(s) is int and s in (1, 2, 3) for s in strides.values()),
                     'per-member stride must be integer 1, 2 or 3')
    return dict(c, k1_runtime_version=11, k1_member_eval_strides=strides)


def current():
    c = legacy.current()
    return validate(dict(c, k1_eval_stride=2, k1_runtime_version=11,
        k1_failure_policy='study_fallback_fail_loud',
        k1_member_eval_strides={m['id']: 1 if m['id'] == 'k1_v3g_seed1' else 2
                               for m in c['models']}))
