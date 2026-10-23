"""K1 N-model registration schema v1.7.1; no device, data or network access.

Changelog v1.7.1: allow up to eight registered K1 models.
Changelog v1.7.0: normalized optional member weights; equal canonicalizes to absent.
Changelog v1.6.0: explicit k1_runtime_version=9; absence preserves legacy stages.
Changelog v1.5.0: optional registered K1 guard hours in [7.0, 8.8].
Changelog v1.4.0: opt-in study fallback/fail-loud policy and failed fraction.
Changelog v1.3.0: optional R6 eval stride 1/2; omission preserves legacy bytes.
Changelog v1.2.0: optional fixed-order alpha_vector overrides scalar alpha.
Changelog: register k1_eval_stride (3 default, 2 allowed).
"""
from __future__ import annotations

import copy
import math
from pathlib import PurePosixPath
import re

from tools import c85r7_rco_overlay_v1 as util

VERSION = '1.7.1'
TRAINER_SHA = '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80'
MODEL_KEYS = {'id', 'dataset_slug', 'checkpoint', 'checkpoint_sha256',
              'trainer_source', 'trainer_sha256', 'manifest_sha256'}


def current():
    """Existing x3 identities; same architecture and math as v1.1.64."""
    pins = [
        ('k1_v2', 'knee-l1-rcof0-v1-assets', util.MODEL_SHA, util.MANIFEST_SHA),
        ('k1_v3g_seed0', 'knee-k1-v3gs0-v1-assets',
         '6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5',
         'b09620471faa361f55e187a5ef188a9768be6c7e238afb69c87436309c140d77'),
        ('k1_v3g_seed1', 'knee-k1-v3gs1-v1-assets',
         '29fc2b2bbaa4e4e8ad341bba17e47c5110388db697458a8cbb9ff8d83fa9399b',
         '154d5c9ae21e69e1b22ec59bed8fb31dbabe5319c36ed38ecb142498156fd6e6'),
    ]
    return dict(schema='k1n-config/v1', release_version='1.1.65-r6w958k1n3sr20',
        kernel_id='taeyangg4/knee-lb-r6w958k1n3sr20-v1', drop_o=False, alpha=.20,
        encoder_chunk=32, models=[dict(id=mid, dataset_slug='taeyangg4/'+slug,
            checkpoint='weights/E_prior_s0_f0.pt', checkpoint_sha256=sha,
            trainer_source='source/compact_train.py', trainer_sha256=TRAINER_SHA,
            manifest_sha256=manifest) for mid, slug, sha, manifest in pins])


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) and len(set(value)) > 4


def valid_slug(value):
    return (isinstance(value, str) and re.fullmatch('[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9-]{5,70}', value)
            and not any(token in value.lower() for token in ('placeholder', 'todo', 'replace', 'pending')))


def safe_path(value, prefix, suffix):
    return (isinstance(value, str) and re.fullmatch('[A-Za-z0-9_./-]+', value)
            and value.startswith(prefix+'/') and value.endswith(suffix)
            and '..' not in PurePosixPath(value).parts
            and str(PurePosixPath(value)) == value)


def validate(value):
    """Return a canonical copy; reject unknown keys, placeholders and alias models."""
    c = copy.deepcopy(value)
    util.need(isinstance(c, dict), 'config must be an object')
    util.need(set(c) <= {'schema', 'release_version', 'kernel_id', 'drop_o', 'alpha',
                         'alpha_vector', 'encoder_chunk', 'k1_eval_stride',
                         'r6_eval_stride', 'models', 'k1_failure_policy',
                         'k1_max_failed_fraction', 'k1_guard_hours',
                         'k1_runtime_version', 'member_weights'}, 'unknown config fields')
    if 'k1_runtime_version' in c:
        util.need(type(c['k1_runtime_version']) is int and c['k1_runtime_version'] == 9,
                  'k1_runtime_version must be 9')
        util.need(c.get('k1_failure_policy') == 'study_fallback_fail_loud',
                  'runtime v9 requires study_fallback_fail_loud')
    if 'k1_guard_hours' in c:
        hours = c['k1_guard_hours']
        util.need(type(hours) in (int, float) and math.isfinite(hours) and 7.0 <= hours <= 8.8,
                  'k1_guard_hours must be finite in [7.0,8.8]')
        util.need(c.get('k1_failure_policy') == 'study_fallback_fail_loud',
                  'k1_guard_hours requires runtime v8 failure policy')
        # Explicit default is the same canonical registration as omission.
        if hours == 8.0:
            del c['k1_guard_hours']
        else:
            c['k1_guard_hours'] = float(hours)
    if 'k1_failure_policy' in c:
        util.need(c['k1_failure_policy'] == 'study_fallback_fail_loud',
                  'unsupported k1_failure_policy')
        c.setdefault('k1_max_failed_fraction', .05)
        limit = c['k1_max_failed_fraction']
        util.need(type(limit) in (int, float) and math.isfinite(limit) and 0 <= limit < 1,
                  'k1_max_failed_fraction must be finite in [0,1)')
    else:
        util.need('k1_max_failed_fraction' not in c, 'failed fraction requires k1_failure_policy')
    util.need(c.get('schema') == 'k1n-config/v1', 'config schema')
    util.need(isinstance(c.get('release_version'), str) and re.fullmatch(
        r'[0-9]+\.[0-9]+\.[0-9]+-[a-z0-9-]+', c['release_version']), 'version must be new and explicit')
    util.need(not c['release_version'].endswith('-n32'), 'base version cannot be N32')
    util.need(valid_slug(c.get('kernel_id')) and not c['kernel_id'].endswith('-n32'), 'kernel slug')
    util.need(type(c.get('drop_o')) is bool, 'drop_o must be a registered boolean')
    c.setdefault('alpha', .25 if c['drop_o'] else .20)
    c.setdefault('encoder_chunk', 32)
    c.setdefault('k1_eval_stride', 3)
    if 'r6_eval_stride' in c:
        util.need(type(c['r6_eval_stride']) is int and c['r6_eval_stride'] in (1, 2),
                  'r6_eval_stride must be 1 or 2')
    util.need(type(c['k1_eval_stride']) is int and c['k1_eval_stride'] in (2, 3),
              'k1_eval_stride must be 2 or 3')
    util.need(type(c['alpha']) in (int, float) and math.isfinite(c['alpha']) and 0 < c['alpha'] < 1,
              'alpha must be a finite weight in (0,1)')
    util.need(c['drop_o'] or c['alpha'] == .20, 'INF3 outer blend is fixed 80/20')
    if 'alpha_vector' in c:
        vector = c['alpha_vector']
        util.need(isinstance(vector, list) and len(vector) == len(util.LABELS),
                  'alpha_vector must be a list of 12 weights in canonical target order')
        util.need(all(type(a) in (int, float) and math.isfinite(a) and 0 <= a <= 1
                      for a in vector), 'alpha_vector weights must be finite numbers in [0,1]')
    util.need(type(c['encoder_chunk']) is int and c['encoder_chunk'] == 32,
              'only chunk32 registered; chunk64 FP16 gate not measured')
    models = c.get('models')
    util.need(isinstance(models, list) and 1 <= len(models) <= 8, 'register 1..8 models')
    if 'member_weights' in c:
        weights = c['member_weights']
        util.need(isinstance(weights, list) and len(weights) == len(models),
                  'member_weights must contain one weight per model')
        util.need(all(type(w) in (int, float) and math.isfinite(w) and w >= 0 for w in weights),
                  'member_weights must be finite non-negative numbers')
        util.need(max(weights) > 0, 'member_weights total must be positive')
        if len(set(weights)) == 1:
            del c['member_weights']
        else:
            util.need(c.get('k1_runtime_version') == 9, 'unequal member_weights requires runtime v9')
            # Scale before summing to avoid overflow. Preserve an already
            # normalized vector so repeated validation is byte-idempotent.
            if max(weights) <= 1 and abs(math.fsum(weights) - 1) <= 1e-15:
                c['member_weights'] = [float(w) for w in weights]
            else:
                scaled = [w / max(weights) for w in weights]
                total = math.fsum(scaled)
                c['member_weights'] = [w / total for w in scaled]
    for m in models:
        util.need(isinstance(m, dict) and set(m) == MODEL_KEYS, 'model identity fields')
        util.need(isinstance(m['id'], str) and re.fullmatch('[a-z][a-z0-9_]{0,63}', m['id']), 'model id')
        util.need(valid_slug(m['dataset_slug']), 'real dataset slug required')
        util.need(safe_path(m['checkpoint'], 'weights', '.pt'), 'checkpoint relative path')
        util.need(safe_path(m['trainer_source'], 'source', '.py'), 'trainer relative path')
        for key in ('checkpoint_sha256', 'trainer_sha256', 'manifest_sha256'):
            util.need(valid_hash(m[key]), 'real SHA256 required: '+m['id']+'/'+key)
        if m['manifest_sha256'] == util.MANIFEST_SHA:
            reference = current()['models'][0]
            util.need(all(m[k] == reference[k] for k in MODEL_KEYS - {'id'}), 'legacy asset identity')
    for key in ('id', 'checkpoint_sha256'):
        util.need(len({m[key] for m in models}) == len(models), 'duplicate model '+key)
    # mount() identifies datasets by basename, so ambiguous owner aliases are unsafe.
    util.need(len({m['dataset_slug'].split('/')[1] for m in models}) == len(models), 'duplicate mount basename')
    return c
