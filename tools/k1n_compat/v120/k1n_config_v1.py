"""K1 N-model registration schema v1.2.0; no device, data or network access.

Changelog v1.2.0: optional fixed-order alpha_vector overrides scalar alpha.
Changelog: register k1_eval_stride (3 default, 2 allowed).
"""
from __future__ import annotations

import copy
import math
from pathlib import PurePosixPath
import re

from tools import c85r7_rco_overlay_v1 as util

VERSION = '1.2.0'
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
                         'alpha_vector', 'encoder_chunk', 'k1_eval_stride', 'models'}, 'unknown config fields')
    util.need(c.get('schema') == 'k1n-config/v1', 'config schema')
    util.need(isinstance(c.get('release_version'), str) and re.fullmatch(
        r'[0-9]+\.[0-9]+\.[0-9]+-[a-z0-9-]+', c['release_version']), 'version must be new and explicit')
    util.need(not c['release_version'].endswith('-n32'), 'base version cannot be N32')
    util.need(valid_slug(c.get('kernel_id')) and not c['kernel_id'].endswith('-n32'), 'kernel slug')
    util.need(type(c.get('drop_o')) is bool, 'drop_o must be a registered boolean')
    c.setdefault('alpha', .25 if c['drop_o'] else .20)
    c.setdefault('encoder_chunk', 32)
    c.setdefault('k1_eval_stride', 3)
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
    util.need(isinstance(models, list) and 1 <= len(models) <= 6, 'register 1..6 models')
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
