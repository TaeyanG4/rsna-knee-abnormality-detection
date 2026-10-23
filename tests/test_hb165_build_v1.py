"""CPU tests v1.0.0 for tools/hb165_build_v1.py (H-B-165 fold-3 OA crop-expert confirmation)."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from tools import hb165_build_v1 as b
from tools import hb155_build_v1 as parent

REGISTERED = all(b.config_path(a, s, e).is_file() for a in b.ARMS for s in b.SEEDS for e in (False, True))
needs_registration = pytest.mark.skipif(not REGISTERED, reason='stage not registered yet')
TRAINING_KEYS = ('arm', 'corpus', 'corpus_contract', 'weights', 'label_path', 'label_semantic_sha256', 'resolution', 'epochs',
                 'batch', 'microbatch', 'accumulation', 'train_k', 'workers', 'gpu_name', 'selection', 'side_unknown_policy',
                 'exclusions_sha256', 'qa_thresholds', 'clean_oof', 'exposure', 'review_required')
ALLOWED_TOP = {'schema', 'version', 'experiment', 'seed', 'fold', 'train', 'val', 'gold', 'inputs', 'gate', 'holdout_status',
               'coordinator_decision', 'estimand', 'selection_exposure', 'exposure_details', 'recipe', 'member',
               'parent_config', 'gold_policy', 'engineering_only', 'optimizer_steps'}


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in ('inputs', 'parent_config'):  # path-keyed maps are leaves
            out.update(flat(v, prefix+k+'.'))
        else:
            out[prefix+k] = v
    return out


def changed(a, c):
    fa, fc = flat(a), flat(c)
    return {k for k in set(fa) | set(fc) if fa.get(k) != fc.get(k)}


def test_derive_is_identity_on_the_parent_population():
    """The ONLY transformation reproduces each sealed H-B-155 config from its own fold-2 population."""
    for (arm, seed) in b.PARENT_CONFIG_SHA256:
        base = parent.read(parent.config_path(arm, seed))
        again = b.derive(base, seed=seed, train=base['train'], val=base['val'], gold=base['gold'],
                         fold_root=parent.FOLD, engineering=False)
        assert again == base


@needs_registration
@pytest.mark.parametrize('arm,seed', [(a, s) for s in b.SEEDS for a in b.ARMS])
def test_scientific_config_changes_only_registered_fields(arm, seed):
    c = b.read(b.config_path(arm, seed)); base = parent.read(parent.config_path(arm, b.PARENT_SEED[seed]))
    assert set(c) - set(base) <= {'member', 'parent_config', 'gold_policy'}
    assert {k for k in set(c) | set(base) if c.get(k) != base.get(k)} <= ALLOWED_TOP
    for key in TRAINING_KEYS:
        assert c[key] == base[key], key
    assert changed(base['recipe'], c['recipe']) == {'roots.ordered', 'roots.groups', 'roots.train_sha256', 'roots.val_sha256'}
    assert c['recipe']['updates'] == base['recipe']['updates'] and c['optimizer_steps'] == base['optimizer_steps'] == 8680
    assert (c['batch'], c['microbatch'], c['accumulation'], c['epochs'], c['resolution'], c['train_k']) == (8, 2, 4, 20, 224, 12)
    assert c['fold'] == 3 and c['seed'] == seed and c['exposure'] == 'EXPOSED' and c['clean_oof'] is False
    assert len(c['train']) == 3479 and len(c['val']) == 864 and len(c['gold']) == 58
    assert not set(c['train']) & set(c['val']) and not set(c['gold']) & set(c['train'] + c['val'])
    for p, d in base['inputs'].items():
        assert c['inputs'][p] == d  # every parent dependency is still bound with the same bytes


@needs_registration
@pytest.mark.parametrize('seed', b.SEEDS)
def test_pair_differs_only_in_input_cache(seed):
    ctl, crop = (b.read(b.config_path(a, seed)) for a in b.ARMS)
    keys = changed({k: v for k, v in ctl.items() if k != 'inputs'}, {k: v for k, v in crop.items() if k != 'inputs'})
    assert keys == {'arm', 'member', 'corpus', 'corpus_contract', 'recipe.windows.native_source', 'recipe.roots.corpus',
                    'recipe.roots.manifest', 'parent_config'}
    diff = {p for p in set(ctl['inputs']) | set(crop['inputs']) if ctl['inputs'].get(p) != crop['inputs'].get(p)}
    assert diff == {parent.rel(parent.config_path('control', b.PARENT_SEED[seed])), parent.rel(parent.config_path('crop', b.PARENT_SEED[seed]))}


@needs_registration
def test_seed_members_differ_only_in_seed_identity():
    a, c = (b.read(b.config_path('control', s)) for s in b.SEEDS)
    keys = changed({k: v for k, v in a.items() if k != 'inputs'}, {k: v for k, v in c.items() if k != 'inputs'})
    assert keys == {'seed', 'member', 'parent_config'}


@needs_registration
def test_fold3_roster_exclusions_and_gold():
    train, val = b.roster()
    assert (len(train), len(val)) == (3479, 864) and b.uid_hash(val) == b.VAL_UID_SHA256
    e = b.read(b.WORK/'exclusions-check.json')
    assert e['removed_train'] == e['removed_val'] == 0 and e['excluded_in_fold3_train'] == e['excluded_in_fold3_val'] == 0
    assert e['excluded_unique_studies'] == 2 and e['registered_exclusions_sha256'] == list(e['exclusions_record'].values())[0]
    gold = b.gold_ids(); assert len(gold) == 58 and not set(gold) & set(train + val)
    if (b.ROOT/b.GOLD_SOURCE).is_file():
        assert b.read(b.WORK/'gold58-ids.json') == b.gold_record()


@needs_registration
def test_registered_configs_validate_and_registration_binds_readout():
    for s in b.SEEDS:
        for a in b.ARMS:
            for e in (False, True):
                assert b.validate(b.config_path(a, s, e))['engineering_only'] is e
    reg = b.read(b.WORK/'registration.json')
    assert reg == b.registration_value() and reg['gate'] == b.GATE
    assert reg['readout']['source_sha256'] == b.sha(b.ROOT/b.READOUT)
    assert reg['clean_oof'] is False and 'D-B-157' in reg['exposure_statement']


def test_gate_registers_exact_coordinator_rule():
    g = b.GATE
    assert g['routed_targets'] == ['Medial OA', 'Lateral OA'] and g['routed_macro_mean_min'] == .0005
    assert g['draws'] == 2000 and g['ci_lower_strictly_above'] == 0 and g['oa_target_min'] == .003
    assert g['oa_target_min_rulers'] == 3 and g['each_seed_mean_oa_strictly_above'] == 0 and g['non_oa_identical'] is True
    assert parent.GATE['macro'] == .002  # H-B-155 thresholds untouched


def test_decision_missing_unknown_or_fabricated_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'WORK', tmp_path)
    with pytest.raises(ValueError, match='missing'):
        b.decision_binding()
    with pytest.raises(ValueError, match='unknown'):
        b.decision_binding('reviewer-PASS')
    (tmp_path/'coordinator-decision.json').write_text(json.dumps(dict(b.COORDINATOR_DECISION, clean_oof=True)))
    with pytest.raises(ValueError, match='differs'):
        b.decision_binding()
    with pytest.raises(ValueError, match='immutable'):
        b.decision_binding(b.DECISION_ID)


def authorized(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'WORK', tmp_path)
    inputs = b.decision_binding(b.DECISION_ID)
    monkeypatch.setattr(b, 'registration_binding', lambda: {'reg.json': 'R'})
    inputs['reg.json'] = 'R'
    return dict(engineering_only=False, holdout_status='EXPOSED_PAIRED_FOLD3_AUTHORIZED', fold=3, seed=44,
                exposure='EXPOSED', clean_oof=False, coordinator_decision=b.COORDINATOR_DECISION,
                estimand=b.COORDINATOR_DECISION['estimand'], selection_exposure=b.SELECTION_EXPOSURE,
                gate=copy.deepcopy(b.GATE), inputs=inputs)


@pytest.mark.parametrize('field,value', [('fold', 2), ('seed', 42), ('clean_oof', True), ('exposure', 'CLEAN'),
    ('selection_exposure', 'none'), ('estimand', 'absolute AUC'), ('coordinator_decision', {}), ('inputs', {}),
    ('gate', dict(b.GATE, routed_macro_mean_min=0.0)), ('holdout_status', 'EXPOSED_PAIRED_FOLD2_AUTHORIZED')])
def test_registered_semantics_cannot_be_changed(tmp_path, monkeypatch, field, value):
    c = authorized(tmp_path, monkeypatch); c[field] = value
    monkeypatch.setattr(b, 'validate', lambda p: c)
    with pytest.raises(ValueError, match='explicit registered'):
        b.require_fit('x', 'fake.json')


def test_fit_needs_bound_independent_non_builder_review(tmp_path, monkeypatch):
    c = authorized(tmp_path, monkeypatch)
    monkeypatch.setattr(b, 'validate', lambda p: c)
    with pytest.raises(ValueError, match='independent review required'):
        b.require_fit('x')
    path = tmp_path/'review.json'
    r = dict(schema='hb165-review/v1', verdict='PASS', independent=True, reviewer='R', bindings={'proof': 'stale'})
    path.write_text(json.dumps(r)); monkeypatch.setattr(b, 'review_bindings', lambda: {'proof': 'current'})
    with pytest.raises(ValueError, match='smoke-bound'):
        b.require_fit('x', path)
    for bad in (dict(r, bindings={'proof': 'current'}, reviewer='claude-hb165-builder'),
                dict(r, bindings={'proof': 'current'}, schema='hb155-review/v1'),
                dict(r, bindings={'proof': 'current'}, independent=False)):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='smoke-bound'):
            b.require_fit('x', path)
    path.write_text(json.dumps(dict(r, bindings={'proof': 'current'})))
    assert b.require_fit('x', path) == c and b.admit(path)['status'] == 'ADMITTED'


def runtime_fixture(tmp_path, monkeypatch, change=None):
    p, w = tmp_path/'parent', tmp_path/'work'
    (p/'runtime').mkdir(parents=True); (w/'runtime').mkdir(parents=True)
    snap = parent.read(parent.WORK/'runtime/4090.json')
    (p/'runtime/4090.json').write_text(json.dumps(snap))
    target = dict(snap, **(change or {}))
    (w/'runtime/4090.json').write_text(json.dumps(target))
    (p/'runtime-review.json').write_text(json.dumps(dict(schema='hb155-runtime-review/v1', verdict='PASS', independent=True,
                                                         reviewer='codex', bindings={'parent': 'x'})))
    monkeypatch.setattr(parent, 'WORK', p); monkeypatch.setattr(b, 'WORK', w)
    monkeypatch.setattr(parent, 'runtime_bindings', lambda: {'parent': 'x'})
    monkeypatch.setattr(b, 'rel', lambda q: Path(q).relative_to(tmp_path).as_posix())
    return w, target


def test_runtime_review_reused_only_for_identical_snapshot(tmp_path, monkeypatch):
    w, _ = runtime_fixture(tmp_path, monkeypatch)
    bindings, diff = b.runtime_evidence()
    assert diff == {} and b.runtime_bindings() == bindings and 'work/runtime/4090.json' in bindings


def test_changed_runtime_needs_new_independent_review(tmp_path, monkeypatch):
    w, target = runtime_fixture(tmp_path, monkeypatch, change={'torch_interop_threads': 64})
    with pytest.raises(ValueError, match='runtime differs'):
        b.runtime_bindings()
    bindings, diff = b.runtime_evidence()
    review = dict(schema='hb165-runtime-review/v1', verdict='PASS', independent=True, reviewer='claude-hb165-builder',
                  bindings=bindings, diff_vs_h155=diff)
    (w/'runtime-review.json').write_text(json.dumps(review))
    with pytest.raises(ValueError, match='absent/stale'):
        b.runtime_bindings()
    (w/'runtime-review.json').write_text(json.dumps(dict(review, reviewer='astra-3')))
    assert 'work/runtime-review.json' in b.runtime_bindings()


@pytest.mark.parametrize('change,match', [({'gpu': 'NVIDIA GeForce RTX 3090'}, '4090'), ({'torch': '2.6.0'}, 'torch'),
    ({'cudnn_benchmark': True}, 'policy'), ({'libs': {'timm': '1.0.29'}}, 'pin')])
def test_target_pins_enforced(tmp_path, monkeypatch, change, match):
    runtime_fixture(tmp_path, monkeypatch, change=change)
    with pytest.raises(ValueError, match=match):
        b.runtime_evidence()


def test_queue_values_are_valid_and_distinct():
    from tools.gpu_queue import validate_plan
    for mode in ('smoke', 'fit'):
        q = b.queue_value(mode); validate_plan(q)
        assert q[0]['tag'] == 'hb165-'+mode+'-v100' and q[0]['command'][:4] == ['python', '-B', '-m', 'tools.hb165_build_v1']
    assert b.queue_value('fit')[0]['command'][-2:] == ['--review', '.local/scratch/slot-b/hb165/pilot-fold3-v100/review.json']
    assert '--review' not in b.queue_value('smoke')[0]['command']
    if REGISTERED:
        for mode in ('smoke', 'fit'):
            assert b.read(b.WORK/b.QUEUES[mode]) == b.queue_value(mode)


def test_fit_sequence_never_launches_without_admission(monkeypatch, tmp_path):
    monkeypatch.setattr(b, 'WORK', tmp_path); calls = []
    monkeypatch.setattr(b, 'disk_gate', lambda mode, path=None: None)
    monkeypatch.setattr(b, 'admit', lambda review=None: dict(status='HOLD', reason='x'))
    monkeypatch.setattr(b.subprocess, 'run', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='admission HOLD'):
        b.sequence('fit', 'review.json')
    def blocked():
        raise ValueError('runtime differs')
    monkeypatch.setattr(b, 'smoke_gate', blocked)
    with pytest.raises(ValueError, match='runtime differs'):
        b.sequence('smoke')
    assert calls == []


def test_sequence_order_and_watchdog_commands(monkeypatch, tmp_path):
    monkeypatch.setattr(b, 'WORK', tmp_path); calls = []
    monkeypatch.setattr(b, 'disk_gate', lambda mode, path=None: None)
    monkeypatch.setattr(b, 'smoke_gate', lambda: {})
    monkeypatch.setattr(b.subprocess, 'run', lambda cmd, **k: calls.append(cmd))
    assert b.sequence('smoke')['status'] == 'FOUR_SMOKE_COMPLETE'
    names = [c[c.index('--config')+1] for c in calls]
    assert [Path(n).stem for n in names] == ['control-s44', 'crop-s44', 'control-s45', 'crop-s45']
    for c in calls:
        assert c[2:4] == ['-m', 'tools.hb165_watchdog_v1'] and '--measurement' in c
        assert c[c.index('--')+1:][2:5] == ['-m', 'tools.hb165_fit_v1', 'smoke'] and c[-1] == 'cuda'


def test_disk_gate_uses_explicit_requirement(monkeypatch, tmp_path):
    class U:
        def __init__(self, free): self.free = free
    assert b.disk_requirement('fit') == 4*(20*b.EPOCH_BYTES + b.FINAL_BYTES + b.OUTPUT_BYTES) + b.DISK_MARGIN
    assert b.disk_requirement('smoke') < b.disk_requirement('fit')
    monkeypatch.setattr(b.shutil, 'disk_usage', lambda p: U(b.disk_requirement('fit') - 1))
    with pytest.raises(ValueError, match='insufficient disk'):
        b.disk_gate('fit', tmp_path)
    assert b.disk_gate('smoke', tmp_path)['status'] == 'DISK_OK'


@needs_registration
def test_local_cpu_smoke_receipts():
    if not all((b.WORK/'cpu-local'/b.member(a, s)/'summary.json').is_file() for a in b.ARMS for s in b.SEEDS):
        pytest.skip('local CPU smokes not run yet')
    assert len(b.cpu_evidence('cpu-local')) == 8


@needs_registration
def test_admit_holds_without_review():
    assert b.admit()['status'] == 'HOLD'
