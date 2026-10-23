"""CPU-only tests for tools/submission_screen.py (H-B-26) on a synthetic artifact tree, plus
read-only regressions on the real EXP099/EXP107 artifacts when they are present."""
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('submission_screen_under_test', ROOT / 'tools/submission_screen.py')
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
T = S.TARGETS
E = 'artifacts/experiments'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wjson(p, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1), encoding='utf-8')


def wcsv(p, df):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


class World:
    """A miniature copy of the project layout: folds, components, three label references,
    label-set registrations, protocol manifests, fold runs, OOF dirs and Public3 outputs."""

    def __init__(self, root, n=100, seed=0):
        self.root = Path(root)
        self.rng = np.random.default_rng(seed)
        self.uids = [f'1.2.{i:03d}' for i in range(n)]
        self.fold = {u: i % 5 for i, u in enumerate(self.uids)}
        self.y = (self.rng.random((n, 12)) < 0.4).astype(int)
        comp = {u: 'c-' + u for u in self.uids}
        comp[self.uids[5]] = comp[self.uids[0]]  # one two-study component, both in fold 0
        wcsv(self.root / E / 'EXP099_KFOLD_PROTOCOL_V1/folds.csv',
             pd.DataFrame({'StudyInstanceUID': self.uids, 'component_id': [comp[u] for u in self.uids], 'fold': [self.fold[u] for u in self.uids]}))
        groups = {}
        for u in self.uids:
            groups.setdefault(comp[u], []).append(u)
        wjson(self.root / E / 'VALIDATION_GROUP_CONTRACT_V1/components.json', [dict(component_id=c, members=m) for c, m in groups.items()])
        self.manifest = self.make_protocol('EXP099_KFOLD_PROTOCOL_V1')
        soft = np.where(self.y == 1, 0.85, 0.05)
        self.write_labels('storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv', soft)
        gold = [f'9.9.{i}' for i in range(3)]
        self.gold = gold
        fl = pd.DataFrame(np.where(self.y == 1, 0.975, 0.25), columns=T)
        fl.insert(0, 'StudyInstanceUID', self.uids)
        fl = pd.concat([fl, pd.DataFrame([[g] + [1.0] * 12 for g in gold], columns=['StudyInstanceUID', *T])])
        wcsv(self.root / 'storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv', fl)
        rl = soft.copy()
        rl[: n // 10, 0] = 1 - rl[: n // 10, 0]
        self.write_labels('storage/labels/derived_relabel/relabel_v1.csv', rl)
        wjson(self.root / 'storage/labels/derived_relabel/provenance.json', dict(labeller='synthetic LLM'))
        self.register('dread', 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv', 'sem-dread')
        self.register('relabel_v1', 'storage/labels/derived_relabel/relabel_v1.csv', 'sem-relabel')
        tr = pd.DataFrame({'StudyInstanceUID': self.uids + gold, 'Report': ['report, text'] * (n + 3)})
        for i, t in enumerate(T):
            tr[t] = [np.nan] * n + [1.0, 0.0, 1.0]
        wcsv(self.root / 'storage/competition/train.csv', tr)
        sample = pd.DataFrame({'StudyInstanceUID': ['t0', 't1', 't2'], **{t: [0.5] * 3 for t in T}})
        wcsv(self.root / 'storage/competition/sample_submission.csv', sample)
        self.registry = self.root / S.DEFAULTS['fingerprint_registry']
        wjson(self.registry, dict(schema=S.REGISTRY_SCHEMA, entries=[]))  # an empty but present duplicate registry

    def write_labels(self, rel, soft):
        df = pd.DataFrame(soft, columns=T)
        df.insert(0, 'StudyInstanceUID', self.uids)
        wcsv(self.root / rel, df)

    def register(self, name, rel, semantic):
        wjson(self.root / E / 'EXP099_KFOLD_PROTOCOL_V1/label-sets' / f'{name}.json',
              dict(name=name, path=rel, sha256=sha(self.root / rel), semantic_sha256=semantic, gold_overlap=0))

    def make_protocol(self, name, leak_fold=None):
        P = self.root / E / name
        folds = {}
        for k in range(5):
            val = [u for u in self.uids if self.fold[u] == k]
            train = [u for u in self.uids if self.fold[u] != k]
            if k == leak_fold:
                train = train + val[:2]
            p = P / f'fold{k}/ordered-ids.json'
            wjson(p, dict(train=train, val=val))
            folds[str(k)] = dict(fold=k, ordered_ids=dict(path=p.relative_to(self.root).as_posix(), sha256=sha(p),
                                                          train_sha256=S.uid_hash(train), val_sha256=S.uid_hash(val)))
        wjson(P / 'manifest.json', dict(schema='synthetic-kfold', seed=42, folds=folds, common_args=dict(pretrained='timm', arch='toy')))
        return P / 'manifest.json'

    def preds(self, noise):
        return np.clip(0.5 + (self.y - 0.5) * 0.3 + noise * self.rng.normal(size=self.y.shape), 0.001, 0.999)

    def member(self, name, noise=0.3, label_set='dread', label_hash='sem-dread', protocol=None, prov_bad_fold=None, extra_result=None, p=None,
               prov_for=None, source_extra=None):
        protocol = protocol or self.manifest
        man = json.loads(Path(protocol).read_text())
        p = self.preds(noise) if p is None else p
        df = pd.DataFrame(p, columns=T)
        df.insert(0, 'StudyInstanceUID', self.uids)
        d = self.root / E / f'{name}_OOF'
        sources, ckpts = {}, []
        for k in range(5):
            oi = json.loads((self.root / man['folds'][str(k)]['ordered_ids']['path']).read_text())
            run = self.root / E / f'{name}_F{k}'
            prov = dict(train_ids_ordered_sha256=S.uid_hash(oi['train']), eval_ids_ordered_sha256=S.uid_hash(oi['val']),
                        label_hash=label_hash, gold_access='disabled', seed=42)
            if k == prov_bad_fold:
                prov['train_ids_ordered_sha256'] = 'not-the-protocol-list'
            if prov_for is not None:
                prov.update(prov_for(k))
            wjson(run / 'provenance.json', prov)
            ck = hashlib.sha256(f'{name}-{k}'.encode()).hexdigest()
            ckpts.append(ck)
            wjson(run / 'output-inventory.json', {'model_best.pt': dict(sha256=ck)})
            fd = df[df['StudyInstanceUID'].isin(oi['val'])]
            wcsv(run / 'oof_k36.csv', fd)
            sources[str(k)] = dict(run=run.relative_to(self.root).as_posix(), oof_k36=(run / 'oof_k36.csv').relative_to(self.root).as_posix(),
                                   oof_k36_sha256=sha(run / 'oof_k36.csv'), **(source_extra(k) if source_extra else {}))
        wcsv(d / 'oof_k36.csv', df)
        res = dict(schema='exp099-oof-v1', label_set=label_set, protocol_sha256=sha(protocol), seed=42, sources=sources,
                   outputs={'oof_k36.csv': sha(d / 'oof_k36.csv')}, **(extra_result or {}))
        wjson(d / 'result.json', res)
        return dict(name=name, oof=(d / 'oof_k36.csv').relative_to(self.root).as_posix(), ckpts=ckpts)

    def release(self, cand, arms, status='PASS_TEST_COHORT_ONLY', tamper=False, bad_ckpt=False, projected=1000.0, weights_reported=None, manifest_sha=None):
        out = self.root / '.local/rel' / f'out-{cand}'
        sub = pd.DataFrame({'StudyInstanceUID': ['t0', 't1', 't2'], **{t: [0.2, 0.4, 0.6] for t in T}})
        wcsv(out / 'submission.csv', sub)
        good_sha = sha(out / 'submission.csv')
        for arm, (m, w) in arms.items():
            cks = list(m['ckpts'])
            if bad_ckpt:
                cks[0] = 'f' * 64
            wjson(out / cand / arm / 'work/result.json', dict(status=status, release=cand, arm=arm, n_models=len(cks), k_eval=36, submission_sha256=good_sha if len(arms) == 1 else 'arm-' + arm,
                                                              checkpoints=[dict(tag=f'{arm}_F{k}', sha256=c) for k, c in enumerate(cks)], elapsed_seconds=10.0,
                                                              **({'release_manifest_sha256': manifest_sha} if manifest_sha and len(arms) == 1 else {})))
        if len(arms) > 1:
            tot = sum(w for _, w in arms.values())
            rw = weights_reported or {a: w / tot for a, (_, w) in arms.items()}
            wjson(out / cand / 'combine/result.json', dict(status='PASS_RANK_MEAN_TEST_COHORT_ONLY' if status.startswith('PASS') else status, release=cand,
                                                           submission_sha256=good_sha, rank_weights=rw, elapsed_seconds=20.0,
                                                           **({'release_manifest_sha256': manifest_sha} if manifest_sha else {})))
        if tamper:
            (out / 'submission.csv').write_text((out / 'submission.csv').read_text().replace('0.2', '0.3'))
        wjson(self.root / '.local/rel' / cand / 'execution-plan.json', dict(kernel=f'user/{cand}', max_studies_passing_screen=2000,
                                                                           screen_at_expected_hidden_N=dict(studies=1300, projected_seconds=projected, internal_seconds=2000.0)))
        return dict(out_dir=out.relative_to(self.root).as_posix())

    def distill_labels(self, teacher):
        rel = 'storage/labels/dx/distill.csv'
        soft = np.where(self.y == 1, 0.85, 0.05) * 0.5 + 0.25
        self.write_labels(rel, soft)
        tp = self.root / teacher['oof']
        wjson(self.root / 'storage/labels/dx/receipt.json', dict(teacher_rule='equal mean of teacher OOF', teachers={'t1': dict(path=teacher['oof'], sha256=sha(tp))},
                                                                   output=dict(path=rel, sha256=sha(self.root / rel))))
        self.register('distill_x', rel, 'sem-dx')


RULES = [dict(on='public', op='>=', value=0.920, reading='transfer confirmed'), dict(on='public', op='<=', value=0.914, reading='no transfer')]
REG = '2026-10-01 09:00 KST'
# a submissions cache that lists no submission: duplicate coverage is complete (astra re-review 2026-10-08 A1: no cache at all is HOLD)
NO_SUBS = dict(path='synthetic-empty-submissions-cache.csv', sha256='0' * 64, modified='2026-10-08T00:00:00+09:00', rows={})


def empty_cache(w):
    p = w.root / 'subs_empty.csv'
    p.write_text('ref,fileName,date,description,status,publicScore,privateScore\n', encoding='utf-8')
    return p


def spec(name, members, release, rules=RULES, ref=None, **kw):
    reg = kw.pop('registered_at', REG)
    dec = dict(plan_item='H-X', rules=rules, reference=dict(public=0.914), basis=kw.pop('basis', 'public')) if rules else dict(plan_item='H-X', text='descriptive')
    if rules and reg is not None:
        dec['registered_at'] = reg
    return dict(candidate=name, members=members, release=release, decision=dec, submission_ref=ref, **kw)


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    w.base = w.member('EXP099_B8_DREAD', noise=0.35)  # sits at the default baseline path
    assert w.base['oof'] == S.DEFAULTS['baseline_oof']
    return w


def ctx_for(w, **kw):
    return S.Ctx(w.root, draws=30, **kw)


def test_auc_and_labels_rules(tmp_path):
    y = np.array([[0], [0], [1], [1]])
    assert S.auc_cols(y, np.array([[0.1], [0.2], [0.3], [0.9]]))[0] == 1.0
    assert S.auc_cols(y, np.array([[0.5], [0.5], [0.5], [0.5]]))[0] == 0.5
    assert np.isnan(S.auc_cols(np.zeros((3, 1)), np.array([[0.1], [0.2], [0.3]]))[0])
    p = tmp_path / 'l.csv'
    df = pd.DataFrame({'StudyInstanceUID': ['a', 'b', 'g'], **{t: [0.5, 0.51, 1.0] for t in T}})
    df.to_csv(p, index=False)
    y_rint, _ = S.load_labels(p, ['a', 'b'], 'rint')
    y_ge, _ = S.load_labels(p, ['a', 'b'], 'ge0.5')
    assert y_rint[:, 0].tolist() == [0, 1] and y_ge[:, 0].tolist() == [1, 1]
    _, dropped = S.load_labels(p, ['a'], 'ge0.5', gold={'g'})
    assert dropped == 1
    with pytest.raises(ValueError):
        S.load_labels(p, ['g'], 'ge0.5', gold={'g'})


def test_rank_blend_formula():
    a = pd.DataFrame(np.array([[0.1] * 12, [0.9] * 12, [0.5] * 12]), columns=T)
    b = pd.DataFrame(np.array([[0.3] * 12, [0.2] * 12, [0.1] * 12]), columns=T)
    got = S.combine([a, b], [1, 1])
    exp = (a.rank(pct=True).to_numpy() + b.rank(pct=True).to_numpy()) / 2
    assert np.allclose(got, exp)
    assert np.allclose(S.combine([a], [5]), a.to_numpy())
    with pytest.raises(ValueError):
        S.combine([a, b], [1, 0])


def test_clean_member_passes_and_gold_is_report_only(world):
    w = world
    m = w.member('GOOD', noise=0.2)
    rel = w.release('good', {'good': (m, 1.0)})
    rep = S.run_many(ctx_for(w), [spec('good', [dict(name='good', arm='good', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    lin = rep['lineage']
    assert lin['status'] == 'PASS'
    mem = lin['members'][0]
    assert mem['direct_overlap'] == 0 and mem['indirect_overlap'] == 0 and mem['studies_validated'] == 100
    assert all(r['train_hash_ok'] and r['eval_hash_ok'] and r['label_hash_ok'] and r['fold_oof_sha_ok'] for r in mem['runs'].values())
    assert lin['gold58']['n_gold'] == 3 and lin['gold58']['absent_from_training'] is True
    assert rep['engineering']['engineering_pass'] is True and rep['engineering']['arms']['good']['binding'] == 'full_reported'
    assert rep['engineering']['runtime']['within_budget'] is True
    assert rep['decision']['state'] == 'OPEN' and rep['decision']['pre_registered']
    assert rep['verdict']['verdict'] == 'SUBMIT-ELIGIBLE'


def test_three_references_always_scored_with_direction_line(world):
    w = world
    m = w.member('GOOD', noise=0.05)
    rel = w.release('good', {'good': (m, 1.0)})
    rep = S.run_many(ctx_for(w), [spec('good', [dict(name='good', arm='good', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    by = rep['oof']['by_label_set']
    assert set(by) == {'dread', 'flight_v4', 'relabel_v1'}
    assert by['flight_v4']['gold_rows_dropped'] == 3 and by['flight_v4']['labels'] == 'soft label >= 0.5'
    assert by['dread']['labels'].startswith('rint')
    for b in by.values():
        assert b['n_folds'] == 5 and len(b['pooled']['ci95']) == 2 and set(b['pooled']['targets']) == set(T)
    d = rep['oof']['direction']
    assert d['consistent'] is True and 'CONSISTENT' in d['line']
    assert 'report-vs-image' in rep['oof']['ceiling_note']
    md = S.render_md(rep)
    assert 'direction consistency' in md and 'report-vs-image' in md


def test_direction_consistency_rule():
    def blk(delta, lo):
        return dict(pooled=dict(delta=delta, ci95=[lo, lo + 0.01]))
    ok = S.direction_consistency(dict(dread=blk(0.01, 0.001), flight_v4=blk(-0.002, -0.01), relabel_v1=blk(0.0, -0.01)), -0.002)
    assert ok['consistent'] is True
    assert S.direction_consistency(dict(dread=blk(0.01, -0.001), flight_v4=blk(0.01, 0), relabel_v1=blk(0.01, 0)), -0.002)['consistent'] is False
    assert S.direction_consistency(dict(dread=blk(0.01, 0.001), flight_v4=blk(-0.0021, 0), relabel_v1=blk(0.01, 0)), -0.002)['consistent'] is False
    assert S.direction_consistency(dict(dread=blk(0.01, 0.001)), -0.002)['consistent'] is None


def test_identical_candidate_has_zero_delta(world):
    w = world
    rel = w.release('base', {'base': (w.base, 1.0)})
    rep = S.run_many(ctx_for(w), [spec('base', [dict(name='base', arm='base', oof=w.base['oof'])], rel)], NO_SUBS, write=False)[0]
    p = rep['oof']['by_label_set']['dread']['pooled']
    assert p['delta'] == 0 and p['ci95'] == [0.0, 0.0] and rep['oof']['filter'] == 'IDENTICAL_TO_BASELINE'
    assert 'no seed/split replicate (single run per member)' not in rep['verdict']['caveats']


def test_bootstrap_is_deterministic(world):
    w = world
    m = w.member('GOOD', noise=0.25)
    ctx = ctx_for(w)
    uids = sorted(ctx.population())
    Y, _ = ctx.labels(ctx.path(ctx.cfg['references']['dread']['path']), uids, 'rint', False)
    A = S.resolve_member(ctx, dict(oof=m['oof']))['pred'].loc[uids].to_numpy()
    B = S.resolve_member(ctx, dict(oof=w.base['oof']))['pred'].loc[uids].to_numpy()
    fa = np.array([ctx.population()[u] for u in uids])
    c = S.comp_index(ctx, uids)
    assert len(c) == 99
    one, two = S.score_block(Y, A, B, fa, c, 40, 7), S.score_block(Y, A, B, fa, c, 40, 7)
    assert one['pooled']['ci95'] == two['pooled']['ci95']
    assert one['pooled']['ci95'] != S.score_block(Y, A, B, fa, c, 40, 8)['pooled']['ci95']


def test_direct_overlap_blocks(world):
    w = world
    proto = w.make_protocol('LEAKY_PROTOCOL', leak_fold=1)
    m = w.member('LEAK', protocol=proto)
    rel = w.release('leak', {'leak': (m, 1.0)})
    rep = S.run_many(ctx_for(w), [spec('leak', [dict(name='leak', arm='leak', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    assert rep['lineage']['status'] == 'FAIL_DIRECT' and rep['lineage']['members'][0]['direct_overlap'] == 2
    assert rep['oof']['eligibility'].startswith('INVALID')
    assert rep['verdict']['verdict'] == 'BLOCK'


def test_teacher_lineage_flags_distilled_student(world):
    w = world
    w.distill_labels(w.base)
    m = w.member('STUDENT', label_set='distill_x', label_hash='sem-dx')
    rel = w.release('student', {'student': (m, 1.0)})
    sp = spec('student', [dict(name='student', arm='student', oof=m['oof'])], rel)
    rep = S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]
    mem = rep['lineage']['members'][0]
    assert rep['lineage']['status'] == 'FLAG_INDIRECT'
    assert mem['direct_overlap'] == 0 and mem['indirect_overlap'] == 100 and mem['indirect_share'] == 1.0
    assert mem['teachers'][0]['status'] == 'PASS' and 'teacher receipt' in mem['label_provenance']
    assert rep['oof']['eligibility'].startswith('INELIGIBLE') and 'distill_x' in rep['oof']['label_sets_not_scored']
    assert rep['verdict']['verdict'] == 'SUBMIT-ELIGIBLE'
    assert any('OOF ineligible' in c for c in rep['verdict']['caveats'])
    sp_oof = spec('student2', [dict(name='student', arm='student', oof=m['oof'])], rel, basis='oof')
    assert S.run_many(ctx_for(w), [sp_oof], NO_SUBS, write=False)[0]['verdict']['verdict'] == 'BLOCK'


def test_spec_named_ancestor_uid_list(world):
    w = world
    m = w.member('ANC')
    val0 = [u for u in w.uids if w.fold[u] == 0][:4]
    wjson(w.root / 'stacker_fit.json', val0)
    rel = w.release('anc', {'anc': (m, 1.0)})
    sp = spec('anc', [dict(name='anc', arm='anc', oof=m['oof'], ancestors=[dict(name='stacker', kind='stacker', train_uids_file='stacker_fit.json')])], rel)
    rep = S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]
    assert rep['lineage']['status'] == 'FLAG_INDIRECT' and rep['lineage']['members'][0]['indirect_overlap'] == 4


def test_provenance_conflict_holds(world):
    w = world
    m = w.member('BADPROV', prov_bad_fold=2)
    rel = w.release('badprov', {'badprov': (m, 1.0)})
    rep = S.run_many(ctx_for(w), [spec('badprov', [dict(name='badprov', arm='badprov', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    assert rep['lineage']['status'] == 'INCOMPLETE' and rep['verdict']['verdict'] == 'HOLD'
    assert any('provenance train' in i for i in rep['lineage']['members'][0]['issues'])


def test_engineering_failures(world):
    w = world
    m = w.member('ENG')
    ctx = ctx_for(w)
    sp = lambda name, rel: spec(name, [dict(name='eng', arm='eng', oof=m['oof'])], rel)
    tampered = S.run_many(ctx, [sp('t1', w.release('t1', {'eng': (m, 1.0)}, tamper=True))], NO_SUBS, write=False)[0]
    assert tampered['engineering']['csv_matches_receipt'] is False and tampered['verdict']['verdict'] == 'BLOCK'
    wrong = S.run_many(ctx, [sp('t2', w.release('t2', {'eng': (m, 1.0)}, bad_ckpt=True))], NO_SUBS, write=False)[0]
    assert wrong['engineering']['arms']['eng']['binding'] == 'conflicting_evidence' and wrong['verdict']['verdict'] == 'BLOCK'
    failed = S.run_many(ctx, [sp('t3', w.release('t3', {'eng': (m, 1.0)}, status='FAIL'))], NO_SUBS, write=False)[0]
    assert failed['engineering']['engineering_pass'] is False and failed['verdict']['verdict'] == 'BLOCK'
    slow = S.run_many(ctx, [sp('t4', w.release('t4', {'eng': (m, 1.0)}, projected=5000.0))], NO_SUBS, write=False)[0]
    assert slow['engineering']['runtime']['within_budget'] is False and slow['verdict']['verdict'] == 'BLOCK'
    missing = S.run_many(ctx, [sp('t5', dict(out_dir='.local/rel/out-none'))], NO_SUBS, write=False)[0]
    assert missing['engineering']['status'] == 'MISSING' and missing['verdict']['verdict'] == 'HOLD'


def test_blend_weights_and_receipt_binding(world):
    w = world
    a, b = w.member('A', noise=0.25), w.member('B', noise=0.25)
    rel = w.release('ab', {'a': (a, 0.7), 'b': (b, 0.3)})
    sp = spec('ab', [dict(name='a', arm='a', oof=a['oof'], weight=0.7), dict(name='b', arm='b', oof=b['oof'], weight=0.3)], rel, weights_selected_on_oof=True)
    rep = S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]
    assert rep['engineering']['reduction_state'] == 'expected_reported'
    assert 'weights a 0.700, b 0.300' in rep['oof']['candidate_formula']
    assert 'blend weight selected on the same OOF' in rep['verdict']['caveats']
    rel2 = w.release('ab2', {'a': (a, 0.7), 'b': (b, 0.3)}, weights_reported={'a': 0.5, 'b': 0.5})
    rep2 = S.run_many(ctx_for(w), [dict(sp, candidate='ab2', release=rel2)], NO_SUBS, write=False)[0]
    assert rep2['engineering']['reduction_state'] == 'changed_reported' and rep2['verdict']['verdict'] == 'BLOCK'


def test_decision_rules_measurement_only_and_scored_outcome(world, tmp_path):
    dec = dict(rules=[dict(on='delta', op='>=', value=0.009, reading='confirm'), dict(on='delta', op='<=', value=-0.005, reading='reject')],
               reference=dict(public=0.914), otherwise='neither')
    assert S.evaluate_rules(dec, 0.923) == 'confirm' and S.evaluate_rules(dec, 0.909) == 'reject' and S.evaluate_rules(dec, 0.912) == 'neither'
    assert S.evaluate_rules(dict(rules=RULES), 0.920) == 'transfer confirmed' and S.evaluate_rules(dict(rules=RULES), 0.914) == 'no transfer'
    w = world
    m = w.member('DESC')
    rel = w.release('desc', {'desc': (m, 1.0)})
    cache = w.root / 'subs.csv'
    pd.DataFrame([dict(ref='111', fileName='submission.csv', date='2026-10-07 00:01:00', description='d', status='SubmissionStatus.COMPLETE', publicScore='0.921', privateScore='')]).to_csv(cache, index=False)
    subs = S.load_submissions(cache)
    desc = S.run_many(ctx_for(w), [spec('desc', [dict(name='desc', arm='desc', oof=m['oof'])], rel, rules=None)], subs, write=False)[0]
    assert desc['decision']['pre_registered'] is False and desc['verdict']['verdict'] == 'HOLD'
    assert 'measurement-only' in desc['decision']['line']
    scored = S.run_many(ctx_for(w), [spec('scored', [dict(name='desc', arm='desc', oof=m['oof'])], rel, ref='111')], subs, write=False)[0]
    assert scored['decision']['state'] == 'SCORED' and scored['decision']['outcome'] == 'transfer confirmed'
    assert 'retrospective' in scored['verdict']['line']


def test_duplicate_of_scored_submission_blocks(world):
    w = world
    m = w.member('DUP')
    rel = w.release('dup', {'dup': (m, 1.0)})
    cache = w.root / 'subs.json'
    wjson(cache, [dict(ref='222', date='2026-10-07 01:00:00', status='SubmissionStatus.COMPLETE', public_score='0.914', description='x')])
    subs = S.load_submissions(cache)
    first = spec('first', [dict(name='dup', arm='dup', oof=m['oof'])], rel, ref='222')
    again = spec('again', [dict(name='dup', arm='dup', oof=m['oof'])], rel)
    reps = S.run_many(ctx_for(w), [first, again], subs, write=False)
    assert reps[0]['decision']['duplicate_of'] == []
    d = reps[1]['decision']['duplicate_of']
    assert d and d[0]['ref'] == '222' and set(d[0]['matched']) == {'members', 'served_checkpoints', 'public3_submission'}
    assert reps[1]['verdict']['verdict'] == 'BLOCK'
    nocache = S.run_many(ctx_for(w), [first, again], None, write=False)[1]
    assert nocache['decision']['duplicate_of'] == [] and nocache['decision']['duplicates_unverified'][0]['ref'] == '222'
    assert nocache['verdict']['verdict'] == 'HOLD' and any('possible duplicate not verifiable' in h for h in nocache['verdict']['holds'])


def test_seed_replicate_auto_detected(world):
    w = world
    m = w.member('REP42', noise=0.25)
    r = w.member('REP43', noise=0.25, extra_result=dict(seed42_protocol_sha256=sha(w.manifest)))
    rel = w.release('rep', {'rep': (m, 1.0)})
    rep = S.run_many(ctx_for(w), [spec('rep', [dict(name='rep', arm='rep', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    reps = rep['replication']
    assert reps['status'] == 'AVAILABLE' and reps['seed_replicates'] >= 1
    found = [x for x in reps['replicates'] if 'REP43' in x['oof']]
    assert found and found[0]['status'] == 'OK' and set(found[0]['delta_other_references']) == {'flight_v4', 'relabel_v1'}


def test_outputs_written_and_not_overwritten(world):
    w = world
    m = w.member('OUT')
    rel = w.release('out', {'out': (m, 1.0)})
    sp = spec('out', [dict(name='out', arm='out', oof=m['oof'])], rel)
    ctx = ctx_for(w)
    S.run_many(ctx, [sp], None)
    d = w.root / E / 'SUBMISSION_SCREEN/out'
    rep = json.loads((d / 'screen.json').read_text(encoding='utf-8'))
    assert rep['schema'] == S.SCHEMA and (d / 'screen.md').read_text(encoding='utf-8').startswith('# Submission screen: out')
    assert json.loads((d / 'spec.json').read_text(encoding='utf-8'))['candidate'] == 'out'
    with pytest.raises(FileExistsError):
        S.run_many(ctx, [sp], None)
    S.run_many(ctx, [sp], None, overwrite=True)
    with pytest.raises(ValueError):
        S.screen(ctx, dict(sp, candidate='../escape'))


def test_cli_member_parsing_and_end_to_end(world):
    w = world
    m = w.member('CLI', noise=0.2)
    rel = w.release('cli', {'cli': (m, 1.0)})
    assert S.parse_member('a=C:/x/oof_k36.csv:0.3') == dict(name='a', oof='C:/x/oof_k36.csv', weight=0.3, arm='a')
    assert S.parse_member('a=rel/oof.csv')['weight'] == 1.0
    assert S.parse_rule('delta<=-0.005: reject')['value'] == -0.005
    rc = S.main(['--root', str(w.root), '--draws', '20', '--candidate', 'cli', '--member', f'cli={m["oof"]}', '--release-out', rel['out_dir'],
                 '--plan-item', 'H-X', '--rule', 'public>=0.920:confirmed', '--rule', 'public<=0.914:no transfer', '--registered-at', REG])
    assert rc == 0
    rep = json.loads((w.root / E / 'SUBMISSION_SCREEN/cli/screen.json').read_text(encoding='utf-8'))
    assert rep['decision']['pre_registered'] is True and rep['engineering']['engineering_pass'] is True


# ----------------------------------------------------------------------------- diagnostic quarantine and EXP116 receipts

def quarantine_member(w, m, where):
    path = w.root / m['oof']
    if where == 'fold':
        res = json.loads((path.parent / 'result.json').read_text(encoding='utf-8'))
        path = w.root / res['sources']['0']['oof_k36']
    if where in ('marker', 'fold'):
        S.DQ.write_marker(path.parent, S.DQ.marker_body('synthetic', 'synthetic teacher ancestry', ['D-B-024']))
    elif where == 'result':
        rp = path.parent / 'result.json'
        rec = json.loads(rp.read_text(encoding='utf-8'))
        wjson(rp, dict(rec, decision_eligible=False, leak='synthetic teacher ancestry'))
    else:
        wjson(path.parent / 'manifest.json', dict(evaluation=dict(decision_eligible='false')))
    return path


def guard_quarantined_predictions(monkeypatch, w):
    original = S.load_pred

    def guarded(path):
        assert S.DQ.quarantine_reason(path, root=w.root) is None, f'quarantined predictions read: {path}'
        return original(path)

    monkeypatch.setattr(S, 'load_pred', guarded)


@pytest.mark.parametrize('role', ['member', 'default_baseline', 'spec_baseline'])
@pytest.mark.parametrize('where', ['marker', 'result', 'manifest', 'fold'])
@pytest.mark.parametrize('basis', ['public', 'oof', 'undeclared'])
def test_quarantined_oof_disables_decision_evidence(world, monkeypatch, role, where, basis):
    w = world
    m = w.member('QUARANTINE', noise=0.1)
    rel = w.release('quarantine', {'q': (m, 1.0)})
    sp = spec('quarantine', [dict(name='q', arm='q', oof=m['oof'])], rel, basis='public' if basis == 'undeclared' else basis)
    if basis == 'undeclared':
        del sp['decision']['basis']
    target = m if role == 'member' else w.base
    if role == 'spec_baseline':
        target = w.member('CUSTOM_BASELINE')
        sp['baseline'] = dict(name='custom baseline', oof=target['oof'])
    quarantine_member(w, target, where)
    guard_quarantined_predictions(monkeypatch, w)
    rep = S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]
    assert rep['oof']['available'] is False and rep['oof']['by_label_set'] == {}
    assert rep['oof']['direction']['consistent'] is None and rep['oof']['gate_failures']
    assert rep['replication']['status'] == 'NOT_AVAILABLE' and rep['replication']['candidate_delta_dread'] is None
    assert rep['verdict']['verdict'] == ('PUBLIC-ONLY' if basis == 'public' else 'BLOCK')
    md = S.render_md(rep)
    assert 'decision_eligible false' in md and 'quarantine gate failed' in md
    if where != 'manifest':
        assert 'synthetic teacher ancestry' in md


def test_cli_baseline_oof_obeys_quarantine(world, monkeypatch):
    w = world
    m = w.member('CLI_QUARANTINE', noise=0.1)
    rel = w.release('cli_quarantine', {'q': (m, 1.0)})
    quarantine_member(w, w.base, 'marker')
    guard_quarantined_predictions(monkeypatch, w)
    assert S.main(['--root', str(w.root), '--draws', '2', '--candidate', 'cli_quarantine', '--member', f'q={m["oof"]}',
                   '--baseline-oof', w.base['oof'], '--release-out', rel['out_dir'], '--plan-item', 'H-X',
                   '--rule', 'public>=0.920:confirmed', '--registered-at', REG, '--basis', 'public', '--submissions', str(empty_cache(w))]) == 0
    report = json.loads((w.root / E / 'SUBMISSION_SCREEN/cli_quarantine/screen.json').read_text(encoding='utf-8'))
    assert report['verdict']['verdict'] == 'PUBLIC-ONLY' and report['oof']['gate_failures']


def test_unreadable_eligibility_record_reports_failure_without_loading_oof(world, monkeypatch):
    w = world
    m = w.member('UNREADABLE_RESULT')
    (w.root / m['oof']).with_name('result.json').write_text('{broken', encoding='utf-8')
    guard_quarantined_predictions(monkeypatch, w)
    rel = w.release('unreadable', {'q': (m, 1.0)})
    rep = S.run_many(ctx_for(w), [spec('unreadable', [dict(name='q', arm='q', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    assert rep['verdict']['verdict'] == 'BLOCK' and rep['oof']['available'] is False
    assert 'eligibility is unverifiable' in S.render_md(rep)


@pytest.mark.parametrize('source', ['auto', 'declared'])
@pytest.mark.parametrize('where', ['marker', 'result', 'manifest', 'fold', 'teacher'])
def test_quarantined_replicates_are_skipped_with_reason(world, monkeypatch, source, where):
    """astra re-review 2026-10-08 A3: a replicate whose own directory is unmarked but whose source fold directory ('fold') or label
    teacher ('teacher') is quarantined is skipped before scoring, with the reason."""
    w = world
    m = w.member('REP_CLEAN', noise=0.1)
    if where == 'teacher':
        teacher = w.member('REP_TEACHER')
        w.distill_labels(teacher)
        r = w.member('REP_QUARANTINED', label_set='distill_x', label_hash='sem-dx', extra_result=dict(seed42_protocol_sha256=sha(w.manifest)))
        quarantine_member(w, teacher, 'marker')
    else:
        r = w.member('REP_QUARANTINED', extra_result=dict(seed42_protocol_sha256=sha(w.manifest)))
        quarantine_member(w, r, where)
    guard_quarantined_predictions(monkeypatch, w)
    ms = dict(name='rep', arm='rep', oof=m['oof'])
    if source == 'declared':
        ms['replicates'] = [dict(oof=r['oof'], kind='seed', label='diagnostic replicate')]
    ctx = ctx_for(w)
    if where in ('fold', 'teacher'):
        assert S.DQ.quarantine_reason(w.root / r['oof'], root=w.root) is None  # the pooled replicate itself is unmarked
    else:
        assert all(p.parent.name != 'REP_QUARANTINED_OOF' for p, _ in ctx.oof_results())
    rel = w.release('rep_clean', {'rep': (m, 1.0)})
    rep = S.run_many(ctx, [spec('rep_clean', [ms], rel)], NO_SUBS, write=False)[0]
    skips = [x for x in rep['replication']['replicates'] if x['oof'] == r['oof']]
    assert skips and all(x['status'] == 'SKIPPED_QUARANTINED' and 'decision_eligible false' in x['reason'] for x in skips)
    if where == 'fold':
        assert all('fold 0 OOF' in x['reason'] for x in skips)
    if where == 'teacher':
        assert all('teacher t1' in x['reason'] for x in skips)
    assert all('delta' not in x and 'ci95' not in x for x in skips)
    assert rep['replication']['seed_replicates'] == 0 and rep['verdict']['verdict'] == 'SUBMIT-ELIGIBLE'
    assert 'SKIPPED_QUARANTINED' in S.render_md(rep) and 'decision_eligible false' in S.render_md(rep)


def teacher_fill_student(w):
    rel = 'storage/labels/derived_teacher_fill/teacher_fill_v1_f0.csv'
    w.write_labels(rel, np.where(w.y == 1, 0.75, 0.15))
    w.register('teacher_fill_x', rel, 'sem-fill')
    second = w.member('SECOND_TEACHER')
    rc = w.root / 'storage/labels/derived_teacher_fill/receipt.json'
    wjson(rc, dict(outputs={'teacher_fill_v1.csv': dict(sha256='0' * 64),
                            Path(rel).name: dict(sha256=sha(w.root / rel))},
                  teachers={name: dict(oof=t['oof'], oof_sha256=sha(w.root / t['oof']))
                            for name, t in [('coatnet5', w.base), ('radr50x5', second)]}))
    m = w.member('TEACHER_FILL_STUDENT', label_set='teacher_fill_x', label_hash='sem-fill')
    release = w.release('teacher_fill', {'student': (m, 1.0)})
    return spec('teacher_fill', [dict(name='student', arm='student', oof=m['oof'])], release), rc


@pytest.mark.parametrize('quarantined', [False, True])
def test_exp116_per_file_receipt_resolves_teachers(world, monkeypatch, quarantined):
    sp, _ = teacher_fill_student(world)
    if quarantined:
        quarantine_member(world, sp['members'][0], 'marker')
    guard_quarantined_predictions(monkeypatch, world)
    rep = S.run_many(ctx_for(world), [sp], NO_SUBS, write=False)[0]
    mem = rep['lineage']['members'][0]
    assert mem['status'] == 'FLAG_INDIRECT' and mem['indirect_overlap'] == 100
    assert mem['teacher_binding'] == 'RECEIPT_BOUND' and mem['gate_failures'] == []
    assert {t['name'] for t in mem['teachers']} == {'coatnet5', 'radr50x5'}
    assert all(t['status'] == 'PASS' and t['teacher_source'] == 'receipt' for t in mem['teachers'])
    assert rep['verdict']['verdict'] == ('PUBLIC-ONLY' if quarantined else 'SUBMIT-ELIGIBLE')


@pytest.mark.parametrize('fault', ['missing_output', 'wrong_output', 'other_file_only', 'malformed_outputs', 'empty_teachers',
                                  'missing_oof', 'missing_sha', 'wrong_teacher_sha', 'missing_teacher_file',
                                  'conflicting_output', 'conflicting_teacher_alias', 'empty_spec_teachers'])
def test_exp116_receipt_binding_stays_fail_closed(world, fault):
    sp, rc = teacher_fill_student(world)
    rec = json.loads(rc.read_text(encoding='utf-8'))
    entry = rec['outputs']['teacher_fill_v1_f0.csv']
    teacher = rec['teachers']['coatnet5']
    if fault == 'missing_output':
        del entry['sha256']
    elif fault == 'wrong_output':
        entry['sha256'] = 'f' * 64
    elif fault == 'other_file_only':
        rec['outputs'] = {'teacher_fill_v1.csv': entry}  # same hash under another filename is insufficient
    elif fault == 'malformed_outputs':
        rec['outputs'] = [entry]
    elif fault == 'empty_teachers':
        rec['teachers'] = {}
    elif fault == 'missing_oof':
        del teacher['oof']
    elif fault == 'missing_sha':
        del teacher['oof_sha256']
    elif fault == 'wrong_teacher_sha':
        teacher['oof_sha256'] = 'f' * 64
    elif fault == 'missing_teacher_file':
        teacher['oof'] = 'missing/oof_k36.csv'
    elif fault == 'conflicting_output':
        rec['output'] = dict(sha256='f' * 64)
    elif fault == 'conflicting_teacher_alias':
        teacher.update(path=teacher['oof'], sha256='f' * 64)
    else:
        sp['members'][0]['teachers'] = []
    wjson(rc, rec)
    rep = S.run_many(ctx_for(world), [sp], NO_SUBS, write=False)[0]
    assert rep['lineage']['status'] == 'FAIL_LINEAGE' and rep['verdict']['verdict'] == 'BLOCK'
    assert rep['lineage']['members'][0]['gate_failures']


@pytest.mark.parametrize('receipt_schema', ['legacy', 'exp116'])
@pytest.mark.parametrize('where', ['marker', 'result', 'manifest', 'fold'])
def test_quarantined_receipt_teacher_fails_lineage(world, monkeypatch, receipt_schema, where):
    w = world
    if receipt_schema == 'exp116':
        sp, _ = teacher_fill_student(w)
    else:
        w.distill_labels(w.base)
        m = w.member('DISTILLED', label_set='distill_x', label_hash='sem-dx')
        sp = spec('distilled', [dict(name='student', arm='student', oof=m['oof'])], w.release('distilled', {'student': (m, 1.0)}))
    quarantine_member(w, w.base, where)
    guard_quarantined_predictions(monkeypatch, w)
    rep = S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]
    mem = rep['lineage']['members'][0]
    assert rep['verdict']['verdict'] == 'BLOCK' and mem['status'] == 'FAIL_LINEAGE'
    assert any(t['status'] == 'FAIL_LINEAGE' for t in mem['teachers'])
    assert any('quarantined' in g for g in mem['gate_failures'])
    assert rep['oof']['available'] is False and rep['replication']['candidate_delta_dread'] is None


def test_quarantined_baseline_teacher_fails_lineage(world, monkeypatch):
    w = world
    student, _ = teacher_fill_student(w)
    m = w.member('CLEAN_CANDIDATE', noise=0.1)
    rel = w.release('clean_candidate', {'clean': (m, 1.0)})
    sp = spec('clean_candidate', [dict(name='clean', arm='clean', oof=m['oof'])], rel,
              baseline=dict(name='distilled baseline', oof=student['members'][0]['oof']))
    quarantine_member(w, w.base, 'marker')
    guard_quarantined_predictions(monkeypatch, w)
    rep = S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]
    assert rep['lineage']['status'] == 'PASS'
    assert rep['oof']['baseline']['lineage']['status'] == 'FAIL_LINEAGE'
    assert rep['verdict']['verdict'] == 'BLOCK' and 'baseline lineage gate failed' in S.render_md(rep)


# ----------------------------------------------------------------------------- fail-closed admission gates (H-B-35)

def test_parse_when_requires_a_zone():
    assert S.parse_when('2026-10-07 12:20 KST') == S.parse_when('2026-10-07T03:20:00Z') == S.parse_when('2026-10-07 03:20:00+00:00')
    assert S.parse_when('2026-10-07T12:20:00+09:00') == S.parse_when('2026-10-07 12:20 KST')
    assert S.parse_when('2026-10-07 12:20') is None and S.parse_when('2026-10-07 12:20', dt.timezone.utc) is not None
    assert S.parse_when('2026-10-07 00:01:34.877000', dt.timezone.utc).minute == 1
    assert S.parse_when('soon') is None and S.parse_when(None) is None and S.parse_when('') is None


def test_registration_gate_fails_closed(world):
    w = world
    m = w.member('REG', noise=0.2)
    rel = w.release('reg', {'reg': (m, 1.0)})
    ctx = ctx_for(w)

    def mk(name, **kw):
        return spec(name, [dict(name='reg', arm='reg', oof=m['oof'])], rel, **kw)
    ok = S.run_many(ctx, [mk('ok', registered_at='2026-10-07T12:20:00+09:00')], NO_SUBS, write=False)[0]
    assert ok['decision']['pre_registered'] is True and ok['decision']['registration']['valid'] is True
    assert ok['decision']['registration']['registered_at_utc'] == '2026-10-07T03:20:00+00:00' and ok['verdict']['verdict'] == 'SUBMIT-ELIGIBLE'
    for bad, why in [(None, 'missing'), ('2099-01-01 00:00 KST', 'not earlier than the screen run'), ('2026-10-07 12:20', 'no time zone'),
                     ('after the score', 'not a parseable time')]:
        r = S.run_many(ctx, [mk('bad', registered_at=bad)], NO_SUBS, write=False)[0]
        d = r['decision']
        assert d['pre_registered'] is False and d['pre_registration_claimed'] is True and d['state'] == 'REGISTRATION_NOT_VERIFIED'
        assert why in d['registration']['reason'] and 'registration NOT VERIFIED' in d['line']
        assert r['verdict']['verdict'] == 'HOLD' and any('not admitted as pre-registered' in h for h in r['verdict']['holds'])
        assert 'registration gate: FAIL' in S.render_md(r)
    # ordered against the cached submission time (Kaggle CLI dates are UTC; the score is recorded after the submission)
    cache = w.root / 'subs_reg.csv'
    pd.DataFrame([dict(ref='333', date='2026-10-07 03:00:00', description='x', status='SubmissionStatus.COMPLETE', publicScore='0.915')]).to_csv(cache, index=False)
    subs = S.load_submissions(cache)
    before = S.run_many(ctx, [mk('before', ref='333', registered_at='2026-10-07 11:59 KST')], subs, write=False)[0]
    after = S.run_many(ctx, [mk('after', ref='333', registered_at='2026-10-07 12:01 KST')], subs, write=False)[0]
    assert before['decision']['pre_registered'] is True and before['decision']['state'] == 'SCORED'
    assert after['decision']['pre_registered'] is False and 'not earlier than submission 333' in after['decision']['registration']['reason']
    assert after['verdict']['verdict'] == 'HOLD'
    # a submission_ref missing from the cache cannot be ordered before its score: REGISTRATION_NOT_VERIFIED (HOLD), not a caveat
    pend = S.run_many(ctx, [mk('pend', ref='999')], subs, write=False)[0]
    d = pend['decision']
    assert d['pre_registered'] is False and d['registration_status'] == 'REGISTRATION_NOT_VERIFIED' and d['registration']['valid'] is False
    assert 'submission 999 is not in the cached submissions list' in d['registration']['reason'] and pend['verdict']['verdict'] == 'HOLD'
    assert any('REGISTRATION_NOT_VERIFIED' in h for h in pend['verdict']['holds']) and not any('screen run only' in c for c in pend['verdict']['caveats'])
    nocache = S.run_many(ctx, [mk('nocache', ref='333')], None, write=False)[0]
    assert nocache['decision']['pre_registered'] is False and 'no submissions cache given' in nocache['decision']['registration']['reason']
    assert nocache['verdict']['verdict'] == 'HOLD'
    # not yet submitted (submission_ref null): ordered against the screen run (the run-time check stays)
    fresh = S.run_many(ctx, [mk('fresh')], subs, write=False)[0]
    assert fresh['decision']['pre_registered'] is True and fresh['decision']['registration_status'] == 'VERIFIED'
    assert fresh['decision']['registration']['score_order'] == 'no submission yet: ordered against the screen run'
    # a control role needs a registered_at too
    ctl = dict(mk('ctl', rules=None), decision=dict(plan_item='H-X', role='control', controls_for=['H-Y test']))
    assert S.run_many(ctx, [ctl], NO_SUBS, write=False)[0]['decision']['pre_registered'] is False
    ctl['decision']['registered_at'] = '2026-10-06 23:00 UTC'
    assert S.run_many(ctx, [ctl], NO_SUBS, write=False)[0]['decision']['pre_registered'] is True


def test_teacher_receipt_hash_mismatch_fails(world):
    w = world
    w.distill_labels(w.base)
    rc = w.root / 'storage/labels/dx/receipt.json'
    r = json.loads(rc.read_text(encoding='utf-8'))
    r['output']['sha256'] = '0' * 64
    wjson(rc, r)
    m = w.member('STUDENTX', label_set='distill_x', label_hash='sem-dx')
    rel = w.release('studentx', {'studentx': (m, 1.0)})
    rep = S.run_many(ctx_for(w), [spec('studentx', [dict(name='studentx', arm='studentx', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    mem = rep['lineage']['members'][0]
    assert rep['lineage']['status'] == 'FAIL_LINEAGE' and mem['teachers'] == []
    assert any('differs from the label file sha' in g for g in mem['gate_failures']) and 'does not bind' in mem['label_provenance']
    assert rep['oof']['eligibility'].startswith('INVALID') and rep['verdict']['verdict'] == 'BLOCK'
    assert any('lineage gate failed' in b for b in rep['verdict']['blockers'])
    rc.write_text('{not json', encoding='utf-8')  # an unreadable receipt is not "no teachers"
    again = S.run_many(ctx_for(w), [spec('studentx', [dict(name='studentx', arm='studentx', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    assert again['lineage']['status'] == 'FAIL_LINEAGE' and again['verdict']['verdict'] == 'BLOCK'


def test_explicit_empty_teacher_list_only_on_known_label_sets(world):
    w = world
    m = w.member('KNOWN', noise=0.2)
    rel = w.release('known', {'known': (m, 1.0)})
    ok = S.run_many(ctx_for(w), [spec('known', [dict(name='known', arm='known', oof=m['oof'], teachers=[])], rel)], NO_SUBS, write=False)[0]
    assert ok['lineage']['status'] == 'PASS' and 'known non-distilled label set dread' in ok['lineage']['members'][0]['label_provenance']
    assert ok['verdict']['verdict'] == 'SUBMIT-ELIGIBLE'
    w.distill_labels(w.base)
    st = w.member('HIDDEN', label_set='distill_x', label_hash='sem-dx')
    rel2 = w.release('hidden', {'hidden': (st, 1.0)})
    bad = S.run_many(ctx_for(w), [spec('hidden', [dict(name='hidden', arm='hidden', oof=st['oof'], teachers=[])], rel2)], NO_SUBS, write=False)[0]
    mem = bad['lineage']['members'][0]
    # the receipt is checked independently of the spec override: its teacher is still traced (full exposure), and the override fails
    assert bad['lineage']['status'] == 'FAIL_LINEAGE' and mem['indirect_overlap'] == 100 and mem['teacher_binding'] == 'RECEIPT_BOUND'
    assert any('not a known non-distilled label set' in g for g in mem['gate_failures']) and bad['verdict']['verdict'] == 'BLOCK'
    # a teacher receipt beside a known label file contradicts teachers: []
    wjson(w.root / 'storage/labels/derived_relabel/receipt.json', dict(teachers={'t': dict(path=w.base['oof'])}))
    rl = w.member('RL', label_set='relabel_v1', label_hash='sem-relabel')
    rel3 = w.release('rl', {'rl': (rl, 1.0)})
    contra = S.run_many(ctx_for(w), [spec('rl', [dict(name='rl', arm='rl', oof=rl['oof'], teachers=[])], rel3)], NO_SUBS, write=False)[0]
    assert contra['lineage']['status'] == 'FAIL_LINEAGE' and any('records teachers' in g for g in contra['lineage']['members'][0]['gate_failures'])


def test_missing_fold_run_or_pooled_oof_fails(world):
    w = world
    m = w.member('NOFOLD', noise=0.2)
    (w.root / E / 'NOFOLD_F3/oof_k36.csv').unlink()
    rel = w.release('nofold', {'nofold': (m, 1.0)})
    rep = S.run_many(ctx_for(w), [spec('nofold', [dict(name='nofold', arm='nofold', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    mem = rep['lineage']['members'][0]
    assert rep['lineage']['status'] == 'FAIL_LINEAGE' and any(g.startswith('fold 3: fold-run OOF file not available') for g in mem['gate_failures'])
    assert rep['oof']['eligibility'].startswith('INVALID') and rep['verdict']['verdict'] == 'BLOCK' and 'fold 3' in rep['verdict']['blockers'][0]
    gone = S.run_many(ctx_for(w), [spec('gone', [dict(name='gone', arm='nofold', oof=f'{E}/NOPE/oof_k36.csv')], rel)], NO_SUBS, write=False)[0]
    assert gone['lineage']['status'] == 'FAIL_LINEAGE' and gone['oof']['filter'] == 'NOT_AVAILABLE' and gone['verdict']['verdict'] == 'BLOCK'


def test_declared_no_oof_reads_public_only(world):
    w = world
    ck = hashlib.sha256(b'all-data').hexdigest()
    rel = w.release('alld', {'alld': (dict(ckpts=[ck]), 1.0)})
    mem = dict(name='alld', arm='alld', oof=None, oof_reason='all-data model: no held-out folds', checkpoint_sha256=ck, weight=1.0)
    rep = S.run_many(ctx_for(w), [spec('alld', [mem], rel)], NO_SUBS, write=False)[0]
    assert rep['lineage']['status'] == 'NOT_AVAILABLE' and rep['oof']['available'] is False and rep['oof']['filter'] == 'NOT_AVAILABLE'
    assert rep['oof']['by_label_set'] == {} and 'NOT AVAILABLE' in rep['oof']['direction']['line']
    assert rep['replication']['status'] == 'NOT_AVAILABLE' and rep['decision']['fingerprints']['members'] is None
    eng = rep['engineering']
    assert eng['arms']['alld']['binding'] == 'declared_match' and eng['engineering_pass'] is True and eng['reduction_state'] == 'expected_reported'
    assert rep['verdict']['verdict'] == 'PUBLIC-ONLY' and 'OOF NOT AVAILABLE' in rep['verdict']['line'] and 'PUBLIC-ONLY' in rep['verdict']['policy']
    assert rep['lineage']['gold58']['absent_from_training'] is None  # nothing to check without training UID lists
    md = S.render_md(rep)
    assert 'every OOF line is NOT AVAILABLE' in md and 'no OOF (all-data model' in md and 'not checked: a member declares no OOF' in md
    assert S.run_many(ctx_for(w), [spec('alld2', [mem], rel, basis='oof')], NO_SUBS, write=False)[0]['verdict']['verdict'] == 'BLOCK'
    nock = {k: v for k, v in mem.items() if k != 'checkpoint_sha256'}
    r3 = S.run_many(ctx_for(w), [spec('alld3', [nock], rel)], NO_SUBS, write=False)[0]
    assert r3['verdict']['verdict'] == 'HOLD' and 'checkpoint_sha256' in ' '.join(r3['engineering']['problems'])
    noreason = {k: v for k, v in mem.items() if k != 'oof_reason'}
    r4 = S.run_many(ctx_for(w), [spec('alld4', [noreason], rel)], NO_SUBS, write=False)[0]
    assert r4['lineage']['status'] == 'FAIL_LINEAGE' and r4['verdict']['verdict'] == 'BLOCK' and 'oof_reason' in r4['verdict']['blockers'][0]


def test_swa16_shaped_spec_gets_an_explicit_verdict(world):
    """The H-B-33 release script writes members=[{name, arm, oof: None, weight}] (no oof_reason) and passes no submissions cache."""
    w = world
    ck = hashlib.sha256(b'swa').hexdigest()
    rel = w.release('swa16', {'swa16': (dict(ckpts=[ck]), 1.0)})
    sp = dict(candidate='swa16', title='H-B-33: all-data CoAtNet SWA 13/15/16, K36 (release v5)',
              members=[dict(name='swa16', arm='swa16', oof=None, weight=1.0)], release=dict(rel, kernel_dir='.local/rel/swa16'), submission_ref=None,
              decision=dict(plan_item='H-B-33', role='test', basis='public', registered_at='2026-10-07 17:15 KST', text='t',
                            reference=dict(name='coat5', ref='56893801', public=0.914), rules=RULES, otherwise='inconclusive'))
    p = w.root / E / 'SUBMISSION_SCREEN/spec_swa16.json'
    wjson(p, sp)
    assert S.main(['--root', str(w.root), '--draws', '20', '--spec', str(p), '--overwrite']) == 0
    out = w.root / E / 'SUBMISSION_SCREEN/swa16'
    rep = json.loads((out / 'screen.json').read_text(encoding='utf-8'))
    assert rep['verdict']['verdict'] == 'BLOCK' and rep['lineage']['status'] == 'FAIL_LINEAGE' and rep['oof']['filter'] == 'NOT_AVAILABLE'
    assert any('oof_reason' in b for b in rep['verdict']['blockers']) and rep['decision']['pre_registered'] is True
    assert (out / 'screen.md').read_text(encoding='utf-8').startswith('# Submission screen: swa16')
    sp['members'][0].update(oof_reason='all-data model: no held-out folds', checkpoint_sha256=ck)
    wjson(p, sp)
    assert S.main(['--root', str(w.root), '--draws', '20', '--spec', str(p), '--overwrite']) == 0
    # without a submissions cache the registry cannot be reconciled: HOLD (astra re-review 2026-10-08 A1)
    nocache = json.loads((out / 'screen.json').read_text(encoding='utf-8'))
    assert nocache['verdict']['verdict'] == 'HOLD' and any(h.startswith('duplicate coverage incomplete: no submissions cache') for h in nocache['verdict']['holds'])
    assert S.main(['--root', str(w.root), '--draws', '20', '--spec', str(p), '--overwrite', '--submissions', str(empty_cache(w))]) == 0
    assert json.loads((out / 'screen.json').read_text(encoding='utf-8'))['verdict']['verdict'] == 'PUBLIC-ONLY'


def test_public3_hash_alone_is_a_warning_not_a_duplicate(world):
    w = world
    a, b = w.member('PA', noise=0.2), w.member('PB', noise=0.22)
    ra, rb = w.release('pa', {'pa': (a, 1.0)}), w.release('pb', {'pb': (b, 1.0)})
    assert sha(w.root / ra['out_dir'] / 'submission.csv') == sha(w.root / rb['out_dir'] / 'submission.csv')  # same three-study output
    cache = w.root / 'subs_p3.json'
    wjson(cache, [dict(ref='444', date='2026-10-07 01:00:00', status='SubmissionStatus.COMPLETE', public_score='0.914', description='x')])
    subs = S.load_submissions(cache)
    reps = S.run_many(ctx_for(w), [spec('pa', [dict(name='pa', arm='pa', oof=a['oof'])], ra, ref='444'),
                                   spec('pb', [dict(name='pb', arm='pb', oof=b['oof'])], rb)], subs, write=False)
    d = reps[1]['decision']
    assert d['duplicate_of'] == [] and d['public3_hash_only_matches'][0]['ref'] == '444' and d['public3_hash_only_matches'][0]['matched'] == ['public3_submission']
    assert reps[1]['verdict']['verdict'] == 'SUBMIT-ELIGIBLE' and any('not a duplicate' in c for c in reps[1]['verdict']['caveats']) and 'WARNING' in d['line']
    assert S.member_fingerprint([dict(weight=1.0)]) is None  # members without OOF never match each other


def test_release_manifest_match_is_a_duplicate(world):
    w = world
    a, b = w.member('MA', noise=0.2), w.member('MB', noise=0.22)
    ra = w.release('ma', {'ma': (a, 1.0)}, manifest_sha='a' * 64)
    rb = w.release('mb', {'mb': (b, 1.0)}, manifest_sha='a' * 64)
    cache = w.root / 'subs_man.json'
    wjson(cache, [dict(ref='555', date='2026-10-07 01:00:00', status='SubmissionStatus.COMPLETE', public_score='0.914', description='x')])
    subs = S.load_submissions(cache)
    reps = S.run_many(ctx_for(w), [spec('ma', [dict(name='ma', arm='ma', oof=a['oof'])], ra, ref='555'),
                                   spec('mb', [dict(name='mb', arm='mb', oof=b['oof'])], rb)], subs, write=False)
    d = reps[1]['decision']['duplicate_of']
    assert d and d[0]['ref'] == '555' and set(d[0]['matched']) == {'release_manifest', 'public3_submission'}
    assert reps[1]['verdict']['verdict'] == 'BLOCK' and reps[1]['decision']['public3_hash_only_matches'] == []


# ----------------------------------------------------------------------------- astra code review 2026-10-08: teacher lineage fails closed

def _unknown_labels(w, name='unk', receipt=None):
    """A label file that is neither on the approved non-distilled list nor (unless receipt) bound by a teacher receipt."""
    rel = f'storage/labels/{name}/labels.csv'
    w.write_labels(rel, np.where(w.y == 1, 0.8, 0.1))
    w.register(name, rel, f'sem-{name}')
    if receipt is not None:
        wjson(w.root / f'storage/labels/{name}/receipt.json', receipt(rel))
    return rel


def _screen_member(w, name, label_set, label_hash, write=False, **member_kw):
    m = w.member(name.upper(), label_set=label_set, label_hash=label_hash)
    rel = w.release(name, {name: (m, 1.0)})
    rep = S.run_many(ctx_for(w), [spec(name, [dict(name=name, arm=name, oof=m['oof'], **member_kw)], rel)], NO_SUBS, write=write)[0]
    return rep, rep['lineage']['members'][0], m


def test_unknown_label_file_without_receipt_fails_even_with_spec_teachers(world):
    w = world
    _unknown_labels(w)
    rep, mem, _ = _screen_member(w, 'unk1', 'unk', 'sem-unk')  # teachers omitted: formerly passed as "external labels"
    assert rep['lineage']['status'] == 'FAIL_LINEAGE' and rep['verdict']['verdict'] == 'BLOCK' and mem['teacher_binding'] == 'UNVERIFIED'
    assert any('has no teacher receipt beside it' in g for g in mem['gate_failures'])
    rep2, mem2, _ = _screen_member(w, 'unk2', 'unk', 'sem-unk', teachers=[dict(name='t', oof=w.base['oof'])])
    assert rep2['lineage']['status'] == 'FAIL_LINEAGE' and rep2['verdict']['verdict'] == 'BLOCK'
    assert any('spec-named teachers cannot replace the receipt' in g for g in mem2['gate_failures'])
    assert len(mem2['teachers']) == 1 and mem2['indirect_overlap'] == 100  # the spec teacher still adds its influence


def test_empty_receipt_teacher_mapping_fails(world):
    w = world
    _unknown_labels(w, 'emp', receipt=lambda rel: dict(teachers={}, output=dict(path=rel, sha256=sha(w.root / rel))))
    rep, mem, _ = _screen_member(w, 'emp', 'emp', 'sem-emp')
    assert rep['lineage']['status'] == 'FAIL_LINEAGE' and rep['verdict']['verdict'] == 'BLOCK'
    assert any('not a non-empty name -> OOF mapping' in g for g in mem['gate_failures']) and mem['teachers'] == []


def test_receipt_without_output_sha_fails(world):
    w = world
    _unknown_labels(w, 'noout', receipt=lambda rel: dict(teachers={'t1': dict(path=w.base['oof'], sha256=sha(w.root / w.base['oof']))}))
    rep, mem, _ = _screen_member(w, 'noout', 'noout', 'sem-noout')
    assert rep['lineage']['status'] == 'FAIL_LINEAGE' and any('records no output sha256' in g for g in mem['gate_failures'])


def test_spec_teachers_do_not_skip_the_receipt_hash_check(world):
    w = world
    w.distill_labels(w.base)
    rc = w.root / 'storage/labels/dx/receipt.json'
    r = json.loads(rc.read_text(encoding='utf-8'))
    r['output']['sha256'] = '0' * 64
    wjson(rc, r)
    rep, mem, _ = _screen_member(w, 'sover', 'distill_x', 'sem-dx', teachers=[dict(name='t1', oof=w.base['oof'])])
    assert rep['lineage']['status'] == 'FAIL_LINEAGE' and rep['verdict']['verdict'] == 'BLOCK'
    assert any('differs from the label file sha' in g for g in mem['gate_failures']) and len(mem['teachers']) == 1


def test_receipt_teacher_oof_must_be_bound(world):
    w = world
    w.distill_labels(w.base)
    rc = w.root / 'storage/labels/dx/receipt.json'
    r = json.loads(rc.read_text(encoding='utf-8'))
    r['teachers']['t1']['sha256'] = 'f' * 64  # the receipt names another file than the teacher OOF on disk
    wjson(rc, r)
    rep, mem, _ = _screen_member(w, 'tsha', 'distill_x', 'sem-dx')
    assert rep['lineage']['status'] == 'FAIL_LINEAGE' and rep['verdict']['verdict'] == 'BLOCK'
    assert any(g.startswith('teacher t1: ') and 'recorded by the referencing teacher receipt' in g for g in mem['gate_failures'])
    r['teachers']['t1'] = dict(path=w.base['oof'])  # no recorded sha256
    wjson(rc, r)
    rep2, mem2, _ = _screen_member(w, 'tnosha', 'distill_x', 'sem-dx')
    assert rep2['lineage']['status'] == 'FAIL_LINEAGE' and any('has no recorded OOF path and sha256' in g for g in mem2['gate_failures'])


def test_spec_label_file_override_cannot_replace_the_registered_file(world):
    w = world
    w.distill_labels(w.base)
    rep, mem, _ = _screen_member(w, 'lfo', 'distill_x', 'sem-dx', label_file='storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv')
    assert rep['lineage']['status'] == 'FAIL_LINEAGE' and any('is not the registered file' in g for g in mem['gate_failures'])


def test_approved_label_file_passes_and_records_binding(world):
    w = world
    rep, mem, _ = _screen_member(w, 'appr', 'dread', 'sem-dread')
    assert rep['lineage']['status'] == 'PASS' and mem['teacher_binding'] == 'APPROVED_NON_DISTILLED' and mem['gate_failures'] == []
    rep2, mem2, _ = _screen_member(w, 'rl', 'relabel_v1', 'sem-relabel')
    assert rep2['lineage']['status'] == 'PASS' and 'labeller synthetic LLM' in mem2['label_provenance']


# ----------------------------------------------------------------------------- astra code review 2026-10-08: persistent duplicate registry

def _cache(w, name, rows):
    p = w.root / name
    wjson(p, [dict(ref=r, date=d, status=f'SubmissionStatus.{st}', public_score=pub, description='x') for r, d, st, pub in rows])
    return S.load_submissions(p)


def test_single_spec_screen_detects_a_historical_duplicate(world):
    w = world
    m = w.member('HIST', noise=0.2)
    rel = w.release('hist', {'hist': (m, 1.0)})
    first = spec('first', [dict(name='hist', arm='hist', oof=m['oof'])], rel, ref='777')
    S.run_many(ctx_for(w), [first], NO_SUBS, write=True)  # an earlier screen run records its fingerprints
    reg = json.loads(w.registry.read_text(encoding='utf-8'))
    assert [e['candidate'] for e in reg['entries']] == ['first'] and reg['entries'][0]['submission_ref'] == '777'
    assert set(reg['entries'][0]['fingerprints']) == set(S.FP_KEYS) and b'\r\n' not in w.registry.read_bytes()
    assert not list(w.registry.parent.glob('.fingerprints.json.*.tmp'))
    again = spec('again', [dict(name='hist', arm='hist', oof=m['oof'])], rel)  # screened ALONE, in a later run
    scored = _cache(w, 's1.json', [('777', '2026-10-07 01:00:00', 'COMPLETE', '0.914')])
    rep = S.run_many(ctx_for(w), [again], scored, write=False)[0]
    d = rep['decision']
    assert d['duplicate_of'] and d['duplicate_of'][0]['ref'] == '777' and set(d['duplicate_of'][0]['matched']) >= {'members', 'served_checkpoints'}
    assert rep['verdict']['verdict'] == 'BLOCK' and d['duplicate_coverage']['available'] is True and d['duplicate_coverage']['registry_entries'] == 1
    # status unknown (no cache, or the ref missing from the cache, or pending) -> HOLD; a failed submission may be resubmitted
    for subs in (None, _cache(w, 's2.json', [('1', '2026-10-07 01:00:00', 'COMPLETE', '0.9')]),
                 _cache(w, 's3.json', [('777', '2026-10-07 01:00:00', 'PENDING', '')])):
        r = S.run_many(ctx_for(w), [again], subs, write=False)[0]
        assert r['decision']['duplicates_unverified'][0]['ref'] == '777' and r['verdict']['verdict'] == 'HOLD'
    err = S.run_many(ctx_for(w), [again], _cache(w, 's4.json', [('777', '2026-10-07 01:00:00', 'ERROR', '')]), write=False)[0]
    assert err['decision']['matches_errored'][0]['ref'] == '777' and err['verdict']['verdict'] == 'SUBMIT-ELIGIBLE'
    # the same submission screened again is not its own duplicate
    own = S.run_many(ctx_for(w), [dict(first, candidate='first_again')], scored, write=False)[0]
    assert own['decision']['duplicate_of'] == [] and own['decision']['state'] == 'SCORED'


def test_match_with_an_unsubmitted_screened_candidate_is_a_caveat(world):
    w = world
    m = w.member('UNSUB', noise=0.2)
    rel = w.release('unsub', {'unsub': (m, 1.0)})
    S.run_many(ctx_for(w), [spec('alpha', [dict(name='u', arm='unsub', oof=m['oof'])], rel)], NO_SUBS, write=True)
    rep = S.run_many(ctx_for(w), [spec('beta', [dict(name='u', arm='unsub', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    d = rep['decision']
    assert d['duplicate_of'] == [] and d['matches_unsubmitted'][0]['candidate'] == 'alpha' and rep['verdict']['verdict'] == 'SUBMIT-ELIGIBLE'
    assert any('without a recorded submission_ref' in c for c in rep['verdict']['caveats'])
    again = S.run_many(ctx_for(w), [spec('alpha', [dict(name='u', arm='unsub', oof=m['oof'])], rel)], NO_SUBS, write=False)[0]
    assert again['decision']['matches_unsubmitted'] == []  # its own earlier screens


def test_missing_or_unreadable_registry_holds_and_is_never_created_by_a_screen(world):
    w = world
    m = w.member('NOREG', noise=0.2)
    rel = w.release('noreg', {'noreg': (m, 1.0)})
    sp = spec('noreg', [dict(name='noreg', arm='noreg', oof=m['oof'])], rel)
    w.registry.unlink()
    rep = S.run_many(ctx_for(w), [sp], NO_SUBS, write=True)[0]
    assert rep['verdict']['verdict'] == 'HOLD' and rep['decision']['duplicate_coverage']['available'] is False
    assert any(h.startswith('duplicate coverage unavailable') for h in rep['verdict']['holds']) and 'UNAVAILABLE' in rep['decision']['line']
    assert not w.registry.exists() and rep['registry_record']['appended'] == 0 and 'not appended' in rep['registry_record']['note']
    md = (w.root / E / 'SUBMISSION_SCREEN/noreg/screen.md').read_text(encoding='utf-8')
    assert 'duplicate coverage: UNAVAILABLE' in md
    w.registry.write_text('{broken', encoding='utf-8')
    bad = S.run_many(ctx_for(w), [sp], NO_SUBS, write=True, overwrite=True)[0]
    assert bad['verdict']['verdict'] == 'HOLD' and 'unreadable' in bad['decision']['duplicate_coverage']['reason']
    assert w.registry.read_text(encoding='utf-8') == '{broken'  # never replaced
    wjson(w.registry, dict(schema='something-else', entries=[]))
    assert S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]['verdict']['verdict'] == 'HOLD'
    wjson(w.registry, dict(schema=S.REGISTRY_SCHEMA, entries=[dict(candidate='x')]))  # malformed entry
    assert S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]['verdict']['verdict'] == 'HOLD'


def test_seed_registry_from_existing_screen_outputs(world):
    w = world
    m = w.member('SEED', noise=0.2)
    rel = w.release('seed', {'seed': (m, 1.0)})
    ctx = S.Ctx(w.root, draws=30, out_root=f'{E}/SUBMISSION_SCREEN/OLD')
    S.run_many(ctx, [spec('old', [dict(name='seed', arm='seed', oof=m['oof'])], rel, ref='888')], NO_SUBS, write=True)
    w.registry.unlink()  # lost registry: rebuild it from the stored reports
    out = w.root / E / 'SUBMISSION_SCREEN/OLD/old/screen.json'
    assert S.main(['--root', str(w.root), '--seed-registry', str(out)]) == 0
    reg = json.loads(w.registry.read_text(encoding='utf-8'))
    assert reg['schema'] == S.REGISTRY_SCHEMA and len(reg['entries']) == 1 and reg['entries'][0]['submission_ref'] == '888'
    assert reg['entries'][0]['source'].startswith('seed from ')
    assert S.seed_registry(S.Ctx(w.root), [out])['appended'] == 0  # idempotent
    subs = _cache(w, 's5.json', [('888', '2026-10-07 01:00:00', 'COMPLETE', '0.91')])
    dup = S.run_many(ctx_for(w), [spec('new', [dict(name='seed', arm='seed', oof=m['oof'])], rel)], subs, write=False)[0]
    assert dup['decision']['duplicate_of'][0]['ref'] == '888' and dup['verdict']['verdict'] == 'BLOCK'
    w.registry.write_text('{broken', encoding='utf-8')
    with pytest.raises(ValueError):
        S.seed_registry(S.Ctx(w.root), [out])  # an unreadable registry is never replaced, not even by seeding


# ----------------------------------------------------------------------------- astra code review 2026-10-08: decision basis enum

@pytest.mark.parametrize('basis', ['OOF', 'oof+public', 'Public', 'public ', None, 1, ''])
def test_decision_basis_is_validated_at_ingestion(world, basis):
    w = world
    ck = hashlib.sha256(b'all-data').hexdigest()
    rel = w.release('bas', {'bas': (dict(ckpts=[ck]), 1.0)})
    mem = dict(name='bas', arm='bas', oof=None, oof_reason='all-data model: no held-out folds', checkpoint_sha256=ck, weight=1.0)
    sp = spec('bas', [mem], rel)
    sp['decision']['basis'] = basis
    with pytest.raises(ValueError, match='decision.basis'):
        S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)
    p = w.root / 'spec_bas.json'
    wjson(p, sp)
    with pytest.raises(ValueError, match='decision.basis'):
        S.main(['--root', str(w.root), '--spec', str(p), '--overwrite'])


def test_public_only_needs_an_explicit_public_basis(world):
    w = world
    ck = hashlib.sha256(b'all-data').hexdigest()
    rel = w.release('pob', {'pob': (dict(ckpts=[ck]), 1.0)})
    mem = dict(name='pob', arm='pob', oof=None, oof_reason='all-data model: no held-out folds', checkpoint_sha256=ck, weight=1.0)
    sp = spec('pob', [mem], rel)
    assert S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]['verdict']['verdict'] == 'PUBLIC-ONLY'
    del sp['decision']['basis']
    r = S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]
    assert r['verdict']['verdict'] == 'HOLD' and r['decision']['basis_declared'] is False
    assert any('explicit public basis' in h for h in r['verdict']['holds'])
    sp['decision']['basis'] = 'oof'
    assert S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]['verdict']['verdict'] == 'BLOCK'


# ----------------------------------------------------------------------------- astra re-review 2026-10-08 A1: coverage reconciled with the cache

def _csv_cache(w, name, rows):
    p = w.root / name
    pd.DataFrame([dict(ref=r, fileName='submission.csv', date=d, description=desc, status=f'SubmissionStatus.{st}', publicScore=pub, privateScore='')
                  for r, d, desc, st, pub in rows]).to_csv(p, index=False)
    return p


def test_readable_but_incomplete_registry_holds(world):
    w = world
    m = w.member('COV', noise=0.2)
    rel = w.release('cov', {'cov': (m, 1.0)})
    sp = spec('cov', [dict(name='cov', arm='cov', oof=m['oof'])], rel)
    subs = _cache(w, 'c1.json', [('901', '2026-09-12 12:00:00', 'COMPLETE', '0.941'), ('902', '2026-09-13 12:00:00', 'ERROR', '')])
    rep = S.run_many(ctx_for(w), [sp], subs, write=False)[0]
    cov = rep['decision']['duplicate_coverage']
    assert cov['available'] is True and cov['complete'] is False and cov['uncovered'] == ['901'] and cov['submissions_requiring_coverage'] == 1
    assert cov['cached_scored_submissions_without_registry_entry'] == 1
    assert rep['verdict']['verdict'] == 'HOLD' and any(h.startswith('duplicate coverage incomplete') and '901' in h for h in rep['verdict']['holds'])
    assert 'duplicate coverage INCOMPLETE' in rep['decision']['line'] and 'duplicate coverage: INCOMPLETE' in S.render_md(rep)
    # no cache at all: completeness cannot be shown (fail closed)
    none = S.run_many(ctx_for(w), [sp], None, write=False)[0]
    assert none['verdict']['verdict'] == 'HOLD' and 'no submissions cache given' in none['decision']['duplicate_coverage']['incomplete_reason']
    # a legacy entry (arm-named 'release' fingerprint, no served_checkpoints) does not cover its submission
    reg = json.loads(w.registry.read_text(encoding='utf-8'))
    reg['entries'].append(dict(candidate='old', submission_ref='901',
                               fingerprints=dict(members=None, release='r' * 64, release_manifest=None, public3_submission=None)))
    wjson(w.registry, reg)
    legacy = S.run_many(ctx_for(w), [sp], subs, write=False)[0]
    assert legacy['decision']['duplicate_coverage']['uncovered_legacy_only'] == ['901'] and legacy['verdict']['verdict'] == 'HOLD'
    # an acknowledged historical entry covers it; with no fingerprint at all that is a caveat, not a HOLD
    hist = w.root / 'hist.json'
    wjson(hist, dict(schema=S.HISTORICAL_SCHEMA, submissions=[dict(
        submission_ref='901', candidate='hist_901_anchor', mapping='anchor probe22 control: external notebook stack', experiments=['EXP018'],
        unavailable={k: 'external anchor notebook: no local OOF, checkpoints, release manifest or saved Public3 output' for k in S.FP_KEYS})]))
    cache = _csv_cache(w, 'c1.csv', [('901', '2026-09-12 12:00:00', 'probe', 'COMPLETE', '0.941'), ('902', '2026-09-13 12:00:00', 'x', 'ERROR', '')])
    assert S.main(['--root', str(w.root), '--record-historical', str(hist), '--submissions', str(cache)]) == 0
    entries = json.loads(w.registry.read_text(encoding='utf-8'))['entries']
    e = entries[-1]
    assert e['kind'] == 'historical' and e['acknowledged'] is True and e['submission_ref'] == '901' and set(e['unavailable']) == set(S.FP_KEYS)
    assert e['submission']['public'] == 0.941 and e['mapping'].startswith('anchor') and b'\r\n' not in w.registry.read_bytes()
    ok = S.run_many(ctx_for(w), [sp], S.load_submissions(cache), write=False)[0]
    cov = ok['decision']['duplicate_coverage']
    assert cov['complete'] is True and cov['acknowledged_historical'] == ['901'] and cov['without_fingerprints'] == ['901']
    assert ok['verdict']['verdict'] == 'SUBMIT-ELIGIBLE' and any('cannot be detected by fingerprint' in c for c in ok['verdict']['caveats'])
    assert S.main(['--root', str(w.root), '--record-historical', str(hist), '--submissions', str(cache)]) == 0  # idempotent
    assert len(json.loads(w.registry.read_text(encoding='utf-8'))['entries']) == len(entries)
    with pytest.raises(SystemExit):
        S.main(['--root', str(w.root), '--record-historical', str(hist)])  # needs the cache


def test_historical_entry_fingerprints_detect_a_repeat(world):
    w = world
    m = w.member('HISTM', noise=0.2)
    rel = w.release('histm', {'histm': (m, 1.0)}, manifest_sha='c' * 64)
    hist = w.root / 'hist2.json'
    wjson(hist, dict(schema=S.HISTORICAL_SCHEMA, submissions=[dict(
        submission_ref='903', candidate='hist_903', mapping='own model submitted before the screen existed', members=[dict(oof=m['oof'], weight=1.0)],
        public3=dict(out_dir=rel['out_dir']))]))
    cache = _csv_cache(w, 'c3.csv', [('903', '2026-10-04 12:00:00', 'own', 'COMPLETE', '0.904')])
    assert S.main(['--root', str(w.root), '--record-historical', str(hist), '--submissions', str(cache)]) == 0
    e = json.loads(w.registry.read_text(encoding='utf-8'))['entries'][-1]
    assert all(e['fingerprints'][k] for k in S.FP_KEYS) and e['unavailable'] == {} and e['fingerprints']['release_manifest'] == 'c' * 64
    renamed = w.release('histm2', {'renamed_arm': (m, 1.0)})
    rep = S.run_many(ctx_for(w), [spec('again', [dict(name='x', arm='renamed_arm', oof=m['oof'])], renamed)], S.load_submissions(cache), write=False)[0]
    d = rep['decision']['duplicate_of']
    assert d and d[0]['ref'] == '903' and {'members', 'served_checkpoints'} <= set(d[0]['matched']) and rep['verdict']['verdict'] == 'BLOCK'
    # refusals: a fingerprint neither computable nor explained, a ref outside the cache, a missing mapping, an ERROR submission
    err = _csv_cache(w, 'c3e.csv', [('903', '2026-10-04 12:00:00', 'own', 'COMPLETE', '0.904'), ('904', '2026-10-04 12:00:00', 'e', 'ERROR', '')])
    for bad, msg in [(dict(submission_ref='903', candidate='h1', mapping='m'), 'no reason is recorded'),
                     (dict(submission_ref='999', candidate='h2', mapping='m', unavailable={k: 'x' for k in S.FP_KEYS}), 'not in the submissions cache'),
                     (dict(submission_ref='903', candidate='h3', unavailable={k: 'x' for k in S.FP_KEYS}), 'mapping'),
                     (dict(submission_ref='904', candidate='h4', mapping='m', unavailable={k: 'x' for k in S.FP_KEYS}), 'ERROR'),
                     (dict(submission_ref='903', candidate='h5', mapping='m', members=[dict(oof=m['oof'])],
                           unavailable={k: 'x' for k in S.FP_KEYS}), 'also declared unavailable')]:
        wjson(hist, dict(schema=S.HISTORICAL_SCHEMA, submissions=[bad]))
        with pytest.raises(ValueError, match=msg):
            S.main(['--root', str(w.root), '--record-historical', str(hist), '--submissions', str(err)])


def test_bind_submission_after_submitting(world):
    w = world
    m = w.member('BIND', noise=0.2)
    rel = w.release('bind', {'bind': (m, 1.0)})
    S.run_many(ctx_for(w), [spec('alpha', [dict(name='b', arm='bind', oof=m['oof'])], rel)], NO_SUBS, write=True)
    pend = _csv_cache(w, 'c_bind.csv', [('905', '2026-10-08 01:00:00', 'alpha', 'PENDING', '')])
    beta = spec('beta', [dict(name='b', arm='bind', oof=m['oof'])], rel)
    before = S.run_many(ctx_for(w), [beta], S.load_submissions(pend), write=False)[0]
    assert before['decision']['duplicate_coverage']['uncovered'] == ['905'] and before['verdict']['verdict'] == 'HOLD'
    with pytest.raises(ValueError, match='not in the submissions cache'):
        S.main(['--root', str(w.root), '--bind-submission', 'alpha', '906', '--submissions', str(pend)])
    with pytest.raises(ValueError, match='no unbound registry entry'):
        S.main(['--root', str(w.root), '--bind-submission', 'nobody', '905', '--submissions', str(pend)])
    assert S.main(['--root', str(w.root), '--bind-submission', 'alpha', '905', '--submissions', str(pend)]) == 0
    entries = json.loads(w.registry.read_text(encoding='utf-8'))['entries']
    bound = entries[-1]
    assert bound['candidate'] == 'alpha' and bound['submission_ref'] == '905' and bound['bound_from']['index'] == 0 and bound['kind'] == 'screen'
    assert bound['fingerprints'] == entries[0]['fingerprints'] and entries[0]['submission_ref'] is None  # append-only
    assert S.main(['--root', str(w.root), '--bind-submission', 'alpha', '905', '--submissions', str(pend)]) == 0  # idempotent
    assert len(json.loads(w.registry.read_text(encoding='utf-8'))['entries']) == len(entries)
    with pytest.raises(ValueError, match='already bound'):
        S.main(['--root', str(w.root), '--bind-submission', 'beta', '905', '--submissions', str(pend)])
    pending = S.run_many(ctx_for(w), [beta], S.load_submissions(pend), write=False)[0]
    assert pending['decision']['duplicate_coverage']['complete'] is True and pending['decision']['duplicates_unverified'][0]['ref'] == '905'
    assert pending['verdict']['verdict'] == 'HOLD'
    done = _csv_cache(w, 'c_bind2.csv', [('905', '2026-10-08 01:00:00', 'alpha', 'COMPLETE', '0.921')])
    scored = S.run_many(ctx_for(w), [beta], S.load_submissions(done), write=False)[0]
    assert scored['decision']['duplicate_of'][0]['ref'] == '905' and scored['verdict']['verdict'] == 'BLOCK'
    # several different current entries of one candidate: the submitted one must be named
    m2 = w.member('BIND2', noise=0.25)
    S.run_many(ctx_for(w), [spec('alpha', [dict(name='b', arm='bind', oof=m2['oof'])], w.release('bind2', {'bind': (m2, 1.0)}))], NO_SUBS,
               write=True, overwrite=True)
    two = _csv_cache(w, 'c_bind3.csv', [('907', '2026-10-08 02:00:00', 'alpha', 'PENDING', '')])
    with pytest.raises(ValueError, match='--bind-index'):
        S.main(['--root', str(w.root), '--bind-submission', 'alpha', '907', '--submissions', str(two)])
    n = len(json.loads(w.registry.read_text(encoding='utf-8'))['entries'])
    assert S.main(['--root', str(w.root), '--bind-submission', 'alpha', '907', '--submissions', str(two), '--bind-index', str(n - 1)]) == 0


# ----------------------------------------------------------------------------- astra re-review 2026-10-08 A2: arm names are cosmetic

def test_renamed_release_keeps_its_checkpoint_identity(world):
    """An all-data model repackaged under another arm name (and another release manifest) is the same served model."""
    w = world
    ck = hashlib.sha256(b'all-data-renamed').hexdigest()

    def mem(arm):
        return dict(name=arm, arm=arm, oof=None, oof_reason='all-data model: no held-out folds', checkpoint_sha256=ck, weight=1.0)
    ra = w.release('alld_a', {'swa': (dict(ckpts=[ck]), 1.0)}, manifest_sha='a' * 64)
    rb = w.release('alld_b', {'swa_renamed': (dict(ckpts=[ck]), 1.0)}, manifest_sha='b' * 64)
    subs = _cache(w, 'c_ren.json', [('908', '2026-10-07 01:00:00', 'COMPLETE', '0.920')])
    reps = S.run_many(ctx_for(w), [spec('alld_a', [mem('swa')], ra, ref='908'), spec('alld_b', [mem('swa_renamed')], rb)], subs, write=False)
    fa, fb = reps[0]['decision']['fingerprints'], reps[1]['decision']['fingerprints']
    assert fa['served_checkpoints'] and fa['served_checkpoints'] == fb['served_checkpoints'] and fa['members'] is None
    assert fa['release_manifest'] != fb['release_manifest'] and 'release' not in fa
    d = reps[1]['decision']['duplicate_of']
    assert d and d[0]['ref'] == '908' and 'served_checkpoints' in d[0]['matched'] and reps[1]['verdict']['verdict'] == 'BLOCK'
    assert reps[1]['decision']['public3_hash_only_matches'] == []
    assert reps[1]['engineering']['checkpoint_fingerprint'] == reps[0]['engineering']['checkpoint_fingerprint']
    # K and blend weight stay part of the identity; arm order and weight scale do not
    assert S.checkpoint_fingerprint([({ck}, 36, 1.0)]) != S.checkpoint_fingerprint([({ck}, 24, 1.0)])
    assert S.checkpoint_fingerprint([({'a'}, 36, 0.7), ({'b'}, 36, 0.3)]) != S.checkpoint_fingerprint([({'a'}, 36, 0.5), ({'b'}, 36, 0.5)])
    assert S.checkpoint_fingerprint([({'a'}, 36, 0.7), ({'b'}, 36, 0.3)]) == S.checkpoint_fingerprint([({'b'}, 36, 3.0), ({'a'}, 36, 7.0)])
    assert S.checkpoint_fingerprint([(set(), 36, 1.0)]) is None


# ----------------------------------------------------------------------------- astra re-review 2026-10-08 A4: every fold's label file

def per_fold_student(w, fault=None):
    """An EXP116-style member: one label file per fold registered as folds[k].label_file, a receipt binding all five files (sha and
    semantic sha) and the teachers, fold provenance and the assembled result recording each fold's file."""
    d = 'storage/labels/derived_teacher_fill'
    files = {k: f'{d}/teacher_fill_v1_f{k}.csv' for k in range(5)}
    sem = {k: f'sem-fill-f{k}' for k in range(5)}
    for k, rel in files.items():
        w.write_labels(rel, np.where(w.y == 1, 0.8 - 0.02 * k, 0.1))
    shas = {k: sha(w.root / rel) for k, rel in files.items()}
    second = w.member('SECOND_TEACHER')
    outputs = {Path(rel).name: dict(sha256=shas[k], semantic_sha256=sem[k]) for k, rel in files.items()}
    if fault == 'receipt_sha':
        outputs['teacher_fill_v1_f3.csv']['sha256'] = 'f' * 64
    if fault == 'receipt_semantic':
        outputs['teacher_fill_v1_f3.csv']['semantic_sha256'] = 'sem-other'
    wjson(w.root / d / 'receipt.json', dict(outputs=outputs, teachers={n: dict(oof=t['oof'], oof_sha256=sha(w.root / t['oof']))
                                                                        for n, t in [('coatnet5', w.base), ('radr50x5', second)]}))
    man = json.loads(w.manifest.read_text(encoding='utf-8'))
    for k in range(5):
        man['folds'][str(k)]['label_file'] = dict(name='teacher_fill_v1', path=files[k], sha256=shas[k], semantic_sha256=sem[k], fold=k)
    if fault == 'registration':
        man['folds']['3']['label_file']['semantic_sha256'] = 'sem-other'
    if fault == 'unregistered':
        del man['folds']['3']['label_file']
    man['label_set'] = dict(name='teacher_fill_v1', calibration='soft', per_fold=True)
    proto = w.root / E / 'PER_FOLD_PROTOCOL/manifest.json'
    wjson(proto, man)

    def prov_for(k):
        p = dict(label_hash=sem[k], label_semantic_sha256=sem[k], label_file=files[k], label_file_sha256=shas[k])
        if k == 3 and fault == 'provenance_sha':
            p['label_file_sha256'] = 'e' * 64
        if k == 3 and fault == 'provenance_semantic':
            p.update(label_semantic_sha256='sem-other', label_hash='sem-other')
        if k == 3 and fault == 'provenance_path':
            p['label_file'] = files[2]
        return p
    m = w.member('PER_FOLD_STUDENT', label_set='teacher_fill_v1', protocol=proto, prov_for=prov_for,
                 source_extra=lambda k: dict(label_file_sha256='d' * 64 if k == 3 and fault == 'assembled' else shas[k]))
    if fault == 'file':
        with open(w.root / files[3], 'ab') as f:
            f.write(b'\n')  # a later fold's bytes change after registration
    rel = w.release('per_fold', {'student': (m, 1.0)})
    return spec('per_fold', [dict(name='student', arm='student', oof=m['oof'])], rel)


@pytest.mark.parametrize('fault', [None, 'file', 'receipt_sha', 'receipt_semantic', 'registration', 'unregistered', 'provenance_sha',
                                   'provenance_semantic', 'provenance_path', 'assembled', 'spec_override'])
def test_per_fold_label_files_are_all_verified(world, fault):
    sp = per_fold_student(world, None if fault == 'spec_override' else fault)
    if fault == 'spec_override':
        sp['members'][0]['label_file'] = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
    rep = S.run_many(ctx_for(world), [sp], NO_SUBS, write=False)[0]
    mem = rep['lineage']['members'][0]
    if fault is None:
        assert mem['status'] == 'FLAG_INDIRECT' and mem['gate_failures'] == [] and mem['teacher_binding'] == 'RECEIPT_BOUND'
        assert sorted(mem['label_files']) == ['0', '1', '2', '3', '4']
        assert all(f['sha256'] == f['registered_sha256'] and f['teacher_binding'] == 'RECEIPT_BOUND' for f in mem['label_files'].values())
        assert {t['name'] for t in mem['teachers']} == {'coatnet5', 'radr50x5'} and mem['indirect_overlap'] == 100
        assert 'per-fold label files' in mem['label_provenance'] and rep['verdict']['verdict'] == 'SUBMIT-ELIGIBLE'
        return
    assert mem['status'] == 'FAIL_LINEAGE' and rep['verdict']['verdict'] == 'BLOCK', mem['gate_failures']
    if fault == 'spec_override':
        assert any('cannot replace the per-fold label files' in g for g in mem['gate_failures'])
        return
    assert any(g.startswith('fold 3 label file') or 'fold(s) [3]' in g for g in mem['gate_failures']), mem['gate_failures']
    assert not any(g.startswith(('fold 0 label', 'fold 1 label', 'fold 2 label', 'fold 4 label')) for g in mem['gate_failures'])


# ----------------------------------------------------------------------------- astra re-review 2026-10-08 C1: the real EXP116 assembly schema

def exp116_assembled(w):
    """An EXP116 OOF assembled by the real tools/exp116_teacher_fill.assemble from marker-carrying synthetic fold runs (the production
    result.json schema), with per-fold label files, a receipt and two teachers."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools import exp116_teacher_fill as g
    from tools import exp107_kfold_radr50 as e107
    from tools import exp113_radr50_local as e113
    from tools import diagnostic_quarantine as dq
    base = json.loads(w.manifest.read_text(encoding='utf-8'))
    second = w.member('RADR50_TEACHER')
    ldir = g.label_dir_for('soft')
    outputs, folds = {}, {}
    for k in range(5):
        rel = f'{ldir}/{g.fold_name(k)}'
        w.write_labels(rel, np.where(w.y == 1, 0.8 - 0.01 * k, 0.1))
        outputs[g.fold_name(k)] = dict(sha256=sha(w.root / rel), semantic_sha256=f'sem-fill-f{k}', rows=len(w.uids))
        oi = json.loads((w.root / base['folds'][str(k)]['ordered_ids']['path']).read_text(encoding='utf-8'))
        folds[str(k)] = dict(base['folds'][str(k)], tag=g.tag_for(k), n_train=len(oi['train']), n_val=len(oi['val']), steps=20, max_fp16_skips=0,
                             pos_weight=[1.0] * 12, label_file=dict(name='teacher_fill_v1', calibration='soft', path=rel, sha256=sha(w.root / rel),
                                                                    semantic_sha256=f'sem-fill-f{k}', fold=k))
    wjson(w.root / ldir / g.RECEIPT_NAME, dict(schema=g.RECEIPT_SCHEMA, outputs=outputs,
                                               teachers={n: dict(oof=t['oof'], oof_sha256=sha(w.root / t['oof'])) for n, t in [('coatnet5', w.base), ('radr50x5', second)]}))
    proto = w.root / g.PROTOCOL / 'manifest.json'
    wjson(proto, dict(schema=g.SCHEMA, seed=42, common_args=e107.EXPECTED_RECIPE, folds=folds, initial_state_sha256='init',
                      label_set=dict(name='teacher_fill_v1', calibration='soft', per_fold=True),
                      source_bindings={g.SHARED_RUNNER: 'runner-sha', g.TOOL: 'guard-sha'}, tags=dict(prefix=g.TAG_PREFIX), paths=dict(corpus='c'),
                      evaluation=dict(decision_eligible=False)))
    df = pd.DataFrame(w.preds(0.3), columns=T)
    df.insert(0, 'StudyInstanceUID', w.uids)
    ckpts = []
    for k in range(5):
        p = json.loads(proto.read_text(encoding='utf-8'))
        view = g.protocol_view(p, k)
        tag, run = g.tag_for(k), w.root / E / g.tag_for(k)
        run.mkdir(parents=True)
        dq.write_marker(run, g.run_marker(tag, k, 'full', proto, w.root))
        oi = json.loads((w.root / folds[str(k)]['ordered_ids']['path']).read_text(encoding='utf-8'))
        fd = df[df['StudyInstanceUID'].isin(oi['val'])]
        fd.to_csv(run / 'oof_best.csv', index=False)
        fd.to_csv(run / 'oof_k36.csv', index=False)
        lf = folds[str(k)]['label_file']
        prov = dict(fold=k, seed=42, label_set='teacher_fill_v1', label_semantic_sha256=lf['semantic_sha256'], label_file_sha256=lf['sha256'],
                    label_hash=lf['semantic_sha256'], label_file=lf['path'], protocol_sha256=sha(proto), gold_access='disabled',
                    eval_ids_ordered_sha256=S.uid_hash(oi['val']), train_ids_ordered_sha256=S.uid_hash(oi['train']), training_mode='full',
                    checkpoint_selection='fixed-final', arch='resnet50', input_policy='center3', norm='radimagenet', remote_relaxed=False,
                    initial_state_sha256='init', initial_state_matches_registered=True, environment_matches_baseline=True,
                    runner_sha256='runner-sha', guard_sha256='guard-sha', init_weights=dict(weights_sha256=e113.RADIMAGENET_SHA256),
                    actual_pos_weight=[1.0] * 12)
        wjson(run / 'provenance.json', prov)
        wjson(run / 'summary.json', dict(provenance=prov, optimizer_updates=20, skipped_updates=0, selected_epoch_1based=view['common_args']['epochs'],
                                         train=len(oi['train']), val=len(oi['val']), args=dict(tag=tag), total_seconds=1))
        (run / 'history.json').write_text('[]', encoding='utf-8')
        (run / 'model_best.pt').write_bytes(f'weights-{k}'.encode())
        ckpts.append(sha(run / 'model_best.pt'))
        names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv']
        wjson(run / 'output-inventory.json', {n: dict(sha256=sha(run / n)) for n in names})
        g.annotate_run_marker(run)
    body = g.assemble(root=w.root)
    return g, body, dict(name='exp116', oof=f'{g.OOF_DIR}/oof_k36.csv', ckpts=ckpts)


def test_exp116_assembly_schema_reads_public_only(world, monkeypatch):
    """C1: the assembled K36 metadata names each fold file (oof_k36 beside oof_k36_sha256), so a correctly quarantined EXP116 member with an
    explicit public basis reads PUBLIC-ONLY (not FAIL_LINEAGE / BLOCK) and no numerical OOF evidence is produced."""
    w = world
    g, body, m = exp116_assembled(w)
    for k in range(5):
        src = body['sources'][str(k)]
        assert src['oof_k36'] == f'{E}/{g.tag_for(k)}/oof_k36.csv' and src['oof_k36_sha256'] == sha(w.root / src['oof_k36'])
        assert src['oof_k24'] == f'{E}/{g.tag_for(k)}/oof_best.csv' and src['oof_k24_sha256'] == sha(w.root / src['oof_k24'])
    assert json.loads((w.root / g.OOF_DIR / 'result.json').read_text(encoding='utf-8'))['sources'] == body['sources']
    guard_quarantined_predictions(monkeypatch, w)
    rel = w.release('exp116', {'exp116': (m, 1.0)})
    sp = spec('exp116', [dict(name='exp116', arm='exp116', oof=m['oof'])], rel)
    rep = S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]
    mem = rep['lineage']['members'][0]
    assert mem['status'] == 'FLAG_INDIRECT' and mem['gate_failures'] == [] and mem['teacher_binding'] == 'RECEIPT_BOUND', mem['gate_failures']
    assert all(r['fold_oof'] and 'decision_eligible false' in r['quarantine_reason'] for r in mem['runs'].values())
    assert sorted(mem['label_files']) == ['0', '1', '2', '3', '4'] and mem['oof_gate_failures']
    assert rep['oof']['available'] is False and rep['oof']['by_label_set'] == {} and rep['oof']['direction']['consistent'] is None
    assert rep['replication']['status'] == 'NOT_AVAILABLE' and rep['replication']['candidate_delta_dread'] is None
    assert rep['engineering']['arms']['exp116']['binding'] == 'full_reported' and rep['engineering']['engineering_pass'] is True
    assert rep['verdict']['verdict'] == 'PUBLIC-ONLY' and rep['verdict']['blockers'] == []
    sp_oof = spec('exp116_oof', [dict(name='exp116', arm='exp116', oof=m['oof'])], rel, basis='oof')
    assert S.run_many(ctx_for(w), [sp_oof], NO_SUBS, write=False)[0]['verdict']['verdict'] == 'BLOCK'
    # the pre-fix assembly schema (no oof_k36 path) could not bind its fold runs: FAIL_LINEAGE / BLOCK
    res = w.root / g.OOF_DIR / 'result.json'
    old = json.loads(res.read_text(encoding='utf-8'))
    for src in old['sources'].values():
        src.pop('oof_k36')
    res.write_text(json.dumps(old), encoding='utf-8')
    stale = S.run_many(ctx_for(w), [sp], NO_SUBS, write=False)[0]
    assert stale['lineage']['status'] == 'FAIL_LINEAGE' and stale['verdict']['verdict'] == 'BLOCK'


REAL = ROOT / E / 'EXP099_B8_DREAD_OOF/oof_k36.csv'


@pytest.mark.skipif(not (REAL.is_file() and (ROOT / E / 'EXP099_B8_DISTILL_V1_OOF/oof_k36.csv').is_file()), reason='real EXP099 artifacts absent')
def test_real_lineage_distill_is_fully_exposed_and_coat5_clean():
    ctx = S.Ctx(ROOT, draws=10)
    coat = S.resolve_member(ctx, dict(name='coat5', oof=f'{E}/EXP099_B8_DREAD_OOF/oof_k36.csv'))
    dist = S.resolve_member(ctx, dict(name='distill5', oof=f'{E}/EXP099_B8_DISTILL_V1_OOF/oof_k36.csv'))
    lc, ld = S.lineage_section(ctx, [coat]), S.lineage_section(ctx, [dist])
    assert lc['status'] == 'PASS' and lc['members'][0]['studies_validated'] == 4343
    assert ld['status'] == 'FLAG_INDIRECT' and ld['members'][0]['indirect_overlap'] == 4343
    assert lc['gold58']['n_gold'] == 58 and lc['gold58']['absent_from_training'] and ld['gold58']['absent_from_training']


@pytest.mark.skipif(not (REAL.is_file() and (ROOT / E / 'EXP107_RADR50_DREAD_OOF/oof_k36.csv').is_file()), reason='real EXP107 artifacts absent')
def test_real_bootstrap_reproduces_db030_intervals():
    """D-B-030: 300 component draws, seed 20261003 -> RadR50 - CoAt [0.000281, 0.007617], blend - CoAt [0.009608, 0.013396]."""
    ctx = S.Ctx(ROOT)
    uids = sorted(ctx.population())
    fa = np.array([ctx.population()[u] for u in uids])
    c = S.comp_index(ctx, uids)
    Y, _ = ctx.labels(ctx.path(ctx.cfg['references']['dread']['path']), uids, 'rint', False)
    coat = S.load_pred(REAL).loc[uids]
    rad = S.load_pred(ROOT / E / 'EXP107_RADR50_DREAD_OOF/oof_k36.csv').loc[uids]
    B = S.combine([coat], [1])
    blend = S.score_block(Y, S.combine([coat, rad], [1, 1]), B, fa, c, 300, 20261003)['pooled']
    assert abs(blend['delta'] - 0.01148432) < 1e-7
    assert abs(blend['ci95'][0] - 0.009608) < 1e-6 and abs(blend['ci95'][1] - 0.013396) < 1e-6
