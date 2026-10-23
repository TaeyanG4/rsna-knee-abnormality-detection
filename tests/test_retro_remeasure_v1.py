"""CPU checks for the frozen RM-v1 registration and measurement primitives."""
import hashlib
import json

import pytest

from tools import retro_remeasure_v1 as rm


def test_registration_hash_is_fail_closed(tmp_path):
    data = tmp_path/'input.json'
    data.write_text('frozen', encoding='utf-8')
    reg = {'inputs_sha256': {'input.json': rm.sha(data),
                             'tools/evidence_ledger_rulers.py': 'old-version-binding'}}
    assert rm.verify_registration(reg, tmp_path)['checked'] == 1
    data.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='input hash mismatch'):
        rm.verify_registration(reg, tmp_path)


def test_declared_prediction_hash_and_missing_paths(tmp_path):
    file = tmp_path/'pred.csv'
    file.write_text('synthetic', encoding='utf-8')
    screen = {'input_files': {'candidate': 'pred.csv'},
              'input_sha256': {'candidate': rm.sha(file)}}
    assert rm.declared_inputs(screen, tmp_path)['status'] == 'VERIFIED'
    screen['input_sha256']['candidate'] = '0'*64
    assert rm.declared_inputs(screen, tmp_path)['status'] == 'HASH_MISMATCH'
    assert rm.declared_inputs({'oof_sha256': '0'*64}, tmp_path)['status'] == 'HASH_ONLY'


def test_saved_vector_check_is_only_arithmetic():
    vec = {target: .002 for target in rm.T}
    screen = {'metric': {'dread': {'target_deltas': vec}}}
    contrast = {'prefix': '/metric', 'old_primary_delta': .002}
    out = rm.old_delta_check(screen, contrast, ['dread'])
    assert out['old_saved_macro'] == pytest.approx(.002)
    contrast['old_primary_delta'] = .003
    with pytest.raises(ValueError, match='arithmetic mismatch'):
        rm.old_delta_check(screen, contrast, ['dread'])


def test_public_agreement_ties_and_exact_mcnemar():
    rows = [{'point':'a','public':.91,'R_map10':.1,'R_old':.2},
            {'point':'b','public':.90,'R_map10':.2,'R_old':.1},
            {'point':'c','public':.901,'R_map10':.2,'R_old':.1}]
    # a-b and a-c count; b-c is a public tie (<.002).
    assert rm.agreement(rows, 'R_map10')['total'] == 2
    assert rm.agreement(rows, 'R_map10')['agree'] == 0
    assert rm.agreement(rows, 'R_old')['agree'] == 2
    assert rm.exact_mcnemar(6, 0) == pytest.approx(.03125)
    assert rm.exact_mcnemar(4, 0) == pytest.approx(.125)


def test_expected_auc_matches_binary_pair_count_and_ties():
    import numpy as np
    assert rm.exp_auc(np.array([.1,.9,.5,.5]), np.array([0.,1.,1.,0.])) == pytest.approx(.875)


def test_exp171_pair_reproduces_old_gate_and_applies_map_bar(monkeypatch):
    import numpy as np
    import pandas as pd
    ids = [f's{i}' for i in range(100)]
    y = np.tile([0., 1.], 50)
    small = pd.DataFrame(np.tile(y[:, None], (1, 12)), index=ids, columns=rm.T)
    tiny = 1-small
    monkeypatch.setattr(rm, 'score_file', lambda path: small if path == 'small' else tiny)
    monkeypatch.setattr(rm, 'component_indices',
                        lambda names: [np.arange(i, i+10) for i in range(0, 100, 10)])
    ruler_map = {'recommendations': {t: {'validity': 'LOW' if t in ('Synovitis','Contusion')
                                               else 'HIGH', 'references': [] if t in
                                               ('Synovitis','Contusion') else ['jev_hybrid']}
                                     for t in rm.T},
                 'sources': {'jev_hybrid': {'ancestors': ['jev_hybrid'],
                                            'evidence_group': 'jev'}}}
    labels = {ref: small for ref in ('jev_hybrid', 'consensus', 'dread')}
    screen = {'input_files': {'small_flip': 'small', 'tiny_flip': 'tiny'}}
    saved = {'old_saved_per_target': {t: 1. for t in rm.T}}
    result = rm.exp171_remeasure(screen, saved, ruler_map, labels, draws=20)
    assert result['status'] == 'REOPEN_CANDIDATE'
    assert result['delta_map10'] == pytest.approx(1.)
    assert result['old_reproduction_max_abs'] == pytest.approx(0.)
    saved['old_saved_per_target']['ACL'] = .9
    with pytest.raises(ValueError, match='not reproduced'):
        rm.exp171_remeasure(screen, saved, ruler_map, labels, draws=2)


def test_multiseed_map_drops_ancestry_exposed_rulers(monkeypatch):
    import numpy as np
    import pandas as pd
    ids = [f's{i}' for i in range(100)]
    y = np.tile([0., 1.], 50)
    candidate = pd.DataFrame(np.tile(y[:, None], (1, 12)), index=ids, columns=rm.T)
    control = 1-candidate
    scores = {(role, seed): candidate if role == 'candidate' else control
              for role in ('candidate','control') for seed in (42, 43)}
    monkeypatch.setattr(rm, 'component_indices',
                        lambda names: [np.arange(i, i+10) for i in range(0, 100, 10)])
    low = ('Synovitis','Contusion')
    ruler_map = {'recommendations': {t: {'validity': 'LOW' if t in low else 'MEDIUM',
                 'references': [] if t in low else ['jev_hybrid','steven_v4']}
                 for t in rm.T},
                 'sources': {'jev_hybrid': {'ancestors':['jev_hybrid'], 'evidence_group':'jev'},
                             'steven_v4': {'ancestors':['steven_v4'], 'evidence_group':'steven'}}}
    labels = {ref: candidate for ref in ('jev_hybrid','consensus','dread','steven_v4')}
    saved = {'old_saved_per_target': {t: 1. for t in rm.T}}
    result = rm.f0_remeasure({}, {}, saved, ruler_map, labels, draws=10,
                             prepared=scores, exposed=('dread','steven_v4'))
    assert result['status'] == 'REOPEN_CANDIDATE'
    assert result['per_target']['ACL']['rulers'] == ['jev_hybrid']
    assert result['delta_map10'] == pytest.approx(1.)


def test_exp139_uses_registered_swa_and_control_paths(monkeypatch):
    seen = []
    monkeypatch.setattr(rm, 'score_file', lambda path: seen.append(path) or path)
    late = {'path': 'artifacts/experiments/EXP139_AUG1_VARIANTS/LATE/screen/result.json'}
    pos = {'path': 'artifacts/experiments/EXP139_AUG1_VARIANTS/POS/screen/result.json'}
    late_screen = {'input_bindings': {f'artifacts/experiments/EXP139_AUG1_LATE_S{s}_F0/{name}': 'x'
                                      for s in (42,43) for name in ('oof_swa_k36.csv','oof_k36.csv')}}
    pos_screen = {'input_bindings': {f'artifacts/experiments/{arm}_S{s}_F0/oof_k36.csv': 'x'
                                     for s in (42,43) for arm in ('EXP139_AUG1_POS','EXP127_AUG1')}}
    result = rm.exp139_scores(late_screen, late, 'swa_vs_final k36')
    assert result[('candidate', 42)].endswith('EXP139_AUG1_LATE_S42_F0/oof_swa_k36.csv')
    assert result[('control', 42)].endswith('EXP139_AUG1_LATE_S42_F0/oof_k36.csv')
    result = rm.exp139_scores(pos_screen, pos, 'variant_vs_control k36')
    assert result[('candidate', 43)].endswith('EXP139_AUG1_POS_S43_F0/oof_k36.csv')
    assert result[('control', 43)].endswith('EXP127_AUG1_S43_F0/oof_k36.csv')
    assert len(seen) == 8
    late_screen['input_bindings'].pop('artifacts/experiments/EXP139_AUG1_LATE_S42_F0/oof_swa_k36.csv')
    with pytest.raises(ValueError, match='registered screen bindings'):
        rm.exp139_scores(late_screen, late, 'swa_vs_final k36')


def test_exp169_pair_reproduces_old_gate_and_excludes_aquila_ancestors(monkeypatch):
    import numpy as np
    import pandas as pd
    ids = [f's{i}' for i in range(100)]
    y = np.tile([0., 1.], 50)
    candidate = pd.DataFrame(np.tile(y[:, None], (1, 12)), index=ids, columns=rm.T)
    baseline = 1-candidate
    monkeypatch.setattr(rm, 'component_indices',
                        lambda names: [np.arange(i, i+10) for i in range(0, 100, 10)])
    low = ('Synovitis', 'Contusion')
    ruler_map = {'recommendations': {t: {'validity': 'LOW' if t in low else 'MEDIUM',
                 'references': [] if t in low else ['jev_hybrid','flight_v4']}
                 for t in rm.T},
                 'sources': {'jev_hybrid': {'ancestors':['jev_hybrid'], 'evidence_group':'jev'},
                             'flight_v4': {'ancestors':['flight_v4'], 'evidence_group':'flight'},
                             'aquila_v1': {'ancestors':['aquila_v1','flight_v4'],
                                           'evidence_group':'aquila'}}}
    labels = {ref: candidate for ref in ('jev_hybrid','aquila_v1','consensus')}
    screen = {'comparisons': {'l2': {'pass': False,
                                   'references': {
                                       'dread': {'delta': 0.},
                                       'jev': {'per_target_delta': {t: 1. for t in rm.T}},
                                       'aquila': {'per_target_delta': {t: 1. for t in rm.T}}}}}}
    saved = {'old_saved_per_target': {t: 1. for t in rm.T}}
    arms = {'bundle': candidate, 'l2': baseline}
    out = rm.exp169_remeasure(screen, saved,
        'bundle minus L2 alone (headline gate, D-B-097)', ruler_map, labels,
        draws=20, prepared=arms)
    assert out['status'] == 'REOPEN_CANDIDATE'
    assert out['bootstrap'] == {'draws':20, 'seed':20261003, 'unit':'component'}
    assert out['per_target']['ACL']['rulers'] == ['jev_hybrid']
    saved['old_saved_per_target']['ACL'] = .9
    with pytest.raises(ValueError, match='not reproduced'):
        rm.exp169_remeasure(screen, saved,
            'bundle minus L2 alone (headline gate, D-B-097)', ruler_map, labels,
            draws=2, prepared=arms)
