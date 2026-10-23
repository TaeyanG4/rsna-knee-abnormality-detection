"""v1.1.0: private fresh stages and untimed contracts; historical QA stays immutable."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools import exp184_aquila_share as exp184
from tools import exp184_cpu as cpu


ROOT = exp184.ROOT


def test_registered_labels_and_control(capsys):
    labels = exp184.build_labels(dry_run=True)
    assert {arm: labels[arm]['share'] for arm in exp184.ARMS} == {'A0': 0, 'A25': 25, 'A50': 50}
    for arm in ('A0', 'A50'):
        assert labels[arm]['csv_sha256'] == exp184.sha(exp184.ROOT / exp184.DERIVED[arm])
        assert exp184.read(exp184.ROOT / exp184.DERIVED_PROVENANCE[arm])['protocol_sha256'] == exp184.sha(exp184.ROOT / exp184.PROTOCOL)
    assert exp184.control_matches_exp170(exp184.read(exp184.ROOT / exp184.protocol_path('A25')))
    assert capsys.readouterr().out == ''


def test_stage_contract_and_tamper(tmp_path):
    root = private_stage_root(tmp_path / 'workspace', exp184)
    stage = tmp_path / 'fresh-stage'
    assert exp184.stage(stage, root)['status'] == 'STAGED_NO_PUSH'
    receipt = exp184.verify_stage(stage, root)
    assert receipt['arms'] == ['A0', 'A25', 'A50']
    assert receipt['kernel_timeout_seconds'] == 30600
    assert receipt['pushed'] is False
    for arm in exp184.ARMS:
        q = exp184.read(root / exp184.protocol_path(arm))
        assert q['label_set']['sha256'] == exp184.sha(exp184.ROOT / exp184.LABELS[arm][0])
        assert q['folds']['0']['n_train'] == 3455 and q['folds']['0']['n_val'] == 888
        assert q['folds']['0']['tag'] == exp184.tag(arm)
    for relative in ('stage-receipt.json', 'kernel/kernel-metadata.json',
                     'kernel/run.ipynb', 'code-dataset/dataset-metadata.json'):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stage / relative, target)
    for archive in (exp184.BASE_NAME, exp184.OVERLAY_NAME):
        os.link(stage / 'code-dataset' / archive, tmp_path / 'code-dataset' / archive)
    meta = json.loads((tmp_path / 'kernel/kernel-metadata.json').read_text(encoding='utf-8'))
    meta['is_private'] = False
    (tmp_path / 'kernel/kernel-metadata.json').write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError, match='unsafe kernel metadata'):
        exp184.verify_stage(tmp_path, root)


def test_bootstrap_finite_policy_and_harm_guard():
    values = np.linspace(-.01, .01, 300)
    values[:30] = np.nan
    ci, valid = cpu._interval(values)
    assert valid == 270 and ci is not None
    values[30] = np.nan
    ci, valid = cpu._interval(values)
    assert valid == 269 and ci is None
    references = {name: {'delta': 0.0, 'per_target_delta': {t: 0.0 for t in exp184.TARGETS}}
                  for name in ('jev', 'consensus', 'dread')}
    assert cpu.harm_guard(references, exp184.TARGETS)['status'] == 'PASS'
    references['jev']['per_target_delta']['MCL'] = -.021
    assert cpu.harm_guard(references, exp184.TARGETS)['status'] == 'HOLD_REVIEW'


def test_exp182_uses_registered_11_macro_and_exp184_weak_four():
    targets = exp184.TARGETS
    class Evaluator:
        @staticmethod
        def auc_targets(_y, scores, _mask, _minimum):
            return scores[0]

        @staticmethod
        def macro(values):
            return float(np.nanmean(values))

    control = np.full((2, 12), .5)
    candidate = control.copy()
    candidate[:, targets.index('Lateral Meniscus')] += .04
    candidate[:, targets.index('Synovitis')] += .08
    refs = {name: (np.zeros((2, 12)), np.ones((2, 12), bool), 1)
            for name in ('jev', 'consensus', 'dread', 'aquila')}
    result = cpu.score_pair(candidate, control, refs, [np.array([0, 1])] * 300,
                            Evaluator, targets)
    assert result['weak_four']['delta'] == pytest.approx(.03)
    assert result['non_synovitis_11']['delta'] == pytest.approx(.04 / 11)
    assert result['non_synovitis_11']['valid_draws'] == 300
    assert cpu.decide('exp182', result, targets)['registered_non_harm_pass'] is True


def test_collect_verifies_all_arms_before_copy(tmp_path, monkeypatch):
    class FakeTool:
        VERSION = '1.0.0'
        EXP = 'experiment'
        ARMS = ('CONTROL', 'FILL')
        REQUIRED_OUTPUTS = ('summary.json',)
        need = staticmethod(exp184.need)
        read = staticmethod(exp184.read)
        tag = staticmethod(lambda arm: arm)

        @staticmethod
        def verify_run(_run, arm, _root):
            if arm == 'FILL':
                raise ValueError('tampered second arm')
            return {'summary.json': 'ok'}

    monkeypatch.setattr(cpu, 'module', lambda _: FakeTool)
    source = tmp_path / 'download'
    source.mkdir()
    (source / 'exp182-status.json').write_text(json.dumps(dict(
        state='COMPLETE', version='1.0.0', arms={a: dict(state='COMPLETE', sha256={'summary.json': 'ok'})
                                                   for a in FakeTool.ARMS})), encoding='utf-8')
    with pytest.raises(ValueError, match='tampered second arm'):
        cpu.collect('exp182', source, tmp_path)
    assert not (tmp_path / 'experiment/outputs').exists()


def private_stage_root(out, module):
    """Copy explicit stage inputs; generated runner/protocol writes stay private."""
    from tools import exp166_convnext_t as base, exp170_convnext_t_l2 as e170
    from tools import exp182_synovitis_fill as e182
    names={base.MANIFEST,base.BASE,base.PROTOCOL,base.TOOL,base.WEIGHT,base.INITIAL,
           base.BASE_RUNNER,module.TOOL,module.PROTOCOL,module.BASE_ARCHIVE,
           module.EXP170_PROTOCOL,module.EXP170_MANIFEST,e170.FOLD0_ROSTER,
           module.L2,module.L2_PROVENANCE,'config/workspace.toml'}
    if module is e182:
        names.update([module.FILL,module.FILL_PROVENANCE,module.FILL_QA])
    else:
        names.update(module.DERIVED.values());names.update(module.DERIVED_PROVENANCE.values())
        names.add(e182.protocol_path('CONTROL'))
    before={name:module.sha(ROOT/name) for name in names}
    for name in names:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name in (base.WEIGHT,base.INITIAL,module.BASE_ARCHIVE):os.link(ROOT/name,target)
        else:shutil.copyfile(ROOT/name,target)
    (out/module.RUNNER).parent.mkdir(parents=True,exist_ok=True)
    assert before == {name:module.sha(out/name) for name in names}
    return out
