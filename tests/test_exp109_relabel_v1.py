import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('exp109', Path(__file__).resolve().parents[1] / 'tools' / 'exp109_relabel_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def rec(s='neg', p=0.1):
    return {t: {'s': s, 'p': p} for t in m.T}


def make_root(tmp_path, batches, gold=('g1',)):
    root = tmp_path
    for d in (m.IN_DIR, m.OUT_DIR, 'docs/runbooks', 'storage/competition'):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / m.PROMPT).write_text('frozen prompt\n', encoding='utf-8')
    rows = [{m.UID: u, **{t: 1.0 for t in m.T}} for u in gold] + [{m.UID: 'x', **{t: None for t in m.T}}]
    pd.DataFrame(rows).to_csv(root / m.GOLD_SOURCE, index=False)
    for b, items in batches.items():
        (root / m.IN_DIR / f'{b}.jsonl').write_text(''.join(json.dumps({'uid': u, 'report': 'r'}) + '\n' for u, _ in items), encoding='utf-8')
        (root / m.OUT_DIR / f'{b}.jsonl').write_text(''.join(json.dumps({'uid': u, 'f': f}) + '\n' for u, f in items), encoding='utf-8')
    return root


def test_convert_frozen_rule():
    f = rec('neg', 0.0)
    f['Effusion'] = {'s': 'pos', 'p': 0.9}
    f['Synovitis'] = {'s': 'na', 'p': 0.0}
    f['ACL'] = {'s': 'na', 'p': 0.7}
    f['MCL'] = {'s': 'pos', 'p': 1.0}
    out = m.convert(f)
    assert out['ACL'] == 0.05                      # na -> 0.05 whatever p says
    assert out['Synovitis'] == pytest.approx(0.54)  # 0.6 x Effusion p
    assert out['MCL'] == 0.98 and out['Lateral OA'] == 0.02  # clipped


def test_convert_rejects_bad_records():
    f = rec()
    f['ACL'] = {'s': 'maybe', 'p': 0.5}
    with pytest.raises(ValueError):
        m.convert(f)
    f = rec()
    del f['Fracture']
    with pytest.raises(ValueError):
        m.convert(f)


def test_build_drops_gold_and_refuses_overwrite(tmp_path):
    root = make_root(tmp_path, {'b00': [('u1', rec('pos', 0.8)), ('g1', rec())], 'b01': [('u2', rec('neg', 0.1))]})
    meta = m.build(root, batches=['b00', 'b01'])
    df = pd.read_csv(root / m.DEST / 'relabel_v1.csv', dtype={m.UID: str})
    assert sorted(df[m.UID]) == ['u1', 'u2'] and meta['gold_rows_removed'] == 1 and meta['rows'] == 2
    assert df.set_index(m.UID).loc['u1', 'ACL'] == pytest.approx(0.8)
    prov = json.loads((root / m.DEST / 'provenance.json').read_text(encoding='utf-8'))
    assert prov['report_text_included'] is False and 'report' not in json.dumps(prov['batches'])
    with pytest.raises(SystemExit):
        m.build(root, batches=['b00', 'b01'])


def test_build_rejects_uid_mismatch_and_missing_batch(tmp_path):
    root = make_root(tmp_path, {'b00': [('u1', rec()), ('u2', rec())]})
    (root / m.OUT_DIR / 'b00.jsonl').write_text(json.dumps({'uid': 'u2', 'f': rec()}) + '\n' + json.dumps({'uid': 'u1', 'f': rec()}) + '\n', encoding='utf-8')
    with pytest.raises(SystemExit):
        m.build(root, batches=['b00'])
    with pytest.raises(SystemExit):
        m.build(root, batches=['b00', 'b01'])
