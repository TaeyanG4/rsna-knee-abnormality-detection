"""CPU-only checks of the Raptor training helpers (no corpus, no GPU)."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip('pandas')  # the training tools run in the GPU Python, which has pandas; the dashboard venv may not

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'src'))
import raptor_common as rc  # noqa: E402
import raptor_report as rr  # noqa: E402
from label_mix import mix  # noqa: E402
from study_meta import normalize_field, normalize_laterality, normalize_manufacturer  # noqa: E402


def test_window_centers_respect_masks_and_counts():
    mask = np.zeros(64, np.uint8); mask[10:30] = 1
    centers = rc.eval_centers(mask, 64, 8)
    assert len(centers) == 8 and min(centers) >= 11 and max(centers) <= 28
    import random
    train = rc.train_centers(mask, 64, 12, random.Random(0))
    assert len(train) == 12 and all(11 <= c <= 28 for c in train)
    assert rc.eval_centers(np.zeros(64, np.uint8), 64, 3) == [1, 1, 1]  # empty study falls back to the first slices


def test_macro_auc_skips_single_class_targets():
    y = np.zeros((6, 12)); y[:3, 0] = 1; y[:, 1] = np.nan
    p = np.random.default_rng(0).random((6, 12)); p[:3, 0] = 0.9
    macro, per = rc.macro_auc(y, p)
    assert per['ACL'] == 1.0 and 'MCL' not in per and len(per) == 1 and macro == 1.0


def test_label_mix_modes():
    uids = [f'u{i}' for i in range(4)]
    teacher = pd.DataFrame({rc.UID: uids, **{t: [0.05, 0.8, 0.05, 0.95] for t in rc.LAB}})
    oof = pd.DataFrame({rc.UID: uids[:3], **{t: [0.9, 0.2, 0.5] for t in rc.LAB}})
    blend, n = mix(teacher, oof, 0.5, 'blend', [], 0.1)
    assert n == 3 and blend.set_index(rc.UID).loc['u0', 'ACL'] == pytest.approx(0.475) and blend.set_index(rc.UID).loc['u3', 'ACL'] == 0.95
    silence, _ = mix(teacher, oof, 0.5, 'silence', [], 0.1)
    s = silence.set_index(rc.UID)
    assert s.loc['u0', 'ACL'] == pytest.approx(0.475) and s.loc['u1', 'ACL'] == 0.8  # confident teacher cell untouched
    target, _ = mix(teacher, oof, 1.0, 'target', ['Synovitis'], 0.1)
    t = target.set_index(rc.UID)
    assert t.loc['u0', 'Synovitis'] == 0.9 and t.loc['u0', 'ACL'] == 0.05


def test_scanner_normalization():
    assert normalize_manufacturer('SIEMENS') == normalize_manufacturer('Siemens Healthineers') == 'Siemens'
    assert normalize_manufacturer('GE MEDICAL SYSTEMS') == 'GE' and normalize_manufacturer('') == 'unknown'
    assert normalize_field('3') == '3T' and normalize_field('1.500000') == '1.5T' and normalize_field(None) == 'unknown'
    assert normalize_laterality('RIGHT') == 'R' and normalize_laterality('') == 'unknown'


def test_bootstrap_delta_detects_a_real_improvement():
    rng = np.random.default_rng(1); n = 300
    y = rng.integers(0, 2, (n, 12)).astype(float)
    pa = y + rng.normal(0, 1.5, (n, 12)); pb = y + rng.normal(0, 0.6, (n, 12))
    d = rr.bootstrap_delta(y, pa, pb, reps=100)
    assert d['low'] > 0 and d['fraction_positive'] == 1.0 and d['reps'] == 100


def _load_tool(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_remote_slugs_match_between_runner_and_bundle():
    rb = _load_tool('remote_bundle', 'tools/remote_bundle.py')
    rr_ = _load_tool('run_remote', 'tools/colab/run_remote.py')
    for tag in ['EXP070_COLAB_F0', 'REMOTE_SMOKE_KAGGLE', 'exp.071 label-mix/a', 'X' * 80]:
        assert rb.results_slug(tag) == rr_.results_slug(tag)
        slug = rb.results_slug(tag)
        assert len(slug) <= 50 and slug == slug.lower() and not slug.endswith('-')
    assert rb.results_slug('EXP070_COLAB_F0') == 'knee-raptor-res-exp070-colab-f0'


def test_kernel_writer_forces_fp16_and_private(tmp_path):
    rb = _load_tool('remote_bundle', 'tools/remote_bundle.py')
    meta = rb.write_kernel(tmp_path / 'k', 'someone', 'EXP071_KAGGLE_F1', '--fold 1 --res 224')
    assert meta['is_private'] and meta['enable_gpu'] and meta['id'] == 'someone/knee-raptor-run-exp071-kaggle-f1'
    code = (tmp_path / 'k' / 'run.py').read_text(encoding='utf-8')
    compile(code, 'run.py', 'exec')
    assert "'--fold 1 --res 224 --amp fp16'" in code and "'EXP071_KAGGLE_F1'" in code
    with pytest.raises(SystemExit):
        rb.write_kernel(tmp_path / 'k2', 'someone', 'T', '--amp bf16')
    for args in ['--amp=bf16', '--amp=bf16 --amp fp16', '--amp fp16 --amp=bf16']:
        with pytest.raises(SystemExit):
            rb.write_kernel(tmp_path / 'bad', 'someone', 'T', args)
    rb.write_kernel(tmp_path / 'explicit', 'someone', 'T', '--amp=fp16')


def test_import_run_never_overwrites(tmp_path):
    rb = _load_tool('remote_bundle', 'tools/remote_bundle.py')
    run = tmp_path / 'remote' / 'whatever'; run.mkdir(parents=True)
    (run / 'summary.json').write_text('{"weak_oof_macro_auc": 0.9}', encoding='utf-8')
    (run / 'run_remote.json').write_text('{"tag": "EXP099_X", "platform": "colab"}', encoding='utf-8')
    (run / 'dataset-metadata.json').write_text('{}', encoding='utf-8')
    dest = tmp_path / 'experiments'
    assert rb.import_run(run, dest) == 'EXP099_X'
    assert (dest / 'EXP099_X' / 'summary.json').exists() and not (dest / 'EXP099_X' / 'dataset-metadata.json').exists()
    (run / 'summary.json').write_text('{"weak_oof_macro_auc": 0.1}', encoding='utf-8')
    assert rb.import_run(run, dest) is None
    assert '0.9' in (dest / 'EXP099_X' / 'summary.json').read_text(encoding='utf-8')


def test_hashes_ignore_line_endings_and_rehash_keeps_legacy(tmp_path, monkeypatch):
    rows = [{rc.UID: '1.2.3', **{c: 0.5 for c in rc.LAB}}, {rc.UID: '4.5.6', **{c: 1.0 for c in rc.LAB}}]
    lf, crlf = tmp_path / 'lf.csv', tmp_path / 'crlf.csv'
    pd.DataFrame(rows).to_csv(lf, index=False, lineterminator='\n')
    pd.DataFrame(rows).to_csv(crlf, index=False, lineterminator='\r\n')
    assert rc.load_label_csv(lf)[1] == rc.load_label_csv(crlf)[1]
    folds = tmp_path / 'folds.csv'
    pd.DataFrame({rc.UID: ['1.2.3', '4.5.6'], 'fold': [0, 1]}).to_csv(folds, index=False)
    rh = _load_tool('rehash_summary', 'tools/rehash_summary.py')
    exp = tmp_path / 'experiments'; (exp / 'RUN').mkdir(parents=True)
    import json
    (exp / 'RUN' / 'summary.json').write_text(json.dumps({'label_hash': 'old', 'split_hash': 'olds',
                                                          'args': {'labels': str(crlf), 'fold_file': str(folds)}}), encoding='utf-8')
    monkeypatch.setattr(rh, 'EXPERIMENTS', exp)
    assert rh.rehash('RUN')['changed'] and not rh.rehash('RUN')['changed']
    s = json.loads((exp / 'RUN' / 'summary.json').read_text(encoding='utf-8'))
    assert s['label_hash'] == rc.load_label_csv(lf)[1] and s['legacy_hashes'] == {'label_hash': 'old', 'split_hash': 'olds'}


def test_mix_labels_are_bundled_but_text_columns_are_refused(tmp_path, monkeypatch):
    rb = _load_tool('remote_bundle', 'tools/remote_bundle.py')
    mix = tmp_path / 'storage' / 'labels' / 'derived_mix'; mix.mkdir(parents=True)
    pd.DataFrame([{rb.UID: '1', **{c: 0.5 for c in rb.LAB}}]).to_csv(mix / 'mix_a50.csv', index=False)
    (mix / 'mix_a50.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(rb, 'ROOT', tmp_path)
    assert [p.name for p in rb.mix_label_files()] == ['mix_a50.csv', 'mix_a50.json']
    pd.DataFrame([{rb.UID: '1', 'Report': 'text', **{c: 0.5 for c in rb.LAB}}]).to_csv(mix / 'bad.csv', index=False)
    with pytest.raises(SystemExit):
        rb.mix_label_files()


def test_label_mix_per_finding_alpha():
    import label_mix as lm
    t = pd.DataFrame([{rc.UID: '1', **{c: 0.0 for c in rc.LAB}}])
    o = pd.DataFrame([{rc.UID: '1', **{c: 1.0 for c in rc.LAB}}])
    amap = lm.parse_alpha_map('Effusion=0.7,ACL=0.3', 0.5)
    out, n = lm.mix(t, o, amap, 'blend', None, 0.1)
    row = out.set_index(rc.UID).loc['1']
    assert n == 1 and abs(row['Effusion'] - 0.7) < 1e-9 and abs(row['ACL'] - 0.3) < 1e-9 and abs(row['MCL'] - 0.5) < 1e-9
    with pytest.raises(SystemExit):
        lm.parse_alpha_map('Nope=0.5', 0.5)
