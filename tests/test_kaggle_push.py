import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import kaggle_push  # noqa: E402


def test_strip_machine_shape_keeps_backup(tmp_path):
    d = tmp_path / 'k'; d.mkdir()
    meta = {'id': 'u/k', 'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': ['u/a']}
    (d / 'kernel-metadata.json').write_text(json.dumps(meta), encoding='utf-8')
    assert kaggle_push.strip_machine_shape(d) == 'NvidiaTeslaT4'
    new = json.loads((d / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert 'machine_shape' not in new and new['enable_gpu'] and new['dataset_sources'] == ['u/a']
    assert json.loads((tmp_path / 'k.kernel-metadata.orig.json').read_text(encoding='utf-8')) == meta
    assert kaggle_push.strip_machine_shape(d) is None


def test_wait_ready_checks_every_dataset_and_reports_timeouts(monkeypatch):
    monkeypatch.setattr(kaggle_push.time, 'sleep', lambda s: None)
    answers = {'me/a': ['ready'], 'other/shared': ['processing', 'ready'], 'other/broken': ['Request failed: 403']}
    seen = []

    def status(ds):
        seen.append(ds)
        seq = answers[ds]
        return seq.pop(0) if len(seq) > 1 else seq[0]

    missing = kaggle_push.wait_ready(['me/a', 'other/shared', 'other/broken'], status=status, tries=3, pause=0)
    assert missing == ['other/broken']
    assert 'other/shared' in seen and seen.count('other/broken') == 3
