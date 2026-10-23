"""HB151 v3.0.1: mechanically preserve EXP143v2 science; add operational hooks.

Changelog: new file; no parent/frozen source edits. The generated diff is reviewable.
v3.0.1: report recoverable export status in completed summary.
"""
from pathlib import Path
import difflib
import hashlib

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'training/src/exp143_v2.py'
DEST = ROOT / 'training/src/hb151_v3.py'
PARENT_SHA = 'eea7f76795f7289e50982d1c69f800534b7169b32e491465bcd1a980deb32c46'


def render():
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA, 'parent drift'
    source = PARENT.read_text(encoding='utf-8').split('\ndef scratch_directory(')[0]
    edits = [
        ('from tools import exp143_v2 as guard', 'from tools import hb151_build_v3 as guard\n    guard.require_child("fit")'),
        ('    seconds = float(saved_seconds)\n    for file in Path(out).glob("stop-*.json"):\n        receipt = json.loads(file.read_text(encoding="utf-8"))\n        seconds = max(seconds, float(receipt["elapsed_seconds"]))\n    return seconds',
         '    from tools.hb151_build_v3 import charged_seconds\n    return max(float(saved_seconds), charged_seconds())'),
        ('    if args.resume:\n        saved =', '    guard.PROGRESS.phase_start("startup")\n    if args.resume:\n        saved ='),
        ('            model.train()\n            optimizer.zero_grad', '            model.train()\n            guard.PROGRESS.phase_start("training_step")\n            optimizer.zero_grad'),
        ('                    this_updates += 1\n                    check_budget()', '                    this_updates += 1\n                    guard.PROGRESS.completed("training_step", {"update": updates})\n                    check_budget()'),
        ('                    for b in bn_loader:', '                    guard.PROGRESS.phase_start("bn_replay")\n                    for bn_index, b in enumerate(bn_loader):'),
        ('                        yield unpack_batch(b)\n            bn =', '                        yield unpack_batch(b)\n                        guard.PROGRESS.completed("bn_replay", {"progress_kind": "bn_replay", "cumulative_units": bn_index + 1})\n            bn ='),
        ('            guard.save_torch_new(out / "model_best.pt", payload)', '            payload.update(exposure="EXPOSED", clean_oof=False)\n            guard.save_torch_new(out / "model_best.pt", payload)\n            export_status = guard.export_checkpoint(out / "model_best.pt")'),
        ('                seconds=wall(), bn=bn, model_sha256=', '                seconds=wall(), bn=bn, export=export_status, exposure="EXPOSED", clean_oof=False, model_sha256='),
    ]
    for old, new in edits:
        assert source.count(old) == 1, 'transformation no longer unique: ' + old[:70]
        source = source.replace(old, new)
    source = '# Generated HB151 v3.0.1; see tools/hb151_trainer_codegen_v3.py.\n' + source
    return source


def verify():
    expected = render()
    assert DEST.read_text(encoding='utf-8') == expected, 'unregistered trainer edit'
    original = PARENT.read_text(encoding='utf-8').split('\ndef scratch_directory(')[0]
    return ''.join(difflib.unified_diff(original.splitlines(True), expected.splitlines(True),
                                      fromfile='EXP143v2/train', tofile='HB151v3/train'))


if __name__ == '__main__':
    if DEST.exists():
        verify()
    else:
        DEST.write_text(render(), encoding='utf-8', newline='\n')
    print('trainer transformation verified')
