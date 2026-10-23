"""v2.1.2 local-only timing instrumentation; never packaged in a release.

Changelog 2.1.2: measured 12 GiB peak cap; desktop/headroom and live
device guard enforced by the controller. Backup: .local/backups/hb153-v212/.

Changelog 2.1.1: remove the calibration-only total-window ceiling; retain
the serving encoder chunk policy. Backup: .local/backups/hb153-v211/.

Changelog 2.1.0: runtime model/shape/flip evidence in measurement journals.
Backup: .local/backups/hb153-v210/.

Changelog: cumulative full-process CUDA peaks (including auxiliary models),
per-study commits, encoder batch cap and a fail-closed full-study window cap.
"""
from __future__ import annotations
import math
import time
from tools import hb145_output_io as io

VERSION = '2.1.2'
MAX_WINDOWS = None  # v1.1.0 processes every generated window, without truncation.
MAX_BATCH_WINDOWS = 192


def window_gate(count, chunk):
    if not count > 0 or chunk != MAX_BATCH_WINDOWS:
        # Escape the frozen worker's ordinary fallback/retry handlers.
        raise SystemExit('DO_NOT_SUBMIT: window admission failed; no truncation allowed')


def commit(torch, path, rows, uid, windows, chunk, started, decode, render, anchor, wall_started=None, runtime=None):
    window_gate(windows, chunk)
    torch.cuda.synchronize()
    reserved = torch.cuda.max_memory_reserved()
    allocated = torch.cuda.max_memory_allocated()
    if not 0 < allocated <= reserved <= 12*2**30:
        raise SystemExit('DO_NOT_SUBMIT: measured reserved VRAM exceeds 12 GiB')
    row = dict(study=uid, windows=windows, batch_windows=chunk,
               gpu_seconds=time.perf_counter()-started, decode_seconds=decode,
               render_seconds=render, anchor_seconds=anchor,
               study_seconds=time.perf_counter()-(started if wall_started is None else wall_started),
               peak_reserved_bytes=reserved, peak_allocated_bytes=allocated)
    if not all(math.isfinite(row[k]) and row[k] >= 0 for k in
               ('gpu_seconds', 'decode_seconds', 'render_seconds', 'anchor_seconds', 'study_seconds')):
        raise SystemExit('DO_NOT_SUBMIT: invalid per-study time')
    rows.append(row)
    io.journal(path, dict(identity='HB153-v2', rows=rows, done=len(rows),
                         device=torch.cuda.get_device_name(), runtime=runtime,
                         exposure='EXPOSED', clean_oof=False), path.parent)


def instrument(source):
    """Patch ONLY the local diagnostic copy, after the frozen adapter patches."""
    if 'hb153_rows' in source:
        raise ValueError('timing instrumentation anchor mismatch: already instrumented')
    def once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('timing instrumentation anchor mismatch: '+old[:70])
        source = source.replace(old, new)
    once('def main():\n', 'def main():\n'
         '    from tools import c85r7_flipall_measure as hb153\n'
         '    hb153_rows = []\n'
         '    torch.cuda.reset_peak_memory_stats()\n')
    once('    out = Path(a.output)',
         '    hb153.window_gate(1, a.chunk)\n    out = Path(a.output)')
    once('        if EAGER and not COMPILE["checked"]:',
         '        hb153.window_gate(len(item["images"]), a.chunk)\n'
         '        torch.cuda.synchronize()\n'
         '        hb153_started = time.perf_counter()\n'
         '        if EAGER and not COMPILE["checked"]:')
    once('        if a.dump_logits:\n',
         '        hb153.commit(torch, Path(os.environ["HB153_MEASURE_STATUS"]), hb153_rows,\n'
         '                     uid, len(item["images"]), a.chunk, hb153_started,\n'
         '                     item["seconds"][0], item["seconds"][1],\n'
         '                     item["seconds"][2] if len(item["seconds"]) > 2 else 0., wall_started=mark,\n'
         '                     runtime=dict(models=[p.name for p in files], models2=[p.name for p in files2],\n'
         '                                  flip_tta=FLIP is not None, workers=a.workers, size=a.size,\n'
         '                                  crop=a.crop, views=a.views, slices=a.slices))\n'
         '        if a.dump_logits:\n')
    compile(source, '<hb153-local-measurement>', 'exec')
    return source
