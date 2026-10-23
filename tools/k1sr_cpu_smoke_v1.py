"""v1.0.0: real-DICOM CPU integration check; no CUDA or qualification claim.

Changelog: spawned real loader, frozen CPU classifier, optional failure/guard,
and exact-parent-byte publication. Fixed lane scratch, no labels or reports.
The Owner's CUDA classifier boundary is explicitly substituted with FP32 CPU.
"""
import csv
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if os.name == 'nt':
    os.environ['S6PLUS_PIN_DLL'] = r'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
    from tools import s6plus

from tools import k1_shared_render_harness_v1 as h
from tools import k1sr_runtime_v1 as r
from tools import k1sr_n32_v1 as n32
from tools import c85r7_r6w958k1sr_v1156 as b


def main():
    os.environ['K1SR_STARTED'] = str(time.time())
    torch = r.w.configure('cpu')
    out = h.OUT / 'integration-cpu'
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / 'receipt.json'
    r.w.write(receipt, dict(status='STARTING', cuda_initialized=False))
    started = time.perf_counter()
    selected = n32.prepare(h.OUT/'raw', out/'n32-data')
    assert selected == n32.prepare(h.OUT/'raw', out/'n32-data')
    selection = r.w.read(h.OUT/'selection.private.json')
    uid = selection['ids'][0]
    assert set(selection['largest8']) <= set(r.w.read(out/'n32-data/selection.json')['largest8'])
    for name, crc, size, sha in selection['members']:
        if name.split('/')[1] == uid:
            r.w.need(r.w.sha(h.OUT/'raw'/name) == sha, 'raw source identity')
    sys.path[:0] = [str(h.runtime_source()), str(h.ASSETS/'cloud_code')]
    import compact_infer as ci
    ds = ci.CStudies(out/'n32-data', 'test', [uid], out/'render', h.ASSETS/'cloud_code', 384,
        eval_stride=1, views='v4', slices=3, anchor_weights=h.ASSETS/'weights/anatomy.pt',
        phys_mm=3.5, side_model=h.BUNDLE/'side_cls.pt', extent_meta=True, fluid=True, device='cpu')
    worker = out/'success/run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True, exist_ok=True)
    # Clear only this smoke's own journals/disable markers; no user data.
    for mode in ('success', 'error', 'guard', 'capture-error', 'loader-guard'):
        directory = out/mode
        if directory.exists():
            for relative in ('run/pr7/pr7-run/s_r7/controller/k1-disabled.json',
                             'run/pr7/pr7-run/s_r7/controller/worker0/k1-studies.jsonl',
                             'k1sr-parent.csv'):
                target = (directory/relative).resolve()
                assert target.is_relative_to(out.resolve())
                if target.is_file():
                    for attempt in range(40):
                        try:
                            target.unlink()
                            break
                        except PermissionError:
                            if attempt == 39: raise
                            time.sleep(.05)
    # Exercise the actual entrypoint import, source pin and main compilation,
    # stopping at argparse help before any classifier/device setup.
    with patch.object(sys, 'argv', ['k1sr_worker.py', '--output', str(out/'startup'), '--help']), \
            contextlib.redirect_stdout(io.StringIO()):
        try:
            r.worker_main()
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError('worker help did not exit')
    assert (out/'startup/k1sr-source.json').is_file()
    loader = torch.utils.data.DataLoader(r.LoaderDataset(ds, worker), batch_size=None,
        num_workers=1, prefetch_factor=1, multiprocessing_context='spawn')
    t = time.perf_counter()
    item = next(iter(loader))
    del loader
    render_seconds = time.perf_counter()-t
    r.w.need('_k1sr' in item and 'error' not in item['_k1sr'], 'spawned capture failed')
    x, meta = r.w.validate_item(item['_k1sr']['item'], uid)
    capture = item['_k1sr']
    fallback_paths = []
    # Both optional-loader failures still render the real parent study.
    for mode in ('capture-error', 'loader-guard'):
        loader_out = out/mode/'run/pr7/pr7-run/s_r7/controller/worker0'
        clock = str(time.time()-r.GUARD_SECONDS-1) if mode == 'loader-guard' else str(time.time())
        with patch.dict(os.environ, K1SR_STARTED=clock), patch.object(
                r.shared, 'SharedR6', side_effect=RuntimeError('CPU smoke capture failure')):
            fallback = r.LoaderDataset(ds, loader_out)[0]
        r.w.need(fallback['error'] is None, 'real parent fallback rendering')
        assert torch.equal(fallback['images'], item['images'])
        assert torch.equal(fallback['meta'], item['meta'])
        fallback_paths.append(mode)
    calls = []
    # Real frozen classifier loading, architecture/state pins, and full-window
    # FP32 forward. No pretending that this exercises CUDA autocast.
    with patch.object(r.util, 'mount', return_value=h.ASSETS):
        model = r.load_classifier('cpu')
    def cpu_predict(model, x, meta, device, fp16):
        assert device == 'cuda' and fp16 is True
        return r.w.predict(model, x, meta, 'cpu')
    with patch.object(r, 'load_classifier', return_value=model), patch.object(r.shared, 'predict', cpu_predict):
        r.Owner(worker, 'cuda').handle(item, lambda value: calls.append(value['uid']))
    row = json.loads((worker/'k1-studies.jsonl').read_text())
    r.w.need(row['ok'] is True and calls == [uid], 'CPU classifier/parent boundary')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow([r.w.UID]+r.w.LABELS)
    writer.writerow([uid]+row['prediction'])  # prediction only, never labels
    parent = stream.getvalue().encode()
    statuses = {}
    for mode in ('success', 'error', 'guard'):
        dest = out/mode
        wdir = dest/'run/pr7/pr7-run/s_r7/controller/worker0'
        wdir.mkdir(parents=True, exist_ok=True)
        (dest/'submission.csv').write_bytes(parent)
        if mode != 'success':
            clone = dict(item, _k1sr=capture)
            clock = str(time.time()-r.GUARD_SECONDS-1) if mode == 'guard' else str(time.time())
            with patch.dict(os.environ, K1SR_STARTED=clock), patch.object(
                    r, 'load_classifier', side_effect=RuntimeError('CPU smoke injected K1 failure')):
                r.Owner(wdir, 'cuda').handle(clone, lambda value: calls.append(value['uid']))
        rec = r.finish(dest, dest/'run', b.config('k1sr20'), render_seconds)
        statuses[mode] = rec['status']
        if mode != 'success':
            assert rec['status'] == 'DEGRADED_PARENT_ONLY' and not rec['k1_executed']
            assert (dest/'submission.csv').read_bytes() == parent
    assert statuses['success'] == 'COMPLETE_SHARED_K1'
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS', n32=selected, spawned_real_loader=True, studies=1, windows=len(x),
        classifier='frozen K1 full-window CPU FP32; Owner CUDA boundary substituted',
        parent='real R6 renderer; classifier callback only (not full parent inference)',
        render_with_spawn_seconds=render_seconds, classifier_seconds=row['classifier_seconds'],
        total_seconds=time.perf_counter()-started, routes=[capture['detail']['route']],
        paths=statuses, loader_fallbacks=fallback_paths, worker_startup_help=True,
        cuda_initialized=False, qualified_for_submission=False,
        sources={Path(p).name:r.w.sha(p) for p in (__file__, r.__file__, n32.__file__, b.__file__)})
    r.w.write(receipt, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
