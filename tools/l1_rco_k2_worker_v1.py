"""K2 single completed K1 deploy model, v1.0.0-rcof0.

New orchestration path: sequential frozen rendering and FP32 classifier,
workers=0. No ensemble, automatic precision, downloads or successful fallback.
This module is importable by name in Windows spawned CPU loader tests.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import socket
import sys
import threading
import time

VERSION = '1.0.0-rcof0'
SLUG = 'taeyangg4/knee-l1-rcof0-v1'
MODEL_SHA = '2c3c033e25605b5fb8bcc08e0030bed081c726e10fc56d8da62524ac08d9638c'
ANATOMY_SHA = '5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e'
UID = 'StudyInstanceUID'
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CONTRACT = dict(model_sha256=MODEL_SHA, models=1, backbone='convnext_small',
    size=384, views='v4', local_mm=100, slices=3, phys_mm=3.5, eval_stride=3,
    crop=[16, 368], chunk=32, spatial=True, extent=False, extent_meta=False,
    state_spec=None, side_model=None, precision='float32', autocast=False,
    tf32=False, workers=0, batch_studies=1, labels=LABELS, formula='sigmoid(float32(logits))',
    clean_oof=False, exposure='EXPOSED_EXPLORATORY')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)


def replace(source, destination):
    for attempt in range(40):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(.05 * (attempt + 1), 1))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(encoded(value))
    replace(tmp, path)


def heartbeat(path, **value):
    try:
        write(path, dict(value, time=time.time(), pid=os.getpid()))
    except OSError:
        pass  # A transient heartbeat sharing violation never stops inference.


class Watchdog:
    """Owned daemon, no GPU calls or subprocesses; required progress guards."""
    def __init__(self, scratch, receipt, enabled):
        self.scratch,self.receipt=Path(scratch),Path(receipt)
        self.last=time.monotonic();self.phase='setup';self.completed=0
        self.window_work=0;self.samples=[];self.healthy=None
        self.stop=threading.Event();self.thread=None
        if enabled:
            self.thread=threading.Thread(target=self.watch,name='k2-progress-watchdog',daemon=True)
            self.thread.start()

    def progress(self,phase,windows=0):
        now=time.monotonic();self.last=now;self.phase=phase;self.window_work+=windows
        self.samples.append((now,self.window_work))
        self.samples=[x for x in self.samples if now-x[0]<=1801]
        heartbeat(self.scratch/'status.json',phase=phase,completed=self.completed,windows=self.window_work)

    def watch(self):
        while not self.stop.wait(5):
            now=time.monotonic();reason=None
            if now-self.last>=900:reason='no progress for 15 minutes'
            if len(self.samples)>1:
                elapsed=self.samples[-1][0]-self.samples[0][0]
                rate=(self.samples[-1][1]-self.samples[0][1])/max(elapsed,1)
                if elapsed>=120 and rate>0:self.healthy=max(self.healthy or 0,rate)
                if elapsed>=1800 and self.healthy and rate<.25*self.healthy:
                    reason='window throughput below 25 percent for 30 minutes'
            if reason:
                write(self.receipt,dict(schema='k2-worker/v1',status='FAILED',reason=reason,
                    completed=self.completed,phase=self.phase,partial_preserved=True,qualified_for_submission=False))
                os._exit(70)

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=6)


@contextlib.contextmanager
def offline():
    """Deny outbound connections, including implicit pretrained-model fetches."""
    old_connect, old_create = socket.socket.connect, socket.create_connection
    def denied(*args, **kwargs):
        raise RuntimeError('K2 offline network denied')
    socket.socket.connect = denied
    socket.create_connection = denied
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = old_connect, old_create


def assets_check(assets, manifest_sha):
    assets = Path(assets).resolve()
    need(sha(assets / 'asset-manifest.json') == manifest_sha, 'asset manifest pin')
    m = read(assets / 'asset-manifest.json')
    need(m['contract'] == CONTRACT and m['version'] == VERSION, 'registered settings differ')
    files = {p.relative_to(assets).as_posix() for p in assets.rglob('*') if p.is_file()}
    need(files - {'asset-manifest.json', 'dataset-metadata.json'} == set(m['files']), 'asset inventory differs')
    for name, pin in m['files'].items():
        path = (assets / name).resolve()
        need(path.is_relative_to(assets) and not (assets / name).is_symlink(), 'unsafe asset')
        need(sha(path) == pin, 'asset hash mismatch: ' + name)
    need(m['files']['weights/E_prior_s0_f0.pt'] == MODEL_SHA, 'deploy identity')
    need(m['files']['weights/anatomy.pt'] == ANATOMY_SHA, 'anatomy identity')
    need({n for n in m['files'] if n.endswith(('.pt', '.safetensors'))} ==
         {'weights/E_prior_s0_f0.pt', 'weights/anatomy.pt'}, 'unexpected model')
    return m


def import_frozen(assets):
    sys.dont_write_bytecode = True
    for path in (Path(assets) / 'cloud_code', Path(assets) / 'source'):
        sys.path.insert(0, str(path.resolve()))
    import compact_train as ct
    import compact_infer as ci
    need(ct.LABELS == LABELS, 'target permutation')
    need(Path(ct.__file__).resolve() == (Path(assets) / 'source/compact_train.py').resolve(), 'trainer import collision')
    need(Path(ci.__file__).resolve() == (Path(assets) / 'source/compact_infer.py').resolve(), 'renderer import collision')
    return ct, ci


def install_label_free_source():
    """Preserve raw/series metadata bytes; never open the study label table.

    Frozen native.prepare requests split.csv only to obtain its UID roster.
    Supply that one current UID in memory; train.csv is never opened, even
    for the private N32 train-split qualification harness.
    """
    import probe_infer as pi
    original=pi.OneStudySource
    class LabelFreeSource(original):
        def read(self,item):
            name=item.filename if hasattr(item,'filename') else item
            if name==self.split+'.csv':
                need(isinstance(type(self).current,str) and type(self).current,'missing current UID')
                buf=io.StringIO();writer=csv.writer(buf);writer.writerow([UID]);writer.writerow([type(self).current])
                return buf.getvalue().encode()
            need(name==self.split+'_series.csv' or
                 (len(Path(name).parts)==4 and Path(name).parts[0]==self.split+'_series' and str(name).endswith('.dcm')),
                 'unexpected raw source read')
            return super().read(item)
    pi.OneStudySource=LabelFreeSource


def configure(device):
    if device == 'cpu':
        # Must run before importing torch or any renderer that selects a device.
        # On Windows a present-empty environment value is deleted at the OS
        # boundary; -1 also survives spawn and actually hides CUDA devices.
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    import torch
    need(device in ('cpu', 'cuda:0'), 'one logical device required')
    if device == 'cpu':
        need(not torch.cuda.is_initialized() and not torch.cuda.is_available(), 'CPU CUDA isolation failed')
        torch.set_num_threads(2)
    else:
        need(torch.cuda.is_available(), 'CUDA unavailable; no precision/device fallback')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    return torch


def load_model(ct, checkpoint, device):
    import torch
    need(sha(checkpoint) == MODEL_SHA, 'checkpoint SHA256 before deserialization')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ct.Net('convnext_small', None, crop=352, in_chans=3,
                   spatial=True, extent=False, state_spec=None)
    expected = model.state_dict()
    need(set(state) == set(expected), 'deploy state key coverage')
    need(all(state[k].shape == expected[k].shape and state[k].dtype == expected[k].dtype for k in state),
         'deploy state shape/dtype coverage')
    need(all(torch.isfinite(v).all().item() for v in state.values()), 'nonfinite checkpoint')
    model.load_state_dict(state, strict=True)
    return model.float().eval().requires_grad_(False).to(device)


def validate_item(item, uid):
    import torch
    need(item.get('uid') == uid and item.get('error') is None, 'renderer UID/error')
    x, m = item['images'], item['meta']
    need(x.dtype == torch.uint8 and x.ndim == 4 and tuple(x.shape[1:]) == (3, 384, 384) and len(x) > 0,
         'empty/wrong complete image windows')
    need(m.dtype == torch.float32 and tuple(m.shape) == (len(x), 6) and torch.isfinite(m).all().item(),
         'wrong/nonfinite metadata')
    need(isinstance(item.get('anchor'), dict) and isinstance(item['anchor'].get('method'), str), 'missing anchor evidence')
    need(item.get('side') is None, 'unexpected side model')
    return x, m


def predict(model, x, m, device):
    import torch
    mask = torch.ones((1, len(x)), dtype=torch.bool, device=device)
    with torch.inference_mode(), torch.autocast(device_type=device.split(':')[0], enabled=False):
        logits = model(x[None, ..., 16:368, 16:368].to(device), m[None].to(device), mask, chunk=32)
        need(tuple(logits.shape) == (1, 12) and logits.dtype == torch.float32 and torch.isfinite(logits).all().item(),
             'nonfinite/wrong classifier logits')
        p = logits.float().sigmoid().cpu().numpy()[0]
    need((p >= 0).all() and (p <= 1).all(), 'probabilities out of range')
    return p


def tensor_sha(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def read_template(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = csv.reader(stream)
        need(next(rows) == [UID] + LABELS, 'template target order')
        ids = [row[0] for row in rows]
    need(ids and all(ids) and len(set(ids)) == len(ids), 'missing/duplicate template UID')
    return ids


def codec_evidence(data, split, ids):
    """Check decoder ABI and actually decode one file of each used syntax."""
    import pydicom
    from pydicom.pixels import get_decoder
    used, files = {}, 0
    for uid in ids:
        paths = sorted((Path(data) / (split + '_series') / uid).glob('*/*.dcm'))
        need(paths, 'empty study series')
        for path in paths:
            header = pydicom.dcmread(path, stop_before_pixels=True)
            need(str(header.StudyInstanceUID) == uid, 'DICOM study UID mismatch')
            syntax = str(header.file_meta.TransferSyntaxUID)
            files += 1
            if syntax not in used:
                decoder = get_decoder(syntax)
                need(decoder.is_available, 'missing codec or ABI mismatch')
                pixels = pydicom.dcmread(path).pixel_array
                need(pixels.size > 0, 'decoder returned empty pixels')
                used[syntax] = dict(plugins=list(decoder.available_plugins), decoded=True,
                                   sample_sha256=sha(path), pixel_shape=list(pixels.shape))
    return dict(status='ACTIVE', python_abi=sys.implementation.cache_tag, files=files, syntaxes=used)


def environment(device):
    import torch
    versions = {}
    for name in ('torch', 'timm', 'pydicom', 'numpy', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'python-gdcm'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(versions=versions, python=sys.version, python_abi=sys.implementation.cache_tag,
                platform=platform.platform(), pid=os.getpid(), device=device,
                cuda_initialized=torch.cuda.is_initialized(), tf32=False, autocast=False,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if device.startswith('cuda') else None)


def run(assets, manifest_sha, data, template, output, scratch, device='cuda:0', split='test'):
    """No launch permission is conferred here; B owns production execution."""
    start = time.perf_counter()
    output, scratch = Path(output).resolve(), Path(scratch).resolve()
    need(not output.exists() and not scratch.exists(), 'output/scratch must be fresh')
    need(not Path(data).resolve().is_relative_to(scratch) and not Path(assets).resolve().is_relative_to(scratch), 'unsafe scratch root')
    scratch.mkdir(parents=True)
    receipts, values, methods = [], [], {}
    receipt_path = output.with_name('worker-receipt.json')
    need(not receipt_path.exists(), 'evidence directory already used')
    watchdog=Watchdog(scratch,receipt_path,device.startswith('cuda'))
    try:
        torch = configure(device)
        memory=None
        if device.startswith('cuda'):
            free,total=torch.cuda.mem_get_info(0)
            memory=dict(used_before_bytes=total-free,total_bytes=total,device=torch.cuda.get_device_name(0),
                        source='measured before inference')
        assets_check(assets, manifest_sha)
        with offline():
            ct, ci = import_frozen(assets)
            install_label_free_source()
            ids = read_template(template)
            codecs = codec_evidence(data, split, ids)
            model = load_model(ct, Path(assets) / 'weights/E_prior_s0_f0.pt', device)
            # Chunk callbacks count actual encoder progress without changing its math.
            hook=model.enc.register_forward_pre_hook(lambda mod,args: watchdog.progress('classifier-chunk',len(args[0])))
            ds = ci.CStudies(data, split, ids, scratch, Path(assets) / 'cloud_code', 384,
                eval_stride=3, views='v4', slices=3, anchor_weights=Path(assets) / 'weights/anatomy.pt',
                phys_mm=3.5, side_model=None, extent_meta=False)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ds._setup()
            observed_filtering={}
            original_prepare=ds.native.prepare
            def observed_prepare(*args,**kwargs):
                result=original_prepare(*args,**kwargs)
                for study in result['studies']:
                    observed_filtering[study[UID]]=dict(valid_series=len(study['series']),
                        study_errors=len(study['errors']),
                        rejected_slices=sum(len(s.get('errors',[])) for s in study['series']))
                return result
            ds.native.prepare=observed_prepare
            setup = time.perf_counter() - start
            for index, uid in enumerate(ids):
                t0 = time.perf_counter()
                watchdog.completed=index;watchdog.progress('render')
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    item = ds[index]
                x, m = validate_item(item, uid)
                render_s = time.perf_counter() - t0
                watchdog.progress('classifier')
                p = predict(model, x, m, device)
                if device.startswith('cuda'):
                    torch.cuda.synchronize(0)
                    peak=torch.cuda.max_memory_reserved(0)
                    need(peak<=12*2**30 and memory['used_before_bytes']+peak+2*2**30<=memory['total_bytes'],
                         'measured memory reserve/headroom exceeded')
                method = item['anchor']['method']
                methods[method] = methods.get(method, 0) + 1
                rec = dict(uid_sha256=hashlib.sha256(uid.encode()).hexdigest(), index=index, windows=len(x),
                    images_sha256=tensor_sha(x), metadata_sha256=tensor_sha(m), anchor_method=method,
                    rendering_seconds=render_s, classifier_seconds=time.perf_counter()-t0-render_s,
                    end_to_end_seconds=time.perf_counter()-t0, renderer_components=item['seconds'],
                    prediction_fallback=0, error=None)
                rec['valid_series_filtering']=observed_filtering.get(uid)
                need(rec['valid_series_filtering'] is not None,'missing native filtering evidence')
                receipts.append(rec)
                values.append(p)
                write(scratch / 'study-receipts.json', receipts)
                del item, x, m, p
            output.parent.mkdir(parents=True, exist_ok=True)
            pending = output.with_suffix('.pending.csv')
            with pending.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow([UID] + LABELS)
                for uid, row in zip(ids, values):
                    writer.writerow([uid] + [format(float(v), '.9g') for v in row])
            need(len(values) == len(ids), 'partial output')
            anatomy = next(ds.bank.model.parameters())
            rec = dict(schema='k2-worker/v1', status='COMPLETE', version=VERSION, kernel_id=SLUG,
                contract=CONTRACT, manifest_sha256=manifest_sha, template_sha256=sha(template),
                roster_sha256=digest(ids), study_count=len(ids), prediction_count=len(values),
                output_sha256=sha(pending), fast_path=False, placeholder=False,
                fallbacks=dict(prediction=0, anchor=sum(v for k, v in methods.items() if not k.startswith('skm_tea_joint'))),
                anchor_methods=methods, codecs=codecs, studies=receipts, setup_seconds=setup,
                total_seconds=time.perf_counter()-start, environment=environment(device),
                anatomy=dict(dtype=str(anatomy.dtype), device=str(anatomy.device)),
                memory=memory,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(0) if device.startswith('cuda') else None,
                qualified_for_submission=False)
            write(receipt_path, rec)
            replace(pending, output)
            hook.remove();watchdog.completed=len(ids);watchdog.progress('complete')
            return rec
    except BaseException as exc:
        write(receipt_path, dict(schema='k2-worker/v1', status='FAILED', version=VERSION,
              exception_type=type(exc).__name__, completed=len(values), partial_preserved=True,
              qualified_for_submission=False))
        raise
    finally:
        watchdog.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'manifest-sha', 'data', 'template', 'output', 'scratch'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    parser.add_argument('--split', choices=['test', 'train'], default='test')
    args = parser.parse_args(argv)
    result = run(args.assets, args.manifest_sha, args.data, args.template, args.output,
                 args.scratch, args.device, args.split)
    print(json.dumps({k: result[k] for k in ('status', 'study_count', 'total_seconds', 'qualified_for_submission')}))


if __name__ == '__main__':
    main()
