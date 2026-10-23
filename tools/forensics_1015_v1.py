"""Failure forensics v1.1.1: CPU analysis and one B-owned GPU job (forensics + lenses).

Changelog 1.1.1: fresh v111 registration after bound HB162/lens CPU-readout edits;
route the unchanged lens module into this run; allow the ended HB162 failure in
the serial queue. Preserve v110 evidence, all numerical code and GPU safeguards.

Changelog 1.1.0 (completed 2026-10-15 by the Claude builder after the astra-b2
attempt failed on a revoked token; its partial state is backed up under
.local/backups/forensics-1015-v1.1.0-astra-partial): ingest recovered HB156 fit
logs; registered real16 fp16/bf16/fp32 precision comparison with per-module
hooks, per-stage maxima, nonfinite evidence and fail-fast serving audit; CPU
FP32 replay avoids native Windows half-convolution access violations; outputs
move to .local/scratch/forensics-1015/v110 (model/raw caches reused from
analysis/, content-verified); ONE gpu_job tag FORENSICS_LENSES_1015 also runs the
seven-lens panel phases of tools/lens_completion_1015_v1.py; independent phase
groups (precision / lenses / paired Gold58) continue after another group fails;
per-process 12 GiB CUDA cap, live VRAM guard, no-progress watchdog only.
Changelog 1.0.0: exact scored serving definitions, raw FAST decode, paired Gold58
bootstrap, HB156 training eval parity; exposed evidence, idle-only supervision.
No scored files are changed. No network, training or submission operations.
"""
from __future__ import annotations
import argparse
import ast
import re
import base64
import csv
from functools import lru_cache
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'training/src'))
from tools import hb145_output_io as io

VERSION = '1.1.1'
WORK = ROOT/'.local/scratch/forensics-1015/v111'
CACHE = ROOT/'.local/scratch/forensics-1015/analysis'  # v1.0.0 model/raw caches, re-hashed on use
TAG = 'FORENSICS_LENSES_1015'
GIB = 2**30
GROUPS = (('precision-calibration', 'precision'), ('mil-calibration', 'mil-full'),
          ('release-calibration', 'release-full'), ('calibration', 'full'))
LENS_PHASES = ('mil-calibration', 'mil-full', 'release-calibration', 'release-full')
PREDECESSOR = 'HB162_SAM_F0'
PYTHON = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
NAME = 'exp143-v2-e14-e11-e12-bn.pt'
RNAME = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
ARCHIVE = 'exp152-assets.zip.bin'
RUNTIME = 'exp152_own0949_release.py'
COAT = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
CNX = 'convnext_small.fb_in22k_ft_in1k_384'
STAGES = {'EXP143v2': ROOT/'.local/exp152-own0949-v1/stage',
          'HB151': ROOT/'artifacts/experiments/HB151_KILLTEST/v1.0.2',
          'HB156': ROOT/'artifacts/experiments/HB156_KILLTEST/v1.0.0',
          'HB157': ROOT/'.local/scratch/hb157-rkd/H-B-157/v1.0.2/killtest-v1'}
ARMS = ('EXP143v2', 'HB151', 'HB156', 'public_R', 'HB157', 'HB156_train_eval')
PRECISION_ARMS = ('HB156', 'EXP143v2')
PRECISIONS = ('fp16', 'bf16', 'fp32')
HB156_LOGS = ROOT/'.local/scratch/slot-b/forensics/hb156-remote-logs'
PAIRS = {'HB151': 'EXP143v2', 'HB156': 'EXP143v2', 'HB157': 'public_R'}
TARGETS = ('ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
           'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture')
NODES = ('build_backbone','RaptorClassifier','_eval_centers','_slot_index',
         '_mirror_tri','eval_windows','gpu_windows','infer_probs','infer_pair_probs','load_model')
POLICY = dict(bootstrap_draws=2000, bootstrap_seed=20261015, parity_max_abs=.002,
              precision='fp16', clean_oof=False, exposure='EXPOSED_DIAGNOSTIC',
              idle_seconds=900, memory_peak_gib=12, headroom_gib=2,
              raw_decode=True, training=False, submission_authorized=False,
              overflow_studies=16, overflow_max_abs=.02, overflow_fp16_limit=65504,
              overflow_rule='HB156 (fp16-fp32 max_abs > .02 OR any fp16 nonfinite) AND NOT same condition in EXP143v2',
              overflow_reference='fp32 with matmul and cudnn TF32 disabled; fp16/bf16 TF32 enabled')
RAWZIP = ROOT/'storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip'
RAWINDEX = ROOT/'.local/kneeops/archive-index/9175e113e4a00fe679c4088b.sqlite'
SERIES = ROOT/'storage/competition/train_series.csv'
CORPUS = ROOT/'storage/cache/derived/EXP143_RAPTOR96_384_C140_S02'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hb156_scored_identity(stage, checkpoint_sha):
    need(checkpoint_sha == '33d9ef8a284460f8f3d199a98edad911170cffcaccec562d68475214c1f613e1', 'scored HB156 checkpoint changed')
    for name,pin in {'release.ipynb':'e795ec936bcc847b99ae06c37a22645069dcc9be49f9d2b60e0179833eaa4081',
                     'kernel-metadata.json':'f73ddbbe333bdd823ca1884bf025585fa227be0fcb297adb82473ef5d5709238'}.items():
        need(sha(stage/'kernel'/name)==pin, 'scored HB156 kernel changed')


def jhash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def read(path):
    return io.read_json(path)


def write_text(path, body):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    io.retry(lambda: temporary.write_text(body, encoding='utf-8'))
    io.replace(temporary, path)


def unique(ids):
    need(len(ids) == len(set(ids)) and all(isinstance(u, str) and u for u in ids), 'invalid/duplicate roster')
    return ids


def exclusion(ids, train):
    unique(ids); unique(train)
    return dict(evaluated=len(ids), trained=len(train), overlap=len(set(ids) & set(train)))


def code_source(nb):
    return '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')


def definitions(source):
    tree = ast.parse(source)
    selected = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NODES]
    need(len(selected) == len(NODES) and {n.name for n in selected} == set(NODES), 'serving definition inventory')
    # Exact source, including decorators. Only enclosing FAST indentation is removed.
    snippets = {}
    for n in selected:
        start = min([n.lineno]+[x.lineno for x in n.decorator_list])
        snippets[n.name] = textwrap.dedent('\n'.join(source.splitlines()[start-1:n.end_lineno]))+'\n'
    return snippets


def constants(source):
    wanted = {'SLOTS','FASTREAD_VARIANT','FR_HDR','FR_PX','_FASTREAD_B64'}
    values = {}
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in wanted:
                    values[t.id] = ast.literal_eval(n.value)
                if isinstance(t, ast.Tuple) and all(isinstance(x, ast.Name) for x in t.elts):
                    names = [x.id for x in t.elts]
                    if set(names) <= wanted:
                        values.update(zip(names, ast.literal_eval(n.value)))
    need(set(values) == wanted, 'FAST decoder constants inventory')
    return values


@lru_cache(maxsize=None)
def payload(arm, materialize=False):
    """Authenticated staged payload, never a regenerated candidate or notebook main."""
    from tools import hb151_killtest_v102 as h151
    from tools import hb157_rkd_killtest_v1 as h157
    if arm == 'HB156_train_eval':
        arm = 'HB156'
    if arm in ('public_R','HB157'):
        if arm == 'public_R':
            path = h157.PARENT
            need(sha(path) == h157.PARENT_SHA, 'public R notebook identity')
            ck = ROOT/h157.b.PARENT
            pin = h157.b.PARENT_SHA
            nb = read(path)
        else:
            stage = STAGES[arm]; receipt = read(stage/'build-receipt.json')
            need(receipt['placeholder'] is False, 'HB157 placeholder')
            nb = read(stage/'kernel/release.ipynb')
            pin = receipt['checkpoint_sha256']; ck = stage/'dataset'/RNAME
            need(nb == h157.payload(pin), 'HB157 notebook scope changed')
            need(pin == 'cec684a0b0d56681f62172617d67208de121f5708e66a83e46b3b14f37609b25', 'scored HB157 checkpoint changed')
        need(sha(ck) == pin, 'public/HB157 checkpoint mismatch')
        return code_source(nb), None, None, ck, {'checkpoint': pin}
    if arm == 'EXP143v2':
        h151.parent()
        folder = STAGES[arm]/'assets'
        manifest = read(folder/'release-manifest.json')
        ck = folder/NAME
        need(sha(ck) == manifest['checkpoints'][NAME]['sha256'], 'parent checkpoint mismatch')
        return (folder/'public-serving.py').read_text(encoding='utf-8'), (folder/RUNTIME).read_bytes(), manifest, ck, {
            'checkpoint': sha(ck), 'manifest': sha(folder/'release-manifest.json')}
    stage = STAGES[arm]
    receipt = read(stage/'build-receipt.json')
    need(receipt['placeholder'] is False, 'placeholder candidate')
    hashes, small = h151.archive_files(stage/'dataset'/ARCHIVE)
    need(hashes == receipt['files'], 'archive/receipt mismatch')
    manifest = json.loads(small['release-manifest.json'])
    if arm == 'HB151':
        h151.scope(read(stage/'kernel/release.ipynb'), read(stage/'kernel/kernel-metadata.json'),
                   hashes, small, sha(stage/'dataset'/ARCHIVE))
        need(hashes[NAME] == '0c350f18d66ccc5630d1739fcef950a853bb90f91199918fd93f82b4648fc33b', 'scored seed42 identity')
    else:
        from tools import hb156_killtest_v1 as h156
        _, original = h156.authenticate()
        need(all(small[n] == b for n, b in h156.code_bytes(original).items()), 'HB156 serving drift')
        binding = read(stage/'export-binding.json')
        need(not binding['placeholder'] and binding['sha256'] == hashes[NAME], 'HB156 checkpoint binding')
        hb156_scored_identity(stage, hashes[NAME])
        # h156.manifest calls training_model/cpu_only and would hide CUDA inside
        # a GPU worker. Reconstruct its metadata without that process-wide effect;
        # checked_model + strict loading below verify the supplied state schema.
        expected = json.loads(original['release-manifest.json'])
        code = h156.code_bytes(original)
        expected.update(protocol_sha256=h156.PROTOCOL_SHA,
            own_state_schema=manifest['own_state_schema'], hb151_export=binding,
            asset_sha256={NAME: binding['sha256'], **{n:digest(v) for n,v in code.items()}},
            serving_sha256=digest(code['public-serving.py']), dataset=h156.DATASET,
            hb156_provenance=dict(version=h156.VERSION, training_sources=h156.SOURCE_PINS,
                inherited_fields_are_parent_ancestry=['average_receipt_sha256','source_final_sha256','source_bindings']))
        expected['checkpoints'][NAME]['sha256'] = binding['sha256']
        need(manifest == expected, 'HB156 manifest drift')
    ck = CACHE/'models'/arm/NAME
    if materialize:
        ck.parent.mkdir(parents=True, exist_ok=True)
        if not ck.exists() or sha(ck) != hashes[NAME]:
            temp = ck.with_suffix('.partial')
            with zipfile.ZipFile(stage/'dataset'/ARCHIVE) as z, z.open(NAME) as src:
                with temp.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 4<<20)
            io.replace(temp, ck)
        need(sha(ck) == hashes[NAME], 'materialized checkpoint mismatch')
    return small['public-serving.py'].decode(), small[RUNTIME], manifest, ck, hashes


def serving(arm):
    import numpy as np
    import torch
    import timm
    source, runtime, manifest, ck, pins = payload(arm, True)
    snippets = definitions(source)
    ns = dict(np=np, torch=torch, nn=torch.nn, F=torch.nn.functional, timm=timm, gc=gc,
              INPUT_RES=320, IMG=384, NORM='imagenet', SLOT_SIZES=[26,22,18,12,18],
              SAGITTAL_SLOTS=(0,1), _WLUT={},
              _MEAN=torch.tensor([.485,.456,.406]).view(3,1,1),
              _STD=torch.tensor([.229,.224,.225]).view(3,1,1))
    if runtime is not None:
        mod = types.ModuleType('_forensics_runtime_'+arm)
        mod.__file__ = str(CACHE/(arm+'-runtime.py'))
        sys.modules[mod.__name__] = mod
        exec(compile(runtime, mod.__file__, 'exec'), mod.__dict__)
        ns.update(_E152=mod, _E152_MANIFEST=manifest, _E152_FILES={NAME: ck})
    exec(compile('\n'.join(snippets.values()), '<exact-stage-serving-definitions>', 'exec'), ns)
    return ns, ck, dict(pins=pins, definitions={n:digest(v.encode()) for n,v in snippets.items()})


def make_model(arm, ns, ck, device):
    import torch
    if arm == 'HB156_train_eval':
        from hb156_model_v1 import build_model
        model = build_model(res=320, backbone=CNX)
        model.load_state_dict(ns['_E152'].safe_load(ck)['model'], strict=True)
        return model.eval().to(device)
    model, res = ns['load_model'](ck, CNX if arm == 'HB156' else COAT, 384, device)
    need(res == 320 and not model.training, 'serving resolution/eval changed')
    return model


def raw_files(uid, records):
    """Copy only this registered study's DICOMs from existing ZIP, never labels/text."""
    from raptor_dicom import IndexedZipDicomSource
    dest = CACHE/'raw'
    with IndexedZipDicomSource(RAWZIP, RAWINDEX) as source:
        count = 0
        for row in records[uid]:
            series = row['SeriesInstanceUID']
            need(all(c.isdigit() or c == '.' for c in uid+series), 'unsafe DICOM identity')
            for member in source.members('train', uid, series):
                name = Path(member.path).name
                need(name.endswith('.dcm'), 'non-DICOM archive member')
                target = dest/uid/series/name
                data = source.read(member)  # zip CRC verified; compare on every replay
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or sha(target) != digest(data):
                    tmp = target.with_suffix('.partial')
                    io.retry(lambda: tmp.write_bytes(data)); io.replace(tmp, target)
                count += 1
    need(count > 0, 'no raw DICOMs')
    return dest, count


def windows_pread_adapter(module):
    """POSIX byte-read semantics for the verbatim Linux reader on Windows."""
    if hasattr(os, 'pread'):
        return False
    lock=threading.Lock()
    def pread(fd, n, offset):
        with lock:
            position=os.lseek(fd,0,os.SEEK_CUR)
            try:
                os.lseek(fd,offset,os.SEEK_SET)
                return os.read(fd,n)
            finally:
                os.lseek(fd,position,os.SEEK_SET)
    module.os=types.SimpleNamespace(**{**vars(os),'pread':pread})
    module._O_FLAGS |= os.O_BINARY
    return True


def decode(arm, uid, records):
    import numpy as np
    from tools.hb158_scale_contrast_v1 import volume
    source = payload(arm)[0]
    c = constants(source)
    data = base64.b64decode(c['_FASTREAD_B64'], validate=True)
    module = types.ModuleType('_forensics_fastread_'+arm)
    module.__file__ = str(CACHE/(arm+'-fastread.py'))
    sys.modules[module.__name__] = module
    exec(compile(data, module.__file__, 'exec'), module.__dict__)
    adapted=windows_pread_adapter(module)
    raw, count = raw_files(uid, records)
    builder = module.make_builder(c['FASTREAD_VARIANT'],384,c['SLOTS'],hdr_threads=c['FR_HDR'],px_threads=c['FR_PX'])
    vol, mask = builder(uid, records, str(raw))
    cached, cmask, pins = volume(uid)
    need(vol.shape == (96,384,384) and vol.dtype == np.uint8 and mask.shape == (96,) and mask.any(), 'raw decode invalid')
    proof = dict(raw_files=count, decoder_sha256=digest(data), windows_pread_adapter=adapted, volume_sha256=digest(vol.tobytes()),
                 mask_sha256=digest(mask.tobytes()), cache_equal=bool(np.array_equal(vol,cached) and np.array_equal(mask,cmask)),
                 cache_max_uint8_diff=int(np.max(np.abs(vol.astype(np.int16)-cached.astype(np.int16)))), cache=pins)
    return vol, mask, proof


def valid_probs(values, n):
    import numpy as np
    a = np.asarray(values)
    need(a.shape == (n,12) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(), 'invalid probabilities')
    return a


def infer(arm, ns, model, vol, mask, device):
    import torch
    if arm == 'HB156_train_eval':
        from exp143_own0949 import infer_volume
        p = infer_volume(model, vol, mask, device=device, precision='none' if device=='cpu' else 'fp16', chunk=16)
    else:
        x = ns['gpu_windows'](vol,mask,94,320,device)
        m = ns['gpu_windows'](vol,mask,94,320,device,mirror=True)
        if device == 'cpu':
            for windows, mirror in ((x,False),(m,True)):
                need(torch.equal(windows,ns['eval_windows'](vol,mask,94,320,mirror=mirror)), 'real window parity')
            with torch.inference_mode():
                p = .5*(model(x[None]).float().sigmoid()[0].numpy()+model(m[None]).float().sigmoid()[0].numpy())
        else:
            p = ns['infer_pair_probs'](model,x,m,device)
    return valid_probs([p],1)[0]


def precision_function(ns, source, precision, device):
    """Only autocast policy and the CPU-only guard differ from scored source."""
    need(precision in PRECISIONS, 'unknown precision')
    original = definitions(source)['infer_probs']
    guard = 'assert str(device).startswith("cuda"), "Qualified inference requires CUDA/fp16"'
    context = 'with torch.autocast("cuda", dtype=torch.float16):'
    need(original.count(guard)==original.count(context)==1, 'precision adapter source changed')
    changed = original
    effective = 'fp32' if device == 'cpu' else precision
    if device == 'cpu':
        changed = changed.replace(guard, 'assert str(device) == "cpu", "CPU diagnostic only"')
    if device == 'cpu' or precision != 'fp16':
        dtype = {'fp16':'float16','bf16':'bfloat16','fp32':'float32'}[effective]
        changed = changed.replace(context, f'with torch.autocast({"cpu" if device=="cpu" else "cuda"!r}, dtype=torch.{dtype}, enabled={effective!="fp32"}):')
    compiled = dict(ns)
    exec(compile(changed, '<precision-only-serving-adapter>', 'exec'), compiled)
    return compiled['infer_probs'], dict(original_sha256=digest(original.encode()),
        executed_sha256=digest(changed.encode()),exact_scored_function=changed==original,
        requested_precision=precision,effective_precision=effective,
        changes='none' if changed==original else 'autocast precision/device; CPU guard only on CPU')


class ActivationAudit:
    """Observe module outputs without retaining activations or changing outputs.

    Counts are observed module-output elements, not unique graph elements: nested
    module outputs overlap. Root model output is reported separately as logits.
    Functional operations inside modules are not individually hooked.
    """
    def __init__(self, model):
        self.stats = {}; self.handles = []; self.first_nonfinite = None
        for name,module in model.named_modules():
            self.handles.append(module.register_forward_hook(self.hook(name or 'logits')))

    def hook(self, name):
        def observe(module, inputs, output):
            import torch
            def tensors(value):
                if torch.is_tensor(value): yield value
                elif isinstance(value, (list,tuple)):
                    for item in value: yield from tensors(item)
                elif isinstance(value,dict):
                    for item in value.values(): yield from tensors(item)
            row=self.stats.setdefault(name,dict(calls=0,elements=0,nonfinite=0,
                nan=0,inf=0,max_abs_finite=0.,at_or_above_fp16_limit=0,dtypes=[]))
            row['calls']+=1
            for tensor in tensors(output):
                if not tensor.is_floating_point() or not tensor.numel(): continue
                value=tensor.detach().float(); finite=torch.isfinite(value)
                row['elements']+=value.numel()
                bad=int((~finite).sum().item())
                row['nonfinite']+=bad
                if bad and self.first_nonfinite is None: self.first_nonfinite=name
                row['nan']+=int(torch.isnan(value).sum().item())
                row['inf']+=int(torch.isinf(value).sum().item())
                absolute=value.abs()
                row['at_or_above_fp16_limit']+=int((finite & (absolute>=65504)).sum().item())
                row['max_abs_finite']=max(row['max_abs_finite'],float(torch.where(finite,absolute,0.).max().item()))
                dtype=str(tensor.dtype)
                if dtype not in row['dtypes']: row['dtypes'].append(dtype)
        return observe

    def close(self):
        for handle in self.handles: handle.remove()


STAGE_PATTERN = re.compile(r'^(logits|norm|att|backbone|backbone\.(stem|stages\.\d+|norm_pre|norm|head))$')


def stage_view(stages):
    """Per-stage view of aggregated module maxima: stem/stages/backbone/head/logits."""
    view = {}
    for name, row in stages.items():
        if STAGE_PATTERN.match(name):
            maximum = 'Infinity' if row['inf'] else row['max_abs_finite']
            view[name] = dict(max_abs=maximum, nonfinite=row['nonfinite'],
                              at_or_above_fp16_limit=row['at_or_above_fp16_limit'],
                              fraction_of_fp16_max=None if row['inf'] else row['max_abs_finite']/65504)
    finite = {k: v['max_abs_finite'] for k, v in stages.items() if not v['inf']}
    largest = max(finite, key=finite.get) if finite else None
    return dict(stages=view, largest_module=largest, largest_module_max_abs=finite.get(largest))


def cap_memory(device):
    """Hard 12 GiB caching-allocator cap: an over-budget allocation fails instead of paging."""
    import torch
    if str(device).startswith('cuda'):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., POLICY['memory_peak_gib']*GIB/total), 0)


def precision_infer(ns, source, model, vol, mask, device, precision):
    import numpy as np
    import torch
    fn,proof=precision_function(ns,source,precision,device)
    original=ns['infer_probs']; audit=ActivationAudit(model)
    previous=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    error=None; probabilities=None
    try:
        torch.backends.cuda.matmul.allow_tf32=device!='cpu' and precision!='fp32'
        torch.backends.cudnn.allow_tf32=device!='cpu' and precision!='fp32'
        ns['infer_probs']=fn
        plain=ns['gpu_windows'](vol,mask,94,320,device)
        mirrored=ns['gpu_windows'](vol,mask,94,320,device,mirror=True)
        try:
            probabilities=ns['infer_pair_probs'](model,plain,mirrored,device)
        except AssertionError as exc:
            # Preserve the genuine serving failure; never replace bad logits.
            if str(exc)!='Nonfinite fp16 logits': raise
            error=str(exc)
        values=None if probabilities is None else np.asarray(probabilities)
        if values is not None:
            need(values.shape==(12,), 'precision output shape')
            probabilities=[float(v) if np.isfinite(v) else None for v in values]
        return dict(probabilities=probabilities,error=error,modules=audit.stats,adapter=proof,
            first_nonfinite_module=audit.first_nonfinite,
            pair_completed=error is None,nonfinite_probabilities=0 if values is None else int((~np.isfinite(values)).sum()),
            saturated_probabilities=0 if values is None else int(((values==0)|(values==1)).sum()),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
    finally:
        ns['infer_probs']=original; audit.close()
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=previous


def serving_nonfinite_audit(source):
    """Fail closed on the authenticated scored consumer/error/final-output contract."""
    lines=source.splitlines()
    markers=('assert torch.isfinite(logits).all()', 'except Exception as e:',
             '_exp152_errors.append', 'arm_probs =', 'ranks[~np.isfinite(ranks)] = 0.5',
             '_E152.need(not _exp152_errors', '_E152.output_values')
    evidence={marker:[dict(line=i+1,text=line.strip()) for i,line in enumerate(lines) if marker in line]
              for marker in markers}
    need(all(evidence[m] for m in markers),'nonfinite serving contract changed')
    return dict(logits='assert-finite before sigmoid; no silent logit replacement',
        consumer='catches exception, leaves preallocated probability row unchanged, records error',
        final_output='abort on any recorded inference error before raw/submission output',
        rank_fill='final nonfinite ranks are filled with 0.5, but only after inference-error and runtime finite checks',
        silent_successful_fallback=False,activation_only='finite logits can hide internal nonfinite/saturation; hooks observe module outputs',
        source_sha256=digest(source.encode()),evidence=evidence)


def precision_summary(rows, expected_ids, device):
    import numpy as np
    unique(expected_ids)
    need(len(rows)==len(expected_ids)*len(PRECISION_ARMS)*len(PRECISIONS),'precision row count')
    result={}
    for arm in PRECISION_ARMS:
        modes={}; arrays={}
        for mode in PRECISIONS:
            subset=[r for r in rows if r['arm']==arm and r['precision']==mode]
            need([r['uid'] for r in subset]==expected_ids,'precision paired roster/order')
            stages={}; probs=[]; failures=0; bad_probabilities=0; saturated=0
            for row in subset:
                failures+=int(row['error'] is not None); bad_probabilities+=row['nonfinite_probabilities']
                saturated+=row['saturated_probabilities']
                probs.append([np.nan]*12 if row['probabilities'] is None else row['probabilities'])
                for name,stat in row['modules'].items():
                    target=stages.setdefault(name,dict(max_abs_finite=0.,nonfinite=0,nan=0,inf=0,at_or_above_fp16_limit=0))
                    target['max_abs_finite']=max(target['max_abs_finite'],stat['max_abs_finite'])
                    for key in ('nonfinite','nan','inf','at_or_above_fp16_limit'): target[key]+=stat[key]
            arrays[mode]=np.asarray(probs,dtype=float)
            for stage in stages.values():
                stage['max_abs']='Infinity' if stage['inf'] else stage['max_abs_finite']
            modes[mode]=dict(stages=stages,stage_view=stage_view(stages),serving_assertion_failures=failures,
                nonfinite_activations=sum(v['nonfinite'] for k,v in stages.items() if k!='logits'),
                nonfinite_logits=stages.get('logits',{}).get('nonfinite',0),
                nonfinite_probabilities=bad_probabilities,saturated_probabilities=saturated)
        for uid in expected_ids:
            paired=[r['input'] for r in rows if r['arm']==arm and r['uid']==uid]
            need(len({(x['volume_sha256'],x['mask_sha256']) for x in paired})==1,'precision inputs differ')
        maximum=None
        if np.isfinite(arrays['fp16']).all() and np.isfinite(arrays['fp32']).all():
            maximum=float(np.max(abs(arrays['fp16']-arrays['fp32'])))
        nonfinite=any(modes['fp16'][k]>0 for k in ('nonfinite_activations','nonfinite_logits','nonfinite_probabilities'))
        reference_ok=bool(np.isfinite(arrays['fp32']).all()) and all(modes['fp32'][k]==0 for k in ('nonfinite_activations','nonfinite_logits','nonfinite_probabilities','serving_assertion_failures'))
        result[arm]=dict(modes=modes,fp16_vs_fp32_max_abs=maximum,fp32_valid=reference_ok,
            bf16_vs_fp32_max_abs=float(np.max(abs(arrays['bf16']-arrays['fp32']))) if np.isfinite(arrays['bf16']).all() and np.isfinite(arrays['fp32']).all() else None,
            criterion=nonfinite or (maximum is not None and maximum>POLICY['overflow_max_abs']))
    for uid in expected_ids:
        paired=[r['input'] for r in rows if r['uid']==uid]
        need(len({(x['volume_sha256'],x['mask_sha256']) for x in paired})==1,'precision models decoded different inputs')
    supported=result['HB156']['criterion'] and not result['EXP143v2']['criterion']
    conclusive=all(v['fp32_valid'] for v in result.values())
    return dict(version=VERSION,studies=len(expected_ids),device=device,models=result,
        registered_reading='CPU_DIAGNOSTIC_ONLY' if device=='cpu' else
            'CALIBRATION_ONLY' if len(expected_ids)<16 else
            'INCONCLUSIVE_INVALID_FP32' if not conclusive else
            ('FP16_SERVING_DEFECT_SUPPORTED' if supported else 'FP16_SERVING_DEFECT_NOT_SUPPORTED'),
        gpu_hypothesis_test=device!='cpu' and len(expected_ids)>=16,
        effective_precisions={mode:'fp32' if device=='cpu' else mode for mode in PRECISIONS},
        count_scope='module-output observations overlap across nested modules; functional internals not separately hooked',
        fp16_max=65504,independent_result_review_required=True)


def execute_precision(device, ids, folder):
    import torch
    from tools.exp143_own0949 import series_records
    torch.set_num_threads(8)
    if device!='cpu':
        cap_memory(device)
        need(torch.cuda.is_bf16_supported(), 'bf16 comparison requires supported B GPU')
        torch.backends.cudnn.benchmark=True
    folder.mkdir(parents=True,exist_ok=True)
    import timm
    io.receipt(folder/'runtime.json',dict(version=VERSION,python=sys.version,torch=torch.__version__,
        timm=timm.__version__,cuda=torch.version.cuda,device=device,benchmark=torch.backends.cudnn.benchmark,
        hardware=None if device=='cpu' else torch.cuda.get_device_name(),modes=list(PRECISIONS),
        effective_precisions={mode:'fp32' if device=='cpu' else mode for mode in PRECISIONS}))
    records=series_records(SERIES); rows=[]; audits={}
    for arm in PRECISION_ARMS:
        ns,ck,proof=serving(arm); source=payload(arm)[0]
        audits[arm]=serving_nonfinite_audit(source)
        for precision in PRECISIONS:
            if device!='cpu': torch.cuda.reset_peak_memory_stats()
            model=make_model(arm,ns,ck,device)
            for i,uid in enumerate(ids):
                vol,mask,inputs=decode(arm,uid,records)
                row=precision_infer(ns,source,model,vol,mask,device,precision)
                row.update(arm=arm,precision=precision,uid=uid,index=i,input=inputs,model=proof,
                    registration_sha256=sha(WORK/'registration.json'),
                    peak_reserved_bytes=0 if device=='cpu' else torch.cuda.max_memory_reserved())
                io.journal(folder/'journal'/f'{arm}-{precision}-{i:03d}.json',row,folder)
                rows.append(row)
                io.receipt(folder/'status.json',dict(phase=f'{arm}-{precision}',completed=len(rows),total=len(ids)*6))
                print(json.dumps(dict(precision=precision,arm=arm,completed=i+1,total=len(ids))),flush=True)
            del model; gc.collect()
            if device!='cpu': torch.cuda.empty_cache()
        del ns; gc.collect()
    result=precision_summary(rows,ids,device); result['serving_nonfinite_audit']=audits
    io.atomic_json(folder/'result.json',result)
    return result


def rank(x):
    from scipy.stats import rankdata
    return rankdata(x, method='average')


def corr(a,b):
    import numpy as np
    return None if np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])


def auc(y,p):
    import numpy as np
    y,p = np.asarray(y),np.asarray(p)
    need(y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all() and np.isin(y,[0,1]).all(), 'AUC inputs')
    n=int(y.sum()); m=len(y)-n
    return None if not n or not m else float((rank(p)[y==1].sum()-n*(n+1)/2)/(n*m))


def bootstrap(y, student, parent, draws=2000, seed=20261015):
    import numpy as np
    y=np.asarray(y); student=valid_probs(student,len(y)); parent=valid_probs(parent,len(y))
    need(y.shape==student.shape and np.isin(y,[0,1]).all(), 'binary complete labels required')
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(y),size=(draws,len(y)))
    result={}
    for j,t in enumerate(TARGETS):
        a,b=auc(y[:,j],student[:,j]),auc(y[:,j],parent[:,j])
        vals=[]
        for ix in indices:
            aa,bb=auc(y[ix,j],student[ix,j]),auc(y[ix,j],parent[ix,j])
            if aa is not None: vals.append(aa-bb)
        result[t]=dict(student_auc=a,parent_auc=b,delta=None if a is None else a-b,
                       ci95=None if not vals else np.quantile(vals,[.025,.975]).tolist(),valid_draws=len(vals))
    return result


def target_structure(soft,l2,r7):
    import numpy as np
    soft,l2,r7=map(np.asarray,(soft,l2,r7))
    need(soft.shape==l2.shape==r7.shape and soft.shape[1]==12, 'target alignment')
    def entropy(x):
        x=np.clip(x,1e-15,1-1e-15)
        return float(np.mean(-x*np.log2(x)-(1-x)*np.log2(1-x)))
    result={}
    for j,t in enumerate(TARGETS):
        s,l,r=soft[:,j],l2[:,j],r7[:,j]
        rs,rl,rr=rank(s),rank(l),rank(r)
        changed=discordant=tie_change=0; total=len(s)*(len(s)-1)//2
        for i in range(len(s)-1):
            a,b=np.sign(s[i+1:]-s[i]),np.sign(l[i+1:]-l[i])
            changed+=int(np.count_nonzero(a!=b))
            discordant+=int(np.count_nonzero(a*b<0))
            tie_change+=int(np.count_nonzero((a==0)!=(b==0)))
        # Exact additive covariance allocation: Var(s)=Cov(.7r,s)+Cov(s-.7r,s).
        variance=float(np.var(s))
        share=None if variance==0 else float(np.mean((.7*r-(.7*r).mean())*(s-s.mean()))/variance)
        result[t]=dict(soft_l2_spearman=corr(rs,rl),soft_r7_spearman=corr(rs,rr),
            entropy_soft_bits=entropy(s),entropy_l2_bits=entropy(l),entropy_r7_bits=entropy(r),
            rank_position_changed_fraction=float(np.mean(rs!=rl)),
            pair_order_changed_fraction=changed/total,pair_strict_reversal_fraction=discordant/total,
            pair_tie_change_fraction=tie_change/total,r7_covariance_share=share,
            r7_dominant=share is not None and share>.5,
            mean_abs_soft_minus_l2=float(np.mean(abs(s-l))),mean_abs_soft_minus_r7=float(np.mean(abs(s-r))))
    return result


def training_telemetry(folder, history):
    import numpy as np
    path=folder/'telemetry.jsonl'; child=folder/'child.log'
    result={'source_hashes': {p.name:sha(p) for p in (path,child) if p.exists()}}
    if path.exists():
        with path.open(encoding='utf-8') as stream: rows=[json.loads(line) for line in stream if line.strip()]
        result.update(samples=len(rows),elapsed_seconds=max(r['elapsed'] for r in rows),
            max_idle_seconds=max(r['idle'] for r in rows),
            phases=sorted({r['phase'] for r in rows}),
            watchdog_decisions=sorted({str(r['decision']) for r in rows if r['decision']}))
        for k in ('used_mib','util','power'):
            values=[r['telemetry'][k] for r in rows if r.get('telemetry') and isinstance(r['telemetry'].get(k),(int,float))]
            result[k]={'max':max(values),'median':float(np.median(values))} if values else None
    if child.exists():
        epochs=[]
        for line in child.read_text(encoding='utf-8').splitlines():
            try: row=json.loads(line)
            except ValueError: continue
            if isinstance(row,dict) and isinstance(row.get('loss'),(int,float)) and 'epoch' in row:
                epochs.append({k:row[k] for k in ('epoch','loss','skips','updates') if k in row})
        result['child_epoch_metrics']=epochs
        result['child_epoch_summary_agree']=None if not epochs else all(any(r['epoch']==h['epoch'] and r['loss']==h['loss'] for h in history) for r in epochs)
    return result


def source_bindings():
    paths = ['tools/forensics_1015_v1.py','tests/test_forensics_1015_v1.py',
        'tools/hb145_output_io.py','tools/hb149_monitor_v125.py','tools/c85r7_flipall_timing.py',
        'tools/hb158_scale_contrast_v1.py','tools/hb151_killtest_v102.py','tools/hb156_killtest_v1.py',
        'tools/hb157_rkd_killtest_v1.py','tools/hb157_rkd_v1.py','tools/hb151_build_v3.py',
        'tools/hb151_ancestry_v2.py','tools/exp130_gold58_compare.py','tools/exp143_own0949.py',
        'training/src/exp143_own0949.py','training/src/hb156_model_v1.py','training/src/raptor_common.py',
        'training/src/raptor_dicom.py','tools/gpu_job.py','tools/gpu_queue.py','tools/job_runtime.py',
        'tools/exp156_k_sweep.py','tools/exp115_vlm_eval.py',
        'tools/lens_completion_1015_v1.py','tests/test_lens_completion_1015_v1.py']
    paths += [str((WORK/'SPEC.md').relative_to(ROOT))]
    return {p:sha(ROOT/p) for p in paths}


def ruler_inventory(ids, train):
    from tools import exp156_k_sweep as e
    paths={'jev':[e.JEV],'aquila':[e.AQUILA],'dread':[e.DREAD],'consensus':list(e.CONSENSUS)}
    result={}
    for name,files in paths.items():
        coverage=[]; pins={}
        for p in files:
            with (ROOT/p).open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream)
                need('StudyInstanceUID' in reader.fieldnames,'ruler UID column')
                roster=[r['StudyInstanceUID'] for r in reader]
            unique(roster); coverage.append(set(roster)); pins[p]=sha(ROOT/p)
        covered=set(ids).intersection(*coverage)
        result[name]=dict(gold_covered=len(covered),source_sha256=pins,
            status='EXPOSED_GOLD_DIAGNOSTIC' if len(covered)==len(ids) else 'NO_COMPLETE_GOLD_COVERAGE',
            direct_training_overlap=len(covered & set(train)),clean_oof=False,
            note='Gold exposure and upstream public-R provenance limits apply; fold888 entirely in training')
    return result


def registration():
    payload.cache_clear()  # reauthenticate disk inputs, not a prior process-local proof
    from tools import hb151_killtest_v102 as h151
    from tools import hb156_killtest_v1 as h156
    from tools import hb157_rkd_v1 as h157
    from tools.exp130_gold58_compare import official_gold, TARGETS as gold_targets
    need(tuple(gold_targets)==TARGETS,'Gold target order')
    ids,_=official_gold(ROOT); unique(ids)
    protocols={'EXP143v2':ROOT/h157.ROSTER,'HB151':h151.protocol_identity('seed42')['protocol'],
               'HB156':h156.PROTOCOL,'HB157':h157.PROTOCOL}
    base=read(protocols['EXP143v2']); train=base['train_ids']; unique(train)
    need(len(train)==4347 and set(base['excluded_gold_ids'])==set(ids), 'parent Gold exclusion')
    rosters={}
    for arm,path in protocols.items():
        p=read(path)
        roster=train if arm=='HB157' else p['train_ids']
        rosters[arm]=exclusion(ids,roster)
        need(rosters[arm]['overlap']==0 and set(roster)==set(train), 'candidate Gold training overlap')
    hp=read(h157.PROTOCOL)
    need(hp['pins'].get(h157.ROSTER)==sha(ROOT/h157.ROSTER), 'HB157 roster pin')
    foldpath=ROOT/'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold0/ordered-ids.json'
    fold=read(foldpath)['val']; unique(fold)
    need(len(fold)==888,'four-ruler fold inventory changed')
    need(exclusion(fold,train)['overlap']==888, 'ruler eligibility requires fresh review')
    rulers=ruler_inventory(ids,train)
    need(rulers['jev']['gold_covered']==58 and all(rulers[k]['gold_covered']==0 for k in ('aquila','dread','consensus')),
         'ruler coverage changed; update registered diagnostic')
    proof={}
    for arm in ARMS[:-1]:
        source,rt,m,ck,pins=payload(arm)
        c=constants(source)
        proof[arm]=dict(pins=pins,source_sha256=digest(source.encode()),
             runtime_sha256=None if rt is None else digest(rt),
             definitions={n:digest(s.encode()) for n,s in definitions(source).items()},
             fastread_sha256=digest(base64.b64decode(c['_FASTREAD_B64'])))
    inputs=[ROOT/'storage/competition/train.csv',SERIES,RAWINDEX,CORPUS/'manifest.json',foldpath,*protocols.values(),
            HB156_LOGS/'S42/summary.json',HB156_LOGS/'fit-attempt-1/child.log',
            HB156_LOGS/'fit-attempt-1/telemetry.jsonl',
            ROOT/'.local/scratch/slot-b/forensics/hb156-parity-report.md']
    return dict(version=VERSION,policy=POLICY,ids=ids,arms=list(ARMS),pairs=PAIRS,
        sources=source_bindings(),inputs={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
        serving=proof,gold_direct_training_exclusion=rosters,four_rulers=rulers,
        public_R_upstream='historical gradient exclusion only; epoch selection exposed; upstream roster unavailable',
        gold_exposure='EXPOSED by label development, public selection and prior research; not clean validation')


def registered():
    saved=read(WORK/'registration.json')
    need(saved==registration(),'registration stale; repeat register/smoke/review')
    return saved


def cpu_analysis():
    import numpy as np
    from tools import hb151_build_v3 as soft
    from tools import hb151_ancestry_v2 as ancestry
    reg=registered()
    base,labels,provenance=soft.build_labels()
    l2,_=soft.probability_rows(ROOT/(ancestry.L2+'.csv'))
    r7,_=soft.probability_rows(ROOT/(ancestry.R7+'heldout_raw.csv'),metadata=True)
    staged=ROOT/'.local/scratch/astra-b4/H-B-151/v3.0.2/soft-labels.csv'
    need(sha(staged)=='dbd0e671778fa418f9ffa776a6bcd134e804a70280cb02e101def04b31fb72fb','soft label pin')
    actual,_=soft.probability_rows(staged)
    need(set(actual)==set(labels),'soft label roster')
    need(np.allclose([actual[u] for u in labels],[labels[u] for u in labels],rtol=0,atol=1e-9),'soft reconstruction mismatch')
    ids=[u for u in base['train_ids'] if u in r7]
    structure=target_structure([labels[u] for u in ids],[l2[u] for u in ids],[r7[u] for u in ids])
    full_ids=base['train_ids']
    full_structure=target_structure([labels[u] for u in full_ids],[l2[u] for u in full_ids],[labels[u] for u in full_ids])
    full_structure={t:{k:v for k,v in row.items() if 'r7' not in k} for t,row in full_structure.items()}
    histories={}; evidence={}
    candidates={'EXP143v2':ROOT/'artifacts/experiments/EXP143_OWN0949/S42v2/summary.json',
                'HB151':ROOT/'.local/scratch/astra-b4/H-B-151/v3.0.2/S42/summary.json',
                'HB157':ROOT/'.local/scratch/hb157-rkd/H-B-157/v1.0.2/fit/summary.json',
                'HB156':HB156_LOGS/'S42/summary.json'}
    for arm,path in candidates.items():
        if not path.exists():
            histories[arm]={'status':'MISSING_LOCAL_SUMMARY'}; continue
        s=read(path)
        rows=[]
        for item in s.get('history',[]):
            rows.append({k:v for k,v in item.items() if isinstance(v,(int,float,str,bool,type(None))) and
                         any(x in k.lower() for x in ('epoch','loss','auc','metric','skip','update','second','lr'))})
        histories[arm]=dict(status=s.get('status'),history=rows,validation='none: all-data fit',
                             summary_sha256=sha(path))
        if arm=='HB156':
            need(s.get('model_sha256')=='33d9ef8a284460f8f3d199a98edad911170cffcaccec562d68475214c1f613e1', 'HB156 recovered summary checkpoint mismatch')
        if arm in ('HB151','HB157','HB156'):
            histories[arm]['telemetry']=training_telemetry(path.parent.parent/'fit-attempt-1',rows)
        evidence[str(path.relative_to(ROOT))]=sha(path)
    histories['HB151_seed43']=dict(status='EXPORT_AND_PROTOCOL_ONLY', history=[],secondary=True)
    hb158=read(ROOT/'.local/scratch/hb158/H-B-158/result.json')
    result=dict(version=VERSION,registration_sha256=sha(WORK/'registration.json'),
        target_structure=structure,target_structure_n=len(ids),excluded_without_r7=len(labels)-len(ids),
        full_cohort_soft_vs_l2=full_structure,full_cohort_n=len(full_ids),
        soft_label_provenance=provenance,curves=histories,evidence=evidence,
        target_movement_is_not_student_prediction_movement=True,
        four_rulers=reg['four_rulers'],HB158_consistency=dict(gate=hb158['gate'],
            comparisons={k:{n:hb158[k][n] for n in ('primary_mean_delta','primary_mean_ci95')} for k in ('full384_minus_crop320','mean_minus_crop320')},
            result_sha256=sha(ROOT/'.local/scratch/hb158/H-B-158/result.json')),
        public_readings={'HB151':dict(student=.944,parent=.946,delta=-.002),
                         'HB156':dict(student=.926,parent=.946,delta=-.020),
                         'HB157':dict(student=.948,parent=.949,delta=-.001)},noise=.004,
        independent_result_review_required=True)
    io.atomic_json(WORK/'cpu-analysis.json',result)
    return dict(status='CPU_ANALYSIS_COMPLETE',target_count=12,matched_r7_studies=len(ids))


def execute(device,ids,folder):
    import numpy as np
    import torch
    import timm
    from tools.exp143_own0949 import series_records
    torch.set_num_threads(8)
    if device!='cpu':
        cap_memory(device)
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
    folder.mkdir(parents=True,exist_ok=True)
    io.receipt(folder/'runtime.json',dict(python=sys.version,torch=torch.__version__,timm=timm.__version__,
        numpy=np.__version__,cuda=torch.version.cuda,device=device,precision='fp32' if device=='cpu' else 'fp16',
        benchmark=torch.backends.cudnn.benchmark,tf32=torch.backends.cuda.matmul.allow_tf32))
    records=series_records(SERIES)
    result={}; completed=0
    for arm in ARMS:
        ns,ck,proof=serving(arm)
        if device!='cpu': torch.cuda.reset_peak_memory_stats()
        model=make_model(arm,ns,ck,device)
        values=[]
        for i,uid in enumerate(ids):
            vol,mask,inputs=decode(arm,uid,records)
            start=time.perf_counter()
            p=infer(arm,ns,model,vol,mask,device)
            if device!='cpu': torch.cuda.synchronize()
            row=dict(arm=arm,index=i,uid=uid,probabilities=p.tolist(),input=inputs,
                registration_sha256=sha(WORK/'registration.json'),model=proof,
                seconds=time.perf_counter()-start,
                peak_reserved_bytes=0 if device=='cpu' else torch.cuda.max_memory_reserved())
            io.journal(folder/'journal'/f'{arm}-{i:03d}.json',row,folder)
            completed+=1
            io.receipt(folder/'status.json',dict(phase=arm,completed=completed,total=len(ids)*len(ARMS)))
            print(json.dumps(dict(arm=arm,completed=i+1,total=len(ids))),flush=True)
            values.append(p)
        result[arm]=np.asarray(values)
        del model,ns; gc.collect()
        if device!='cpu': torch.cuda.empty_cache()
    return result


def cpu_smoke():
    import numpy as np
    import torch
    from tools import exp156_k_sweep as e
    from tools.exp130_gold58_compare import official_gold
    reg=registered()
    arms=execute('cpu',reg['ids'][:2],WORK/'cpu-smoke')
    ids,y=official_gold(ROOT)
    need(ids==reg['ids'],'Gold smoke order changed')
    jev=e._reference(ROOT/e.JEV,ids[:2],'binary')
    for labels in (y[:2],jev):
        for a,p in PAIRS.items():
            bootstrap(labels,arms[a],arms[p],draws=20,seed=POLICY['bootstrap_seed'])
    maximum=float(np.max(abs(arms['HB156']-arms['HB156_train_eval'])))
    precision=execute_precision('cpu',reg['ids'][:2],WORK/'cpu-precision')
    raw_identity_check(WORK/'cpu-smoke', 2)
    need(maximum<=POLICY['parity_max_abs'],'HB156 CPU training/serving parity failed')
    need(not torch.cuda.is_initialized(),'CPU smoke initialized CUDA')
    result=dict(status='PASS',studies=2,arms=list(ARMS),registration_sha256=sha(WORK/'registration.json'),
        hb156_max_abs_probability_difference=maximum,cuda_validated=False,
        precision_diagnostic=dict(studies=2,modes=list(PRECISIONS),arms=list(PRECISION_ARMS),
            result_sha256=sha(WORK/'cpu-precision/result.json'),gpu_hypothesis_test=False),
        metric_replay='official Gold and JEV, all three pairs, 20 paired draws on real2',
        scope='real raw DICOM -> exact FAST -> K94/mirror; original six arms and precision request paths CPU FP32; native half precision requires CUDA',
        journals={p.name:sha(p) for p in sorted((WORK/'cpu-smoke/journal').glob('*.json'))},
        precision_journals={p.name:sha(p) for p in sorted((WORK/'cpu-precision/journal').glob('*.json'))})
    io.atomic_json(WORK/'cpu-smoke.json',result)
    return result


def raw_identity_check(folder, count):
    for i in range(count):
        inputs=[read(folder/'journal'/f'{arm}-{i:03d}.json')['input'] for arm in ARMS]
        need(len({(x['volume_sha256'],x['mask_sha256']) for x in inputs})==1,
             'paired arms decoded different inputs; inspect retained journals')


def lens():
    from tools import lens_completion_1015_v1 as module
    module.OUT = WORK
    module.LWORK = WORK/'lenses'
    return module


def check_review(review):
    need(review.get('verdict')=='PASS' and review.get('independent') is True and
         review.get('reviewer') not in (None,'','builder','root','claude-forensics-lens-builder') and
         review.get('scope')==TAG and
         review.get('registration_sha256')==sha(WORK/'registration.json') and
         review.get('cpu_smoke_sha256')==sha(WORK/'cpu-smoke.json') and
         review.get('lens_registration_sha256')==sha(WORK/'lenses/registration.json') and
         review.get('lens_cpu_smoke_sha256')==sha(WORK/'lenses/cpu-smoke.json'),'independent bound review required')
    path=Path(review.get('evidence_path','')).resolve()
    need(ROOT in path.parents and path.is_file() and sha(path)==review.get('evidence_sha256'),'review evidence stale/missing')


def admit(dry=False):
    registered()
    s=read(WORK/'cpu-smoke.json')
    need(s.get('status')=='PASS' and s.get('studies')==2 and s.get('arms')==list(ARMS) and
         s.get('registration_sha256')==sha(WORK/'registration.json'),'CPU smoke missing/stale')
    for name,pin in s['journals'].items():
        need(Path(name).name==name and sha(WORK/'cpu-smoke/journal'/name)==pin,'smoke journal changed')
    need(s['precision_diagnostic']['result_sha256']==sha(WORK/'cpu-precision/result.json'), 'precision smoke result changed')
    need(len(s['precision_journals'])==12, 'precision smoke incomplete')
    for name,pin in s['precision_journals'].items():
        need(Path(name).name==name and sha(WORK/'cpu-precision/journal'/name)==pin,'precision smoke journal changed')
    L=lens(); L.registered(); L.smoke_binding()
    if dry and not (WORK/'review.json').exists():
        return dict(status='HOLD_REVIEW_REQUIRED',cuda_started=False,lens_smoke='PASS')
    check_review(read(WORK/'review.json'))
    return dict(status='READY_FOR_B_CALIBRATION',memory_admission='PENDING_MEASUREMENT',cuda_started=False)


def binding():
    return {n:sha(WORK/n) for n in ('registration.json','cpu-smoke.json','review.json',
                                    'lenses/registration.json','lenses/cpu-smoke.json')}


def require_gpu_job():
    import psutil
    from tools.hb158_scale_contrast_v1 import no_other_ml
    from tools.job_runtime import job_status
    ancestors={p.pid:p for p in psutil.Process().parents()}
    job=read(ROOT/'artifacts/logs'/TAG/'job.json')
    need(job.get('tag')==TAG and job.get('pid') in ancestors and
         ancestors[job['pid']].create_time()==job.get('create_time') and
         job.get('phase') in ('spawning','running'),'live FORENSICS_LENSES_1015 gpu_job required')
    # GPU exclusivity only: HB162 must have ENDED; its success is required only by the HB162 lens tasks.
    need(job_status(ROOT/'artifacts/logs',PREDECESSOR)['state'] in ('SUCCESS','FAILED'),
         'HB162_SAM_F0 must have ended before this job uses the GPU')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'0'),'physical GPU0 required')
    no_other_ml(set(ancestors)|{os.getpid()})


PREREQUISITE={'precision':'precision-calibration','full':'calibration',
              'mil-full':'mil-calibration','release-full':'release-calibration'}


def require_controller(phase):
    import psutil
    parent=psutil.Process().parent()
    ticket=json.loads(os.environ.get('FORENSICS_CONTROLLER','{}'))
    need(parent is not None,'supervised worker required')
    argv=parent.cmdline()
    need(ticket.get('pid')==parent.pid and ticket.get('created')==parent.create_time() and
         ticket.get('phase')==phase and ticket.get('binding')==binding() and
         len(argv)==3 and Path(argv[1]).resolve()==Path(__file__).resolve() and argv[2]=='run',
         'direct worker bypass rejected')
    if phase in PREREQUISITE:
        completed_phase(PREREQUISITE[phase])


def idle_reason(now,progress,sample=None,total_mib=None):
    """No-progress watchdog only (checklist item 2); GPU activity never resets it."""
    if now-progress>=POLICY['idle_seconds']:
        return 'no committed progress for 900 seconds'
    if (sample is not None and total_mib is not None and sample[1]>=95 and sample[2]<90
            and sample[0]>=total_mib-3072 and now-progress>=600):
        return 'WDDM stall signature and no committed progress for 600 seconds'
    return None


def memory(desktop,peaks):
    from tools.c85r7_flipall_timing import memory_admission
    return memory_admission(desktop,peaks)


def phase_count(phase):
    if phase in LENS_PHASES:
        return len(read(WORK/'gpu'/phase/'plan.json')['tasks'])
    return {'precision-calibration':12,'precision':POLICY['overflow_studies']*6,
            'calibration':12,'full':58*len(ARMS)}[phase]


def completed_phase(phase,count=None):
    folder=WORK/'gpu'/phase
    r=read(folder/'controller.json')
    if phase in LENS_PHASES:
        need(r.get('plan') is not None and r['plan']==read(folder/'plan.json'),'lens plan changed')
    expected=phase_count(phase) if count is None else count
    need(expected>0 and r.get('exit_code')==0 and r.get('cleanup_errors')==[] and r.get('binding')==binding() and
         len(r.get('journals',{}))==expected,'completed bound phase required')
    peaks=[]
    for name,pin in r['journals'].items():
        need(Path(name).name==name and sha(folder/'journal'/name)==pin,'phase journal changed')
        peaks.append(read(folder/'journal'/name)['peak_reserved_bytes'])
    memory(r['desktop'],peaks)
    if phase.startswith('precision'):
        need(r.get('result_sha256')==sha(folder/'result.json'),'precision phase result changed')
    return r


def supervise(phase,desktop):
    import psutil
    from tools import hb149_monitor_v125 as mon
    from tools.c85r7_flipall_timing import live_memory_guard
    folder=WORK/'gpu'/phase; folder.mkdir(parents=True,exist_ok=True)
    need(not (folder/'controller.json').exists(),'existing GPU attempt retained; review before rerun')
    plan=None
    if phase in LENS_PHASES:
        L=lens()
        output,reason=L.hb162_outputs() if phase.startswith('release') else (None,'mil phase')
        plan=dict(phase=phase,tasks=L.plan(phase,output is not None),hb162_unavailable_reason=reason)
        io.atomic_json(folder/'plan.json',plan)
    env=os.environ.copy()
    env['FORENSICS_CONTROLLER']=json.dumps(dict(pid=os.getpid(),created=psutil.Process().create_time(),phase=phase,binding=binding()))
    child=job=None; tracked={}; code=1; error=None; hashes={}; cleanup=[]
    last_progress=time.monotonic(); previous=-1; started=time.monotonic()
    try:
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_worker','--phase',phase],
                stdout=log,stderr=subprocess.STDOUT,env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
                start_new_session=os.name!='nt')
            proc=psutil.Process(child.pid); tracked[proc.pid]=proc; job=mon._WindowsJob(child)
            while child.poll() is None:
                now=time.monotonic()
                # Only status is read while child runs; never live journal/listing.
                status=io.read_receipt(folder/'status.json')
                if status.get('completed',-1)>previous:
                    previous=status['completed']; last_progress=now
                sample=None
                try:
                    sample=mon._gpu_sample(env)
                except (OSError,ValueError,subprocess.SubprocessError):
                    pass  # telemetry best effort; the no-progress watchdog still applies
                if sample is not None:
                    live_memory_guard(sample[0],desktop)
                reason=idle_reason(now,last_progress,sample,desktop['total_mib'])
                need(reason is None,reason)
                try:
                    for p in proc.children(recursive=True): tracked[p.pid]=p
                except psutil.NoSuchProcess: pass
                time.sleep(1)
            code=child.wait()
            need(code==0,'worker failed; retained run.log')
            rows=sorted((folder/'journal').glob('*.json'))
            expected=len(plan['tasks']) if plan else phase_count(phase)
            need(len(rows)==expected,'incomplete GPU phase')
            memory(desktop,[read(p)['peak_reserved_bytes'] for p in rows])
            hashes={p.name:sha(p) for p in rows}
    except BaseException as exc:
        code=1; error=f'{type(exc).__name__}: {exc}'
    finally:
        cleanup=mon._cleanup(child,tracked,job,psutil)
        io.receipt(folder/'controller.json',dict(exit_code=code,error=error,cleanup_errors=cleanup,
            phase=phase,desktop=desktop,binding=binding(),journals=hashes,idle_only=True,no_wall_cap=True,
            plan=plan,elapsed_seconds=time.monotonic()-started,
            result_sha256=sha(folder/'result.json') if phase.startswith('precision') and (folder/'result.json').exists() else None))
    need(code==0 and not cleanup,error or 'cleanup failed')


def evaluate():
    import numpy as np
    from tools.exp130_gold58_compare import official_gold
    reg=registered(); admit()
    for phase in ('calibration','full'):
        completed_phase(phase)
    try:
        completed_phase('precision-calibration'); completed_phase('precision')
        overflow=read(WORK/'gpu/precision/result.json')
    except (OSError,ValueError,KeyError) as exc:
        overflow=dict(registered_reading='UNAVAILABLE',reason=f'precision group incomplete: {exc}')
    ids,y=official_gold(ROOT); need(ids==reg['ids'],'Gold roster order changed')
    raw_identity_check(WORK/'gpu/full', len(ids))
    arms={}
    decode_equal=True
    for arm in ARMS:
        rows=[read(WORK/'gpu/full/journal'/f'{arm}-{i:03d}.json') for i in range(len(ids))]
        need([r['uid'] for r in rows]==ids and all(r['arm']==arm and
             r['registration_sha256']==sha(WORK/'registration.json') for r in rows),'prediction identity/order')
        arms[arm]=valid_probs([r['probabilities'] for r in rows],len(ids))
        decode_equal &= all(r['input']['cache_equal'] for r in rows)
    comparisons={a:bootstrap(y,arms[a],arms[p],POLICY['bootstrap_draws'],POLICY['bootstrap_seed']) for a,p in PAIRS.items()}
    from tools import exp156_k_sweep as e
    jev=e._reference(ROOT/e.JEV, ids, 'binary')
    jev_comparisons={a:bootstrap(jev,arms[a],arms[p],POLICY['bootstrap_draws'],POLICY['bootstrap_seed']) for a,p in PAIRS.items()}
    parity=float(np.max(abs(arms['HB156']-arms['HB156_train_eval'])))
    result=dict(status='RESULT_REVIEW_REQUIRED',registration_sha256=sha(WORK/'registration.json'),
        policy=POLICY,comparisons=comparisons,jev_exposed_diagnostic=jev_comparisons,
        overflow_diagnostic=overflow,lens_role='Gold58 paired student-vs-parent = L7 lens for the three kill tests',
        hb156_max_abs_probability_difference=parity,
        hb156_parity_pass=parity<=POLICY['parity_max_abs'],all_raw_cache_equal=decode_equal,
        four_rulers=reg['four_rulers'],gold_exposure=reg['gold_exposure'],
        public_R_upstream=reg['public_R_upstream'],independent_result_review_required=True)
    io.atomic_json(WORK/'gpu-result.json',result)
    return dict(status=result['status'],studies=58,parity_max_abs=parity,
                overflow_reading=overflow.get('registered_reading'))


def prepare():
    reg=registered()
    for arm in ARMS[:-1]: payload(arm,True)
    from tools.gpu_queue import validate_plan
    # HB162 must have ended, not succeeded (also enforced by require_gpu_job).
    plan=[dict(tag=TAG,command=[PYTHON,str(Path(__file__).resolve()),'run'],after=PREDECESSOR,continue_on_failure=True)]
    validate_plan(plan); io.atomic_json(WORK/'queue.json',plan)
    return dict(status='PREPARED_CPU_ONLY',queue_started=False,studies=len(reg['ids']))


def run_all():
    """One controller: independent phase groups; a failed group never blocks the others."""
    admit(); require_gpu_job()
    from tools.c85r7_flipall_timing import desktop_snapshot
    groups={}
    for group in GROUPS:
        try:
            for phase in group:
                supervise(phase,desktop_snapshot()); completed_phase(phase)
            groups[group[-1]]='COMPLETE'
        except Exception as exc:
            groups[group[-1]]=f'FAILED: {type(exc).__name__}: {exc}'
        io.atomic_json(WORK/'gpu/run-summary.json',dict(version=VERSION,tag=TAG,groups=groups))
    result=dict(version=VERSION,tag=TAG,groups=groups)
    if groups.get('full')=='COMPLETE':
        try: result['evaluation']=evaluate()
        except Exception as exc: result['evaluation']=f'FAILED: {type(exc).__name__}: {exc}'
    io.atomic_json(WORK/'gpu/run-summary.json',result)
    need(all(v=='COMPLETE' for v in groups.values()) and isinstance(result.get('evaluation'),dict),
         'one or more phase groups failed; retained receipts in gpu/run-summary.json')
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['register','validate','prepare','cpu-analysis','cpu-smoke',
        'lens-register','lens-validate','lens-cpu-smoke','admit','run','_worker','evaluate'])
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--phase',choices=['precision-calibration','precision','calibration','full',*LENS_PHASES])
    args=ap.parse_args(argv)
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if args.command not in ('run','_worker'): os.environ['CUDA_VISIBLE_DEVICES']='-1'
    WORK.mkdir(parents=True,exist_ok=True)
    if args.command=='register':
        io.atomic_json(WORK/'registration.json',registration()); result=dict(status='REGISTERED_CPU_ONLY')
    elif args.command=='validate': registered(); result=dict(status='VALIDATED_CPU_ONLY')
    elif args.command=='prepare': result=prepare()
    elif args.command=='cpu-analysis': result=cpu_analysis()
    elif args.command=='cpu-smoke': result=cpu_smoke(); result={k:v for k,v in result.items() if 'journals' not in k}
    elif args.command.startswith('lens-'):
        L=lens()
        L.main([args.command.removeprefix('lens-')])
        return
    elif args.command=='admit': result=admit(args.dry_run)
    elif args.command=='evaluate': result=evaluate()
    elif args.command=='_worker':
        need(args.phase is not None,'phase required'); require_controller(args.phase); admit(); require_gpu_job()
        reg=registered(); folder=WORK/'gpu'/args.phase
        if args.phase in LENS_PHASES:
            L=lens(); L.registered()
            output,_=L.hb162_outputs() if args.phase.startswith('release') else (None,None)
            if args.phase.startswith('mil'):
                os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')  # strict runtime before CUDA init
            cap_memory('cuda:0')
            L.worker(args.phase,'cuda:0',folder,output)
        elif args.phase.startswith('precision'):
            count=2 if args.phase=='precision-calibration' else POLICY['overflow_studies']
            execute_precision('cuda:0',reg['ids'][:count],folder)
        else:
            execute('cuda:0',reg['ids'][:2] if args.phase=='calibration' else reg['ids'],folder)
        result=dict(status='WORKER_COMPLETE')
    else:
        result=run_all()
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__': main()
