"""H-B-166 offline serving adapter 1.0.0. New file; no frozen source edits.

Uses authenticated, verbatim fitted definitions in private module dictionaries.
No notebook execution, downloads, thread settings or process changes.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import types

VERSION = '1.0.0'
DATASET = 'taeyangg4/knee-hb166-multiview-v1-assets'
SOURCE_SHA = 'BUILD_SOURCE_SHA'
MANIFEST_SHA = 'BUILD_MANIFEST_SHA'
TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
           'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def recipe(source_path, expected=SOURCE_SHA):
    """Exec only sealed original definition spans; no import-time fit machinery."""
    need(sha(source_path) == expected, 'HB166 source payload drift')
    import contextlib
    import random
    import numpy as np
    import torch
    payload = read(source_path)
    modules = {}
    for key, entry in payload['modules'].items():
        module = types.ModuleType('_hb166_' + key)
        module.__dict__.update(np=np, torch=torch, nn=torch.nn, Dataset=torch.utils.data.Dataset,
            Path=Path, hashlib=hashlib, json=json, random=random, contextlib=contextlib,
            io=io, time=time, need=need, sha=sha, **modules)
        if key in ('mil',):
            module.__dict__.update({k:v for k,v in modules['tokens'].__dict__.items() if not k.startswith('__')})
            module.__dict__['common'] = modules['common']
        if key in ('native', 'crops', 'mv'):
            module.__dict__['h'] = modules['mil']
        if key == 'crops':
            module.__dict__.update(CONTRACT=modules['geometry'].CONTRACT, MILWindows=modules['mil'].MILWindows)
        for span in entry['spans']:
            data = base64.b64decode(span['bytes'])
            need(hashlib.sha256(data).hexdigest() == span['sha256'], 'definition span drift')
            exec(compile(data, entry['path'] + ':' + span['name'], 'exec'), module.__dict__)
        modules[key] = module
    return types.SimpleNamespace(**modules)


class MemoryCorpus:
    """Transport adapter only. Pixels/masks already produced by fitted code."""
    def __init__(self, uid, volume, mask, spec, contract, unknown=0):
        import numpy as np
        need(volume.shape == (64,448,448) and volume.dtype == np.uint8, 'volume contract')
        need(mask.shape == (64,) and np.isin(mask,[0,1]).all(), 'mask contract')
        need(unknown in (0,1), 'side flag contract')
        self.ids, self.row, self.spec, self.contract = [uid], {uid:0}, spec, contract
        self._volume, self._mask = volume, mask
        self._side_flags = np.full((1,64,1,1), unknown, np.uint8)
        self.dir = Path('.')

    def volume(self, uid):
        need(uid in self.row, 'unknown study')
        return self._volume

    def mask(self, uid):
        need(uid in self.row, 'unknown study')
        return self._mask


class DirectoryIndex:
    def __init__(self, root):
        self.root = Path(root)

    def execute(self, query, args):
        need('FROM files WHERE split=? AND uid=? ORDER BY path' in query.replace(', ',','), 'unexpected index query')
        split, uid = args
        # Competition serving uses test_series; frozen raw archives use train_images.
        directory = self.root / (split + '_series')
        if not directory.is_dir():
            directory = self.root / (split + '_images')
        study = directory / uid
        rows = [(p.as_posix(), p.parent.name, '{}') for p in sorted(study.glob('*/*.dcm'))]
        return types.SimpleNamespace(fetchall=lambda: rows)


def configure_transport(r, root):
    import cv2
    import pydicom
    from pydicom.pixels import apply_modality_lut
    r.raw._WORKER.update(con=DirectoryIndex(root), cv2=cv2, pydicom=pydicom, lut=apply_modality_lut)
    # Only archive byte transport is substituted; rendering definitions untouched.
    r.raw._member_bytes = lambda path, info: Path(path).read_bytes()


def preprocess(r, uid, rows, split='test'):
    spec = dict(r.common.CONTRACTS['raptor64_336_c140_s02'], depth=64)
    r.raw._WORKER['spec'] = spec
    _, _, _, full_mask, _ = r.raw.build_study((split,uid,rows))
    spec = dict(spec, img=448)
    r.raw._WORKER['spec'] = spec
    _, _, full, _, picks = r.raw.build_study((split,uid,rows))
    crop, crop_mask, provenance = r.geometry.build_study(r.raw,(split,uid,rows),spec)
    need('laterality_conflict' not in provenance['failures'], 'contradictory laterality')
    # Missing/undecodable slices remain masked exactly as in the fitted caches.
    need(full_mask.any() and crop_mask.any(), 'empty full/crop view')
    f = MemoryCorpus(uid,full,full_mask,spec,r.mv.FULL_CONTRACT)
    c = MemoryCorpus(uid,crop,crop_mask,spec,r.mv.CROP_CONTRACT,int(provenance['laterality']=='unknown'))
    return f,c,dict(full_picks=picks,crop=provenance)


def dataset(r, ids, full, crop):
    import numpy as np
    ds = r.mv.make_dataset('.', '.', ids, {u:np.zeros(12,np.float32) for u in ids},36,False,42,224)
    ds.full.corpus, ds.crop.corpus = full,crop
    return ds


class Progress:
    def phase_start(self, phase):
        pass
    def completed(self, phase, counts):
        pass


def model(r, path, entry, device):
    import torch
    need(entry['status']=='PRESENT' and sha(path)==entry['sha256'], 'checkpoint missing or hash mismatch')
    state = torch.load(path,map_location='cpu',weights_only=True)
    need(state['model_kind']=='hb166_multiview_finding_mil_v1' and state['identity']==entry['identity']
         and state['epoch']==19, 'checkpoint identity mismatch')
    m = r.mv.attach_views(r.mil.FindingMIL(r.common.build_backbone('convnext_small.fb_in22k_ft_in1k',pretrained=False)))
    r.mil.strict_load(m,state['model'])
    return m.to(device=device,memory_format=torch.channels_last).eval()


def predict(r, m, ids, full, crop, device, progress=None):
    import torch
    ds = dataset(r,ids,full,crop)
    loader = torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=r.mv.collate)
    value, order, _ = r.mv.evaluate(r.mv.ARM,m,loader,device,'fp16' if device=='cuda' else 'fp32',
        full,crop,progress or Progress(),tta=True)
    need(order==list(ids), 'prediction order mismatch')
    return value


def roster(asset_dir, expected=MANIFEST_SHA):
    asset_dir = Path(asset_dir)
    need(sha(asset_dir/'manifest.json')==expected, 'HB166 manifest drift')
    m = read(asset_dir/'manifest.json')
    need(m['exposure']=='EXPOSED' and m['clean_oof'] is False and m['slot']=='Q_A5q', 'slot identity drift')
    need(m['targets']==TARGETS, 'target order drift')
    need([(e['fold'],e['seed']) for e in m['roster']]==[(2,42),(3,44)], 'roster drift')
    for e in m['roster']:
        need(e['status']=='PRESENT', 'required fold missing')
        need(sha(asset_dir/e['file'])==e['sha256'], 'required checkpoint drift')
    return m


def run(c85, competition, output, asset_dir=None, device='cuda', progress=None):
    """Same Q phase/process; an asset/member failure masks the complete slot."""
    import numpy as np
    import pandas as pd
    output, competition = Path(output), Path(competition)
    ids = c85.read_json(output/'ids.json')
    c85.v5q.validate_ids(ids)
    binding = dict(model='HB166_MULTIVIEW', manifest_sha256=MANIFEST_SHA, source_sha256=SOURCE_SHA,
        exposure='EXPOSED',clean_oof=False,slot='Q_A5q',targets=TARGETS)
    journal = c85.Journal(output/'hb166-journal',ids,binding)
    started = time.monotonic()
    asset_dir = Path(asset_dir) if asset_dir else Path('/kaggle/input')/DATASET.split('/')[1]
    try:
        need(list(c85.p.LABELS)==TARGETS, 'parent target order drift')
        manifest = roster(asset_dir)
        r = recipe(Path(__file__).with_name('hb166-sources.json'))
        configure_transport(r,competition)
        rows = pd.read_csv(competition/'test_series.csv',dtype=str)
        models = [model(r,asset_dir/e['file'],e,device) for e in manifest['roster']]
        for uid in ids:
            try:
                f,c,_ = preprocess(r,uid,rows.loc[rows.StudyInstanceUID==uid].to_dict('records'))
                values = np.mean(np.stack([predict(r,m,[uid],f,c,device,progress)[0] for m in models]),axis=0,dtype=np.float32)
                c85.p.RBinding.prediction(values)
                journal.record(uid,'HB166',values=values.tolist())
            except Exception as exc:
                journal.record(uid,'HB166',error=type(exc).__name__+': '+str(exc))
    except Exception as exc:
        for uid in ids:
            journal.record(uid,'HB166',error=type(exc).__name__+': '+str(exc))
    c85.write_once(output/'q-timings.json',c85.json_bytes(dict(model='HB166_MULTIVIEW',seconds=time.monotonic()-started)))


def authenticated(c85,directory,ids):
    import numpy as np
    binding = dict(model='HB166_MULTIVIEW',manifest_sha256=MANIFEST_SHA,source_sha256=SOURCE_SHA,
        exposure='EXPOSED',clean_oof=False,slot='Q_A5q',targets=TARGETS)
    journal = Path(directory)/'hb166-journal'
    if not (journal/'journal-identity.json').exists():
        return np.full((len(ids),12),.5),dict.fromkeys(ids,'MissingPrediction: HB166 child aborted')
    need(list(c85.p.LABELS)==TARGETS, 'parent target order drift')
    need(c85.read_json(journal/'ids.json')==list(ids), 'journal order drift')
    values, failures = c85.read_journal(journal,ids,['HB166'],binding)
    return values['HB166'],failures['HB166']
