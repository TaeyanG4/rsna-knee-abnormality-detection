"""H-B-166 OAI inputs v1.0.1: CPU cache adapter and registered readout plan.

Changelog 2026-10-15: reuse authenticated H-B-147 native448, render ONLY
missing side-v2 compartments through unchanged H-B-155 geometry functions.
No GPU/admission/network commands. No label rows are exported or printed.
v1.0.1: exact audited picks, immutable finished resume, CPU review gate,
aggregate-only real label validation and corrected control forward signature.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools import hb147_oai_native448_v1 as native
from tools import hb155_crop_v1 as crop
from tools import hb155_laterality_v1 as side
# Historical side module sets -1; empty must be restored before torch use.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from tools import hb145_output_io as io
from tools.extcache import common, oai_cache_v2 as oai

VERSION = '1.0.1'
ROOT = native.ROOT
WORK = ROOT / '.local/scratch/hb166-oai/H-B-166'
OUT = WORK / 'crop448'
TRAIN_CROP = ROOT / '.local/scratch/slot-b/hb155/crop-cache-v220'
need, sha, digest = native.require, native.sha, native.digest
CODE = (*native.CODE, 'tools/hb166_oai_inputs_v1.py', 'tools/hb155_crop_v1.py',
        'tools/hb155_laterality_v1.py', 'tools/hb155_cache_v2.py',
        'tools/hb150_io_v1.py', 'training/arms/hb166_multiview_v1.py',
        'training/arms/hb155_crops_v1.py', 'training/arms/native_resolution_v2.py',
        'training/arms/mil_screen_hooks_v3.py', 'training/arms/mil.py',
        'tools/lens_completion_1015_v1.py', 'tools/multi_lens_readout.py',
        'tools/oai_moaks_panel.py')
RULE = dict(panel='frozen557, one V00 knee/participant, smallest SHA256(study_key)',
            seed_mean='float32 arithmetic probabilities within fold; never rank',
            weak_indices=[3, 5, 6, 8], threshold=.002, draws=300, seed=20261003,
            missing='MCL/Contusion/Fracture and intermediate OA are unknown, never negative',
            eligible='requires independently reviewed OAI-free checkpoint ancestry AND valid cache/inference',
            folds='separate seed-pair deltas; descriptive equal-weight fold deltas only, no pooled probabilities',
            ci='participant singleton paired bootstrap; CI descriptive, point threshold only',
            gold58='EXPOSED/descriptive/ineligible', clean_oof=False,
            final_pair='L7 does not override existing fold gates or authorize submission')


def read(path):
    return io.read_json(Path(path))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def smoke_ids(ids):
    """Deterministic first L and first R, in canonical panel order."""
    chosen = {next(u for u in ids if f'_{s}_' in u) for s in ('L', 'R')}
    return [u for u in ids if u in chosen]


def registration():
    ctx = native.context()
    nm = read(native.OUT / 'manifest.json')
    cm = read(TRAIN_CROP / 'manifest.json')
    need(nm['finished'] and nm['contract'] == native.CONTRACT and nm['binding'] == ctx['binding'],
         'native metadata/source binding differs')
    need(cm['finished'] and cm['contract'] == side.CONTRACT, 'training crop contract differs')
    need(cm['spec']['slots'] == ctx['spec']['slots'] and cm['spec']['img'] == 448,
         'training view layout differs')
    configs = {}
    for stage, seeds in (('pilot-v100', (42, 43)), ('pilot-fold3-v110', (44, 45))):
        for seed in seeds:
            p = ROOT / f'.local/scratch/slot-b/hb166/{stage}/configs/multiview-s{seed}.json'
            c = read(p)
            need(c['corpus_contract'] == native.CONTRACT and c['crop_corpus_contract'] == side.CONTRACT
                 and c['resolution'] == 224, 'fit input contract differs')
            need(sha(ROOT / c['control_config']) == c['control_config_sha256'], 'control config drift')
            configs[rel(p)] = sha(p)
            configs[c['control_config']] = c['control_config_sha256']
    return dict(version=VERSION, native=rel(native.OUT), native_manifest_sha256=sha(native.OUT/'manifest.json'),
                source_binding=ctx['binding'], training_crop_manifest_sha256=sha(TRAIN_CROP/'manifest.json'),
                crop_contract=side.CONTRACT, spec=cm['spec'], configs=configs, rule=RULE,
                side_policy='audited OAI study/GENERAL laterality; exact side.permutation/updated_record; no classifier',
                renderer='unchanged crop.build_study; GENERAL geometry/pixels, audited strict slot selection',
                code={p: sha(ROOT/p) for p in CODE}, studies=557,
                projected_pixel_bytes=557*64*448*448, native_copied_bytes=0)


def register():
    WORK.mkdir(parents=True, exist_ok=True)
    reg = registration()
    p = WORK/'registration.json'
    if p.exists():
        need(read(p) == reg, 'registration drift; review and version before changing')
    else:
        io.atomic_json(p, reg)
    return reg


class GeneralAdapter:
    """Only the raw IO boundary changes. Localization/crop arithmetic is shared."""
    def __init__(self, rows, uid):
        match = re.fullmatch(r'OAI_([0-9]+)_([RL])_(V00)', uid)
        need(match is not None, 'invalid audited OAI study identity')
        self.side = match[2]
        self.headers, self.pixels, self.members, self.rows = {}, {}, [], []
        for row in rows:
            need(row['study_key'] == uid and row['knee_side'] == self.side,
                 'GENERAL laterality/study disagreement')
            s = common.load_series(row, 'oai', oai.GEN_ROOT)
            need(s.vol.ndim == 3 and s.ipp.shape == (len(s.vol), 3) and s.iop.shape == (6,)
                 and s.pixel_spacing.shape == (2,) and np.isfinite(s.vol).all(), 'GENERAL array geometry')
            sid = row['series_key']
            self.rows.append(dict(SeriesInstanceUID=sid, Anatomical_Plane=row['plane'],
                                  Fluid_Sensitive=row['fluid_sensitive']))
            for z, pixels in enumerate(s.vol):
                key = f'{sid}/{z:06d}'
                need(key not in self.headers, 'duplicate slice adapter key')
                self.members.append((key, sid, z))
                self.headers[key] = SimpleNamespace(ImageOrientationPatient=s.iop,
                    ImagePositionPatient=s.ipp[z], PixelSpacing=s.pixel_spacing,
                    Rows=pixels.shape[0], Columns=pixels.shape[1], ImageLaterality=self.side)
                self.pixels[key] = pixels.astype(np.float32)
        self._WORKER = dict(con=self, pydicom=self)

    def execute(self, query, params):
        return self

    def fetchall(self):
        return self.members

    def dcmread(self, stream, stop_before_pixels):
        need(stop_before_pixels, 'headers only adapter')
        return self.headers[stream.read().decode()]

    def _member_bytes(self, path, info):
        return path.encode()

    def _read_pixels(self, path, info):
        return self.pixels[path]

    pick_series = staticmethod(common.pick_strict)


def render(ctx, uid, spec):
    rows = oai.plan_rows(ctx['plans'][uid], ctx['rows'])
    raw = GeneralAdapter(rows, uid)
    volume, mask, record = crop.build_study(raw, ('oai', uid, raw.rows), spec)
    expected = [p['group_key'] is not None for p in ctx['plans'][uid]]
    need([s['series'] is not None for s in record['slots']] == expected, 'audited slot availability differs')
    oai._check_picks(ctx['plans'][uid], rows, record['slots'], ctx['audit'].groups)
    need(record['laterality'] == raw.side, 'rendered laterality differs')
    resolution = side.resolve({raw.side}, None)
    resolution['source'] = 'audited_OAI_GENERAL_study_identity'
    perm = side.permutation(spec, raw.side)
    return volume[perm], mask[perm], dict(uid=uid, **side.updated_record(record, resolution, spec))


def verify_shard(directory, item, i, uid):
    need(item['start'] == i and item['stop'] == i+1 and item['file'] == f'vols_shard{i:04d}.npy',
         'crop shard bounds/path drift')
    need(item['qa_file'] == f'qa-{i:04d}.json', 'QA path drift')
    for name, field in ((item['file'], 'sha256'), (item['qa_file'], 'qa_sha256')):
        need(sha(directory/name) == item[field], 'crop shard/QA hash drift')
    v = np.load(directory/item['file'], mmap_mode='r', allow_pickle=False)
    need(v.shape == (1,64,448,448) and v.dtype == np.uint8, 'crop shape/dtype')
    need((directory/item['file']).stat().st_size == item['bytes'], 'crop byte count')
    q = read(directory/item['qa_file'])
    need(q['uid'] == uid, 'QA roster drift')
    return (v[0].reshape(64,-1).sum(1)>0).astype(np.uint8), q


def check_review(reg):
    p = WORK/'pre-review.json'
    need(p.exists(), 'independent CPU prepare pre-review missing')
    r = read(p)
    need(r.get('verdict') == 'PASS' and r.get('reviewer') != 'builder' and bool(r.get('reviewer'))
         and r.get('registration_sha256') == digest(reg)
         and r.get('scope') == 'CPU crop prepare only; no GPU/admit', 'CPU pre-review binding differs')
    report = WORK/r['report']
    need(report.resolve().is_relative_to(WORK.resolve()) and sha(report) == r['report_sha256'],
         'CPU review report drift')


def prepare(*, smoke=False):
    reg = register()
    if not smoke:
        check_review(reg)
    ids = native.panel_ids()
    if smoke:
        ids = smoke_ids(ids)
    directory = WORK/'smoke-crop448' if smoke else OUT
    directory.mkdir(parents=True, exist_ok=True)
    ctx = native.context(ids)
    mp = directory/'manifest.json'
    if mp.exists():
        m = read(mp)
        need(m['binding_sha256'] == digest(reg) and m['ids'] == ids, 'resume registration/roster drift')
    else:
        m = dict(version=VERSION, contract=side.CONTRACT, spec=reg['spec'], ids=ids,
                 binding_sha256=digest(reg), studies=len(ids), shape_total=[len(ids),64,448,448],
                 dtype='uint8', shards=[], finished=False, exposure='EXPOSED', clean_oof=False)
        io.atomic_json(mp,m)
    masks=[]
    for i,s in enumerate(m['shards']):
        mask,_=verify_shard(directory,s,i,ids[i]); masks.append(mask)
    if m['finished']:
        validate(smoke=smoke, verify_native=False)
        return dict(status='REUSED_FINISHED',studies=len(ids))
    need(shutil.disk_usage(directory).free >= (len(ids)-len(masks))*64*448*448+10*(1<<30),
         'insufficient disk headroom')
    started=time.monotonic()
    for i in range(len(masks),len(ids)):
        v,mask,q=render(ctx,ids[i],reg['spec'])
        name=f'vols_shard{i:04d}.npy'; qname=f'qa-{i:04d}.json'
        # Only fixed task-owned uncommitted files can be overwritten on recovery.
        native.atomic_array(directory/name,v[None]); io.atomic_json(directory/qname,q)
        m['shards'].append(dict(file=name,start=i,stop=i+1,sha256=sha(directory/name),
            bytes=(directory/name).stat().st_size,qa_file=qname,qa_sha256=sha(directory/qname),done=True))
        masks.append(mask); io.atomic_json(mp,m)
        status=dict(phase='crop_commit',completed=i+1,total=len(ids),seconds=round(time.monotonic()-started,3))
        try:
            io.atomic_json(directory/'status.json',status)
        except OSError:
            pass  # A status write cannot invalidate a durable manifest commit.
        if i%10==0 or i+1==len(ids): print(json.dumps(status),flush=True)
    for name,value in (('ids',np.asarray(ids)),('masks',np.asarray(masks,np.uint8)),
                       ('side_unknown',np.zeros((len(ids),64,1,1),np.uint8))):
        native.atomic_array(directory/f'{name}.npy',value)
        m[name+'_sha256']=sha(directory/f'{name}.npy')
    m.update(finished=True,seconds_this_run=time.monotonic()-started)
    io.atomic_json(mp,m)
    return dict(status='PREPARED',studies=len(ids),pixel_file_bytes=sum(s['bytes'] for s in m['shards']))


def validate(*, smoke=False, verify_native=True):
    reg=register(); directory=WORK/'smoke-crop448' if smoke else OUT
    ids=smoke_ids(native.panel_ids()) if smoke else native.panel_ids()
    m=read(directory/'manifest.json')
    need(m['finished'] and m['binding_sha256']==digest(reg) and m['ids']==ids and
         m['contract']==side.CONTRACT and m['spec']==reg['spec'] and
         m['shape_total']==[len(ids),64,448,448] and len(m['shards'])==len(ids),'crop manifest drift')
    arrays={}
    for name in ('ids','masks','side_unknown'):
        need(sha(directory/f'{name}.npy')==m[name+'_sha256'],'array hash drift')
        arrays[name]=np.load(directory/f'{name}.npy',allow_pickle=False)
    need(arrays['ids'].tolist()==ids,'array roster differs')
    need(arrays['masks'].shape==(len(ids),64) and arrays['masks'].dtype==np.uint8 and
         np.isin(arrays['masks'],[0,1]).all(),'mask contract differs')
    need(arrays['side_unknown'].shape==(len(ids),64,1,1) and arrays['side_unknown'].dtype==np.uint8
         and not arrays['side_unknown'].any(),'audited side flags differ')
    counts={}; padding=0.; missing=0
    for i,s in enumerate(m['shards']):
        mask,q=verify_shard(directory,s,i,ids[i])
        need(np.array_equal(mask,arrays['masks'][i]),'mask differs from pixels')
        expected_side=ids[i].split('_')[2]
        need(q['laterality']==expected_side and q['side_resolution']['side']==expected_side,'QA side differs')
        for k,v in q['failures'].items(): counts[k]=counts.get(k,0)+v
        missing+=q['missing_series']
        for slot in q['slots']:
            for e in slot['crops']: padding=max(padding,e['padding_fraction'])
    metrics,violations=side.limits(counts,len(ids),int(arrays['masks'].sum()),padding,
                                  int((arrays['masks'].sum(1)==0).sum()))
    need(not violations,'crop QA HOLD: '+','.join(violations))
    nv=native.verify(native.OUT) if verify_native else None
    result=dict(status='VALIDATED',studies=len(ids),metrics=metrics,violations=violations,
        missing_series_masked=missing,manifest_sha256=sha(directory/'manifest.json'),
        native_verification=nv,native_manifest_sha256=sha(native.OUT/'manifest.json'),
        cache_bytes=sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
        scientific_admission='HOLD_INDEPENDENT_REVIEW_NO_GPU')
    io.atomic_json(directory/'validation.json',result)
    return result


def members():
    """Inspect only completed directories; defer absent/running fold3 outputs."""
    reg=register(); result=[]
    for stage,seeds in (('pilot-v100',(42,43)),('pilot-fold3-v110',(44,45))):
        for seed in seeds:
            mp=ROOT/f'.local/scratch/slot-b/hb166/{stage}/configs/multiview-s{seed}.json'
            c=read(mp)
            for arm,cp in (('multiview',mp),('control',ROOT/c['control_config'])):
                cfg=read(cp); directory=cp.parent.parent/'fits'/f'{arm}-s{seed}'
                entry=dict(arm=arm,seed=seed,fold=c['fold'],config=rel(cp),config_sha256=sha(cp),
                           checkpoint=rel(directory/'final.pt'),status='DEFERRED_COMPLETED_TRANSFER_REQUIRED')
                hp=directory/'heartbeat.json'
                if hp.exists() and read(hp).get('status')=='complete':
                    summary=read(directory/'summary.json'); ck=directory/'final.pt'
                    need(sha(ck)==summary['checkpoint_sha256'],'completed checkpoint digest differs')
                    identity=summary['identity']
                    need(identity['arm']==arm and identity['seed']==seed and identity['fold']==c['fold']
                         and identity['config_sha256']==sha(cp) and not identity['engineering_only'],
                         'completed checkpoint identity differs')
                    if arm=='control':
                        need(c['control_fits'][rel(ck)]==sha(ck),'registered control checkpoint differs')
                    entry.update(status='PINNED_COMPLETE',checkpoint_sha256=sha(ck),
                                 summary_sha256=sha(directory/'summary.json'),
                                 measured_readout_seconds=summary.get('validation_seconds'))
                # No OAI studies may occur in any fitted train/validation roster.
                train=list(cfg['train']); val=list(cfg['val'])
                need(not (set(train+val)&set(native.panel_ids())) and
                     all(not u.startswith('OAI_') for u in train+val),'OAI exposure in fit roster')
                entry['oai_roster_disjoint']=True
                entry['ancestry_status']='HOLD_INDEPENDENT_INITIAL_WEIGHTS_AND_SOFT_LABEL_PROVENANCE_REVIEW'
                result.append(entry)
    return result


def cpu_smoke():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from training.arms import hb166_multiview_v1 as mv
    from training.arms import mil_screen_hooks_v3 as h
    from training.arms import native_resolution_v2 as nr
    from training.arms import hb155_crops_v1 as crops
    torch.set_num_threads(2)
    need(not torch.cuda.is_initialized(),'CUDA already initialized')
    validate(smoke=True,verify_native=False)
    ids=smoke_ids(native.panel_ids()); ctx=native.context(ids); started=time.monotonic()
    corpus=h.common.Corpus(native.OUT); cc=h.common.Corpus(WORK/'smoke-crop448')
    zeros={u:np.zeros(12,np.float32) for u in ids}
    ds=mv.make_dataset(native.OUT,cc.dir,ids,zeros,36,False,42,224)
    for u in ids:
        v,_=native.render_checked(ctx,u)
        need(np.array_equal(v,corpus.volume(u)),'native byte replay differs')
        v,mask,_=render(ctx,u,cc.spec)
        need(np.array_equal(v,cc.volume(u)) and np.array_equal(mask,cc.mask(u)),'crop replay differs')
    pinned=[m for m in members() if m['seed']==42]
    proofs=[]
    for entry in pinned:
        need(entry['status']=='PINNED_COMPLETE','smoke needs completed checkpoint')
        cfg=read(ROOT/entry['config']); model=h.build_model(cfg['recipe']['model'],ROOT)
        if entry['arm']=='multiview': model=mv.attach_views(model)
        state=torch.load(ROOT/entry['checkpoint'],map_location='cpu',weights_only=True)
        need(state['identity']['config_sha256']==entry['config_sha256'] and state['epoch']==19
             and state['identity']['seed']==42 and state['identity']['arm']==entry['arm'],
             'checkpoint internal identity differs')
        h.strict_load(model,state['model']); del state
        model=model.eval().to(memory_format=torch.channels_last)
        with torch.inference_mode():
            for i,u in enumerate(ids):
                x,_,uid,views=ds[i]; need(x.shape==(72,3,224,224),'K36 per-view layout differs')
                ctrl=crops.Windows(h.MILWindows(native.OUT,ids,zeros,36,False,42),224)[i][0]
                need(torch.equal(ctrl,x[:36]),'control native windows differ')
                flipped=mv.flip_views(x[None],views[None],[u],corpus,cc,36)
                if entry['arm']=='multiview':
                    a=mv.ARM.forward(model,x[None],views[None]); b=mv.ARM.forward(model,flipped,views[None])
                else:
                    a=nr.NativeArm().forward(model,x[None,:36],None); b=nr.NativeArm().forward(model,flipped[:,:36],None)
                p=(a.float().sigmoid()+b.float().sigmoid())*.5
                need(p.shape==(1,12) and torch.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'invalid smoke probabilities')
        proofs.append(dict(arm=entry['arm'],checkpoint_sha256=entry['checkpoint_sha256'],studies=2,
                           k_per_view=36,flip_tta=True,finite=True))
        del model
    need(os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized(),'CPU isolation failed')
    result=dict(status='CPU_SMOKE_PASS',studies=2,sides=['L','R'],native_byte_replay=True,crop_replay=True,
        model_proofs=proofs,seconds=time.monotonic()-started,cuda_visible_devices='',cuda_initialized=False,
        engineering_only=True,real_oai_labels_used=False,registration_sha256=sha(WORK/'registration.json'))
    io.atomic_json(WORK/'cpu-smoke.json',result)
    return result


def label_check():
    from tools import oai_moaks_panel as panel
    ids = panel.validate()
    need(ids == native.panel_ids(), 'label roster differs')
    y = panel.outcomes(ids)
    need(y.shape == (557,12) and np.isnan(y[:,[1,10,11]]).all(), 'label missing-target mapping differs')
    p = np.full((557,12), .5, np.float32)
    r = l7(ids, {'s1':p,'s2':p}, {'s1':p,'s2':p}, y, eligible=False)
    need(r['weak4']['value'] == 0 and not r['weak4']['PASS'] and not r['weak4']['eligible'],
         'real-label equal-arm L7 replay differs')
    result = dict(status='LABEL_RULE_PASS',studies=557,targets=list(panel.TARGETS),
        counts=panel.counts(y),synthetic_predictions=True,weak4_equal_arm=r['weak4'],
        label_rows_exported=False,scientific_result=False)
    io.atomic_json(WORK/'label-check.json',result)
    return dict(status=result['status'],studies=557,label_rows_exported=False)


def l7(ids, arms, controls, y, *, eligible):
    """Pure registered per-fold readout; never read labels during inference."""
    from tools import lens_completion_1015_v1 as lens
    from tools import multi_lens_readout as ml
    need(len(arms)==len(controls)==2,'exactly two seeds per fold required')
    item=lens.image_item(ids,arms,controls,y,eligible,'one frozen knee per participant')
    w=ml.weights(ids,item['components'])
    delta=ml.auc_draws(item['arm'],item['labels'],item['mask'],w)-ml.auc_draws(item['control'],item['labels'],item['mask'],w)
    result=ml.stat(delta[:,ml.WEAK].mean(1),threshold=RULE['threshold'],eligible=eligible)
    return dict(weak4=result,mapped_nine=lens.mapped_nine(item),rule=RULE)


def plan():
    entries=members()
    # The same 4090 measurements are a lower-context reference, not a 4070 measurement.
    measured=[]
    for e in entries:
        d=e.get('measured_readout_seconds')
        if isinstance(d,dict) and 'val' in d:
            c=read(ROOT/e['config']); measured.append(float(d['val'])/len(c['val']))
    reference_seconds_per_study=max(measured) if measured else None
    minutes=None if reference_seconds_per_study is None else 557*8*reference_seconds_per_study/60
    result=dict(status='PLAN_ONLY_NO_ADMISSION',version=VERSION,registration_sha256=sha(WORK/'registration.json'),
        members=entries,rule=RULE,device='4070 Ti SUPER',batch=4,microbatch=2,workers=0,k_per_view=36,
        resolution=224,tta='per-view own-mask coronal/axial flip mean probability',amp='fp16 strict',
        order='serial 8 members; all arms on same GPU/OS/software; inference only',
        output='each member canonical557 x12 finite probabilities plus checkpoint/config/cache/runtime hashes',
        memory_gate=dict(peak_reserved_gib_max=12,headroom_gib=2,
            equation='measured desktop baseline + maximum full-shape/TTA peak + 2 GiB <= measured total'),
        watchdog=dict(idle_seconds=900,wddm_stall_seconds=600,wddm='util ~100%, power<90W, near-full VRAM',
            throughput_fraction=.25,throughput_window_seconds=1800,cleanup='verified owned process tree in finally'),
        launch='B only: reviewed outer wrapper through gpu_queue.py -> gpu_job.py; no admission in this build',
        prerequisites=['independent cache/smoke/code/result review PASS','completed authenticated fold3 transfer',
            'independent OAI-free ancestry PASS incl. soft-label teacher lineage',
            'reviewed GPU worker/outer wrapper; real4070 full-shape both-arm TTA memory gate',
            'actual device + Shared Active compute idle, claim before launch; provenance/runtime capture'],
        reference_seconds_per_study=reference_seconds_per_study,reference_eight_member_minutes=minutes,
        projected_4070_minutes=None if minutes is None else [round(minutes*1.5),round(minutes*3)],
        projection='planning estimate 1.5-3x slowest completed4090 per-study K36flip; replace with real4070 gate throughput',
        no_gpu_run=True,no_admit=True)
    io.atomic_json(WORK/'gpu-readout-plan.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','prepare','validate','cpu-smoke','label-check','plan'])
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.command=='register':
        r=register(); result=dict(status='REGISTERED',studies=r['studies'],pixel_bytes=r['projected_pixel_bytes'],
                                 native_copied_bytes=0,free_bytes=shutil.disk_usage(WORK).free)
    elif a.command=='prepare': result=prepare(smoke=a.smoke)
    elif a.command=='validate': result=validate(smoke=a.smoke)
    elif a.command=='cpu-smoke': result=cpu_smoke()
    elif a.command=='label-check': result=label_check()
    else:
        r=plan(); result={k:r[k] for k in ('status','projected_4070_minutes','no_gpu_run','no_admit')}
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    main()
