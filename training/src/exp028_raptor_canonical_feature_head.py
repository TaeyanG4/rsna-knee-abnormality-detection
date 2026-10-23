try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import json,time
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F,timm
from sklearn.metrics import roc_auc_score
ROOT=workspace_root()
CACHE=ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'
ORI=ROOT/'kaggle_notebooks'/'own_exp027_orientation_metadata'/'orientation_metadata.csv'
WEIGHT=ROOT/'weights'/'dreaddevelopment__raptor-knee-maxspan'/'raptor_ft_coatnet_v5_full_swa.pt'
OUT=ROOT/'experiments'/'EXP028_RAPTOR_CANONICAL_FEATURE_HEAD'; OUT.mkdir(parents=True,exist_ok=True)
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
SLOTS=['SAG_FLUID','SAG_STRUCT','COR_FLUID','COR_STRUCT','AX_FLUID','AX_STRUCT']

def build_backbone(arch):
    hybrid=arch.startswith(('maxvit','maxxvit','coatnet','coat_','convnext')); is_vit=(not hybrid) and any(k in arch for k in ('vit','deit','dinov2','eva','beit'))
    kw=dict(pretrained=False,num_classes=0,in_chans=3); kw.update(global_pool='token',dynamic_img_size=True) if is_vit else kw.update(global_pool='avg'); return timm.create_model(arch,**kw)

class Raptor(nn.Module):
    def __init__(self,bb,Fdim,n=12):
        super().__init__(); self.backbone=bb; self.norm=nn.LayerNorm(Fdim); self.att=nn.Sequential(nn.Linear(Fdim,256),nn.Tanh(),nn.Dropout(.2),nn.Linear(256,n)); self.clsW=nn.Parameter(torch.zeros(n,Fdim)); self.clsb=nn.Parameter(torch.zeros(n))
    def masked_head(self,f,mask):
        h=self.norm(f); s=self.att(h).masked_fill(mask[...,None]<=0,-1e4); a=torch.softmax(s,dim=1); pooled=torch.einsum('bkn,bkf->bnf',a,h); return (pooled*self.clsW).sum(-1)+self.clsb

class Shards:
    def __init__(self):
        m=json.loads((CACHE/'train_cache_shards_manifest.json').read_text()); self.s=[(r['start'],r['stop'],np.load(CACHE/r['file'],mmap_mode='r')) for r in m['shards']]
    def get(self,i):
        for a,b,x in self.s:
            if a<=i<b:return x[i-a]
        raise IndexError

def load_transforms(n):
    d=pd.read_csv(ORI)
    rev=np.zeros((n,6),np.uint8); fh=np.zeros((n,6),np.uint8); fv=np.zeros((n,6),np.uint8)
    smap={s:i for i,s in enumerate(SLOTS)}
    for r in d.itertuples(index=False):
        i=int(r.index); s=smap[r.slot]
        if not (pd.notna(r.header_ok) and float(r.header_ok)==1): continue
        rev[i,s]=int(float(r.reverse_slice_to_canonical)>0.5)
        # DICOM first IOP triplet = increasing image column (horizontal); second = increasing image row (vertical)
        fh[i,s]=int(float(r.row_dot_desired_h)<0)
        fv[i,s]=int(float(r.col_dot_desired_v)<0)
    return rev,fh,fv

def canonicalize(x,mask,rev,fh,fv):
    # x 6,9,H,W; all transforms are exact index operations, no interpolation
    y=np.array(x,copy=True); m=np.array(mask,copy=True)
    for s in range(6):
        if rev[s]: y[s]=y[s,::-1]; m[s]=m[s,::-1]
        if fh[s]: y[s]=y[s,:,:,::-1]
        if fv[s]: y[s]=y[s,:,::-1,:]
    return y,m

def windows_fixed(x,mask):
    ws=np.zeros((42,3,224,224),np.float32); wm=np.zeros(42,np.float32); q=0
    for s in range(6):
        avail=mask[s].sum()>0
        for c in range(1,8):
            if avail:
                ws[q]=x[s,[c-1,c,c+1]].astype(np.float32)/255.; wm[q]=1
            q+=1
    return ws,wm

def aucs(y,p):
    a=[float(roc_auc_score(y[:,k],p[:,k])) for k in range(12)]; return a,float(np.mean(a))

def main():
    u=pd.read_csv(CACHE/'train_cache_uids.csv'); ids=u.StudyInstanceUID.astype(str).tolist(); masks6=np.load(CACHE/'train_cache_masks.npy'); sh=Shards(); rev,fh,fv=load_transforms(len(ids))
    print('transforms',{'reverse':int(rev.sum()),'flip_h':int(fh.sum()),'flip_v':int(fv.sum())},flush=True)
    ck=torch.load(WEIGHT,map_location='cpu',weights_only=False); bb=build_backbone(ck.get('arch','coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k')); model=Raptor(bb,bb.num_features); model.load_state_dict(ck['model'],strict=True); model.eval().cuda(); Fdim=bb.num_features
    fp=OUT/'features_f16.npy'; mp=OUT/'window_mask.npy'
    feat=np.lib.format.open_memmap(fp,mode='w+',dtype=np.float16,shape=(len(ids),42,Fdim)); wmout=np.lib.format.open_memmap(mp,mode='w+',dtype=np.uint8,shape=(len(ids),42)); t0=time.time()
    with torch.no_grad():
        for i in range(len(ids)):
            xx,mm=canonicalize(np.array(sh.get(i),copy=False),masks6[i],rev[i],fh[i],fv[i]); w,wm=windows_fixed(xx,mm); xt=torch.from_numpy(w).cuda(); chunks=[]
            for k in range(0,42,14):
                z=F.interpolate(xt[k:k+14],size=(384,384),mode='bilinear',align_corners=False)
                with torch.amp.autocast('cuda',dtype=torch.float16): chunks.append(model.backbone(z).float().cpu().numpy().astype(np.float16))
            feat[i]=np.concatenate(chunks); wmout[i]=wm.astype(np.uint8)
            if (i+1)%250==0: print(f'extract {i+1}/{len(ids)} elapsed={time.time()-t0:.1f}s',flush=True)
    feat.flush(); wmout.flush(); del feat,wmout
    feat=np.load(fp,mmap_mode='r'); wm=np.load(mp)
    teacher=pd.read_csv(ROOT/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv').set_index('StudyInstanceUID'); train=pd.read_csv(ROOT/'data'/'train.csv').set_index('StudyInstanceUID')
    y=np.zeros((len(ids),12),np.float32); conf=np.zeros_like(y); gold=np.zeros(len(ids),bool); gy=np.zeros_like(y)
    for i,uid in enumerate(ids):
        r=teacher.loc[uid]; y[i]=r[TARGETS].to_numpy(np.float32); conf[i]=r[[t+'__conf' for t in TARGETS]].to_numpy(np.float32); vals=train.loc[uid,TARGETS].to_numpy(np.float32); gold[i]=np.isfinite(vals).all(); gy[i]=np.nan_to_num(vals)
    tri=np.where(~gold)[0]; vi=np.where(gold)[0]
    for p in model.backbone.parameters():p.requires_grad=False
    params=list(model.norm.parameters())+list(model.att.parameters())+[model.clsW,model.clsb]; opt=torch.optim.AdamW(params,lr=2e-4,weight_decay=1e-3)
    rng=np.random.default_rng(123); hist=[]
    def predict(indices):
        out=[]; model.eval()
        with torch.no_grad():
            for s in range(0,len(indices),16):
                ii=indices[s:s+16]; f=torch.from_numpy(np.asarray(feat[ii],dtype=np.float32)).cuda(); m=torch.from_numpy(wm[ii].astype(np.float32)).cuda()
                with torch.amp.autocast('cuda',dtype=torch.float16): out.append(torch.sigmoid(model.masked_head(f,m)).float().cpu().numpy())
        return np.concatenate(out)
    p0=predict(vi); a0,m0=aucs(gy[vi],p0); print('init macro',m0,dict(zip(TARGETS,a0)),flush=True)
    for ep in range(10):
        model.train(); order=rng.permutation(tri); losses=[]
        for s in range(0,len(order),32):
            ii=order[s:s+32]; f=torch.from_numpy(np.asarray(feat[ii],dtype=np.float32)).cuda(); m=torch.from_numpy(wm[ii].astype(np.float32)).cuda(); yy=torch.from_numpy(y[ii]).cuda(); ww=torch.from_numpy(conf[ii]).cuda(); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',dtype=torch.float16):
                logit=model.masked_head(f,m); per=F.binary_cross_entropy_with_logits(logit,yy,reduction='none'); loss=(per*(.25+.75*ww)).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.); opt.step(); losses.append(float(loss))
        pv=predict(vi); aa,mm=aucs(gy[vi],pv); row={'epoch':ep+1,'loss':float(np.mean(losses)),'macro':mm,**{t:aa[k] for k,t in enumerate(TARGETS)}}; hist.append(row); print(json.dumps(row),flush=True)
    pd.DataFrame(hist).to_csv(OUT/'history.csv',index=False); pd.DataFrame(pv,columns=TARGETS).assign(StudyInstanceUID=[ids[i] for i in vi]).to_csv(OUT/'epoch10_gold_pred.csv',index=False)
    summary={'selection':'fixed_epoch10_not_gold_selected','initial_macro':m0,'epoch10':hist[-1],'exploratory_best_macro':max(x['macro'] for x in hist),'transform_counts':{'reverse':int(rev.sum()),'flip_h':int(fh.sum()),'flip_v':int(fv.sum())},'feature_seconds':time.time()-t0}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
