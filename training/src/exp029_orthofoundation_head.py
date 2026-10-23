try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import argparse,json,math,time
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F
from sklearn.metrics import roc_auc_score
ROOT=workspace_root()
FEAT=ROOT/'experiments'/'EXP029_ORTHOFOUNDATION_CANONICAL_54_FEATURES'; CACHE=ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
class Head(nn.Module):
    def __init__(self,indim=1024,d=256,drop=.15):
        super().__init__();self.norm=nn.LayerNorm(indim);self.proj=nn.Sequential(nn.Linear(indim,d),nn.GELU(),nn.LayerNorm(d));self.slot=nn.Parameter(torch.randn(6,d)*.02);self.pos=nn.Parameter(torch.randn(9,d)*.02);self.q=nn.Parameter(torch.randn(12,d)*.02);self.w=nn.Parameter(torch.randn(12,d)*.02);self.b=nn.Parameter(torch.zeros(12));self.drop=nn.Dropout(drop)
    def forward(self,x,m):
        z=self.proj(self.norm(x))+self.slot[None,:,None,:]+self.pos[None,None,:,:]
        tok=z.reshape(z.shape[0],54,-1);mm=m.reshape(m.shape[0],54)
        sc=torch.einsum('bnd,td->btn',tok,self.q)/math.sqrt(tok.shape[-1]);sc=sc.masked_fill(mm[:,None,:]<=0,-1e4);a=torch.softmax(sc,-1);pool=torch.einsum('btn,bnd->btd',a,tok);return (self.drop(pool)*self.w[None]).sum(-1)+self.b

def aucs(y,p):
    a=[float(roc_auc_score(y[:,k],p[:,k])) for k in range(12)];return a,float(np.mean(a))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--seed',type=int,default=2601);ap.add_argument('--lr',type=float,default=5e-4);ap.add_argument('--rank',type=float,default=.0);args=ap.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    x=np.load(FEAT/'features_f16.npy',mmap_mode='r');m=np.load(FEAT/'valid.npy').astype(np.float32);uids=pd.read_csv(CACHE/'train_cache_uids.csv').StudyInstanceUID.astype(str).tolist();teacher=pd.read_csv(ROOT/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv').set_index('StudyInstanceUID');trn=pd.read_csv(ROOT/'data'/'train.csv').set_index('StudyInstanceUID')
    y=np.zeros((len(uids),12),np.float32);conf=np.zeros_like(y);gold=np.zeros(len(uids),bool);gy=np.zeros_like(y)
    for i,u in enumerate(uids):
        r=teacher.loc[u];y[i]=r[TARGETS].to_numpy(np.float32);conf[i]=r[[t+'__conf' for t in TARGETS]].to_numpy(np.float32);v=trn.loc[u,TARGETS].to_numpy(np.float32);gold[i]=np.isfinite(v).all();gy[i]=np.nan_to_num(v)
    ti=np.where(~gold)[0];vi=np.where(gold)[0];dev='cuda';model=Head().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=5e-4);rng=np.random.default_rng(args.seed);hist=[];out=ROOT/'experiments'/f'EXP029_ORTHOFOUNDATION_HEAD_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    def pred(ii):
        model.eval();oo=[]
        with torch.no_grad():
            for a in range(0,len(ii),64):
                jj=ii[a:a+64];xx=torch.from_numpy(np.asarray(x[jj],dtype=np.float32)).to(dev);mm=torch.from_numpy(m[jj]).to(dev);oo.append(torch.sigmoid(model(xx,mm)).cpu().numpy())
        return np.concatenate(oo)
    for ep in range(args.epochs):
        model.train();order=rng.permutation(ti);ls=[]
        for a in range(0,len(order),64):
            jj=order[a:a+64];xx=torch.from_numpy(np.asarray(x[jj],dtype=np.float32)).to(dev);mm=torch.from_numpy(m[jj]).to(dev);yy=torch.from_numpy(y[jj]).to(dev);ww=torch.from_numpy(conf[jj]).to(dev);opt.zero_grad(set_to_none=True);logit=model(xx,mm);loss=(F.binary_cross_entropy_with_logits(logit,yy,reduction='none')*(.25+.75*ww)).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();ls.append(float(loss.detach()))
        pp=pred(vi);aa,macro=aucs(gy[vi],pp);row={'epoch':ep+1,'loss':float(np.mean(ls)),'gold_macro_auc':macro,**{TARGETS[k]:aa[k] for k in range(12)}};hist.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame(hist).to_csv(out/'history.csv',index=False);last=hist[-1];(out/'summary.json').write_text(json.dumps({'selection':'fixed_final_epoch_not_gold_selected','final':last,'best_exploratory':max(hist,key=lambda r:r['gold_macro_auc'])},indent=2),encoding='utf-8');pd.DataFrame(pred(vi),columns=TARGETS).assign(StudyInstanceUID=[uids[i] for i in vi]).to_csv(out/'gold_pred_final.csv',index=False)
if __name__=='__main__':main()
