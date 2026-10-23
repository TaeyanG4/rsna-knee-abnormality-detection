from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd,pydicom

ROOT=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'); OUT=Path('/kaggle/working')
SLOTS=[('SAG_FLUID','Sagittal',1),('SAG_STRUCT','Sagittal',0),('COR_FLUID','Coronal',1),('COR_STRUCT','Coronal',0),('AX_FLUID','Axial',1),('AX_STRUCT','Axial',0)]
TAGS=['ImagePositionPatient','ImageOrientationPatient','PixelSpacing','Rows','Columns','Laterality','FrameLaterality']

def count(folder):
    try:return sum(1 for _ in folder.glob('*.dcm'))
    except:return 0

def pick(rows,plane,fluid,study_dir):
    p=[r for r in rows if str(r.get('Anatomical_Plane'))==plane and int(r.get('Fluid_Sensitive',r.get('Fat_Suppression',0)) or 0)==fluid]
    if not p:return None
    return sorted([(count(study_dir/str(r['SeriesInstanceUID'])),str(r['SeriesInstanceUID']),r) for r in p],key=lambda z:(-z[0],z[1]))[0][2]

def sample_headers(folder,maxn=5):
    fs=list(folder.glob('*.dcm'))
    if not fs:return []
    idx=np.linspace(0,len(fs)-1,min(maxn,len(fs))).round().astype(int)
    out=[]
    for j in idx:
        try:
            d=pydicom.dcmread(str(fs[int(j)]),stop_before_pixels=True,specific_tags=TAGS)
            iop=np.asarray(getattr(d,'ImageOrientationPatient',[]),float);ipp=np.asarray(getattr(d,'ImagePositionPatient',[]),float);ps=np.asarray(getattr(d,'PixelSpacing',[]),float)
            if iop.shape!=(6,) or ipp.shape!=(3,) or ps.shape!=(2,):continue
            row=iop[:3];col=iop[3:]; nr=np.linalg.norm(row);nc=np.linalg.norm(col)
            if nr<=0 or nc<=0:continue
            row=row/nr;col=col/nc; normal=np.cross(row,col);nn=np.linalg.norm(normal)
            if nn<=0:continue
            normal=normal/nn; rows=float(getattr(d,'Rows',0) or 0);cols=float(getattr(d,'Columns',0) or 0)
            center=ipp+row*float(ps[1])*(max(cols-1,0)/2.0)+col*float(ps[0])*(max(rows-1,0)/2.0)
            tag=str(getattr(d,'Laterality','') or getattr(d,'FrameLaterality','') or '').strip().upper()
            out.append((center,row,col,normal,tag))
        except Exception:pass
    return out

def medvec(xs,k):return np.median(np.stack([x[k] for x in xs]),axis=0) if xs else np.full(3,np.nan)

def main():
    t=time.time(); tr=pd.read_csv(ROOT/'train.csv',dtype={'StudyInstanceUID':str}); se=pd.read_csv(ROOT/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str}); groups={u:g.to_dict('records') for u,g in se.groupby('StudyInstanceUID')}; rows=[]
    for i,u in enumerate(tr.StudyInstanceUID.astype(str)):
        sd=ROOT/'train_series'/u; sr=groups.get(u,[]); per=[]
        for si,(name,plane,fluid) in enumerate(SLOTS):
            r=pick(sr,plane,fluid,sd)
            if r is None:
                per.append({'slot':name,'missing':1});continue
            su=str(r['SeriesInstanceUID']); hs=sample_headers(sd/su)
            if not hs:
                per.append({'slot':name,'series_uid':su,'missing':0,'header_ok':0});continue
            cen=medvec(hs,0);rv=medvec(hs,1);cv=medvec(hs,2);nv=medvec(hs,3);rv=rv/max(np.linalg.norm(rv),1e-9);cv=cv/max(np.linalg.norm(cv),1e-9);nv=nv/max(np.linalg.norm(nv),1e-9); tags=[x[4] for x in hs if x[4] in ('L','R')]; tag=max(set(tags),key=tags.count) if tags else ''
            per.append({'slot':name,'series_uid':su,'missing':0,'header_ok':1,'center_x':float(cen[0]),'center_y':float(cen[1]),'center_z':float(cen[2]),'row_x':float(rv[0]),'row_y':float(rv[1]),'row_z':float(rv[2]),'col_x':float(cv[0]),'col_y':float(cv[1]),'col_z':float(cv[2]),'normal_x':float(nv[0]),'normal_y':float(nv[1]),'normal_z':float(nv[2]),'tag':tag})
        xs=[p['center_x'] for p in per if p.get('header_ok') and np.isfinite(p['center_x'])]
        sx=float(np.median(xs)) if xs else np.nan; lat='L' if sx>0 else ('R' if sx<0 else '')
        tags=[p.get('tag','') for p in per if p.get('tag') in ('L','R')]; tag=max(set(tags),key=tags.count) if tags else ''
        for p in per:
            base={'index':i,'StudyInstanceUID':u,'study_center_x':sx,'laterality_geom':lat,'laterality_tag':tag,'tag_agrees_geom':int(bool(tag and lat and tag==lat)) if tag and lat else np.nan};base.update(p)
            if p.get('header_ok') and lat:
                medial=np.array([-1.,0,0]) if lat=='L' else np.array([1.,0,0])
                desired_slice={'SAG_FLUID':medial,'SAG_STRUCT':medial,'COR_FLUID':np.array([0.,1.,0.]),'COR_STRUCT':np.array([0.,1.,0.]),'AX_FLUID':np.array([0.,0.,1.]),'AX_STRUCT':np.array([0.,0.,1.])}[p['slot']]
                n=np.array([p['normal_x'],p['normal_y'],p['normal_z']]);base['reverse_slice_to_canonical']=int(np.dot(n,desired_slice)<0)
                if p['slot'].startswith('SAG'): h=np.array([0.,1.,0.]);v=np.array([0.,0.,-1.])
                elif p['slot'].startswith('COR'): h=medial;v=np.array([0.,0.,-1.])
                else: h=medial;v=np.array([0.,1.,0.])
                rvec=np.array([p['row_x'],p['row_y'],p['row_z']]);cvec=np.array([p['col_x'],p['col_y'],p['col_z']]);base['row_dot_desired_h']=float(np.dot(rvec,h));base['row_dot_desired_v']=float(np.dot(rvec,v));base['col_dot_desired_h']=float(np.dot(cvec,h));base['col_dot_desired_v']=float(np.dot(cvec,v))
            rows.append(base)
        if (i+1)%200==0:print(i+1,'/',len(tr),'elapsed',round((time.time()-t)/60,1),'m',flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'orientation_metadata.csv',index=False)
    study=df.drop_duplicates('StudyInstanceUID'); known=study.dropna(subset=['tag_agrees_geom']); summary={'studies':int(study.shape[0]),'rows':int(df.shape[0]),'geom_L':int((study.laterality_geom=='L').sum()),'geom_R':int((study.laterality_geom=='R').sum()),'tag_known':int(len(known)),'tag_agreement':float(known.tag_agrees_geom.mean()) if len(known) else None,'header_ok_rows':int((df.header_ok.fillna(0)==1).sum()),'elapsed_seconds':time.time()-t}
    (OUT/'orientation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
