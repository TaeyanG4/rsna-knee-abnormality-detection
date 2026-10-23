import modal
app=modal.App('rsna-public-model-fetch')
vol=modal.Volume.from_name('rsna-public-models',create_if_missing=True)
img=modal.Image.debian_slim().pip_install('requests>=2.32')
@app.function(image=img,volumes={'/models':vol},timeout=1800)
def fetch_ortho():
    import requests,hashlib,time,os
    url='https://media.githubusercontent.com/media/ytrsk/OrthoFoundation/main/OrthoFoudation-L.pth'
    dst='/models/OrthoFoudation-L.pth'; exp='385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9'; size=1213056638
    if os.path.exists(dst) and os.path.getsize(dst)==size:
        h=hashlib.sha256()
        with open(dst,'rb') as f:
            for b in iter(lambda:f.read(8<<20),b''):h.update(b)
        if h.hexdigest()==exp:return {'status':'already_verified','bytes':size,'sha256':exp}
    h=hashlib.sha256();n=0;t=time.time();tmp=dst+'.part'
    with requests.get(url,stream=True,timeout=(20,120),headers={'User-Agent':'rsna-public-model-audit/1.0'}) as r:
        r.raise_for_status()
        with open(tmp,'wb') as f:
            for b in r.iter_content(8<<20):
                if b:f.write(b);h.update(b);n+=len(b)
    if n!=size or h.hexdigest()!=exp:raise RuntimeError((n,h.hexdigest()))
    os.replace(tmp,dst);vol.commit();return {'status':'downloaded_verified','bytes':n,'sha256':h.hexdigest(),'seconds':round(time.time()-t,2)}
@app.local_entrypoint()
def main():print(fetch_ortho.remote())
