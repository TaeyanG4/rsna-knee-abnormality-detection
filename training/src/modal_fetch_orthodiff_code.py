import modal
app=modal.App('rsna-orthodiff-code-fetch')
vol=modal.Volume.from_name('rsna-public-models',create_if_missing=True)
img=modal.Image.debian_slim().pip_install('requests>=2.32')
COMMIT='99d15186bb8face1728263023d8414d9405c8f7e'
@app.function(image=img,volumes={'/models':vol},timeout=600)
def fetch():
 import requests,hashlib,os
 url=f'https://codeload.github.com/lt-0123/OrthoDiffusion/zip/{COMMIT}'
 dst=f'/models/orthodiffusion-{COMMIT}.zip'
 r=requests.get(url,timeout=(20,120),headers={'User-Agent':'rsna-public-code-audit/1.0'});r.raise_for_status();data=r.content
 h=hashlib.sha256(data).hexdigest();open(dst,'wb').write(data);vol.commit()
 return {'file':os.path.basename(dst),'bytes':len(data),'sha256':h}
@app.local_entrypoint()
def main(): print(fetch.remote())
