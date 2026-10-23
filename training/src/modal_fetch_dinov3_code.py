import modal
app=modal.App('rsna-public-code-fetch')
vol=modal.Volume.from_name('rsna-public-models',create_if_missing=True)
img=modal.Image.debian_slim().pip_install('requests>=2.32')
@app.function(image=img,volumes={'/models':vol},timeout=600)
def fetch():
 import requests,hashlib,os
 commit='6876159a11b4df116f30f667f8c9888617df0751';url=f'https://codeload.github.com/facebookresearch/dinov3/zip/{commit}';dst=f'/models/dinov3-{commit}.zip'
 r=requests.get(url,timeout=(20,120),headers={'User-Agent':'rsna-public-code-audit/1.0'});r.raise_for_status();data=r.content;h=hashlib.sha256(data).hexdigest();open(dst,'wb').write(data);vol.commit();return {'bytes':len(data),'sha256':h,'file':os.path.basename(dst)}
@app.local_entrypoint()
def main():print(fetch.remote())
