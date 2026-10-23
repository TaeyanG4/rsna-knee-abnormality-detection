"""Verify four documentation PNGs against their pixel-only asset manifest."""
from pathlib import Path
import hashlib
import json
import sys
from PIL import Image

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
folder = root / 'docs/reports/visuals-20261022'
manifest = json.loads((folder / 'asset-manifest.json').read_text(encoding='utf-8'))
expected = {'study-to-prediction.png','candidate-family-map.png','public-score-snapshot.png','fullbag-local-evidence.png'}
assert len(manifest['assets']) == 4
assert {x['file'] for x in manifest['assets']} == expected
for asset in manifest['assets']:
    path = folder / asset['file']
    assert path.resolve().parent == folder.resolve()
    assert path.stat().st_size == asset['bytes'] < 20 * 1024**2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256']
    assert asset['patient_data'] is False
    with Image.open(path) as im:
        assert im.format == 'PNG' and im.mode == 'RGB'
        assert im.size == (asset['width'], asset['height'])
        assert not im.info and not im.getexif(), 'Unexpected image metadata'
    with Image.open(path) as im:
        im.verify()
print(json.dumps({'status':'PASS','assets':4,'metadata':'none','hashes':'match','scope':'file integrity; content provenance is documented in the manifest and renderer'}))
