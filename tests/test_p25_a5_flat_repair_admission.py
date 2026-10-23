"""v1.0.0: exact repaired notebook/config acceptance with synthetic receipts only."""
import json
from test_k1_admit_v1 import case, change, transport
from test_k1_admit_a5_v1 import a5_case, refresh
from test_p25_a5_flat_stage_v1 import flat_case
from tools import p25_a5_flat_stage_v1 as b
from tools import k1_admit_v1 as admit


def test_exact_repaired_artifacts_admitted_offline(flat_case, capsys):
    outputs, reg = b.artifacts()
    stage = b.stage()
    for name, data in outputs.items():
        assert (stage / name).read_bytes() == data
    flat_case.update(stage=stage, kernel=b.SLUG)
    for key in ('visible_dir', 'n32_dir'):
        d = flat_case[key]
        diagnostic = key == 'n32_dir'
        cfg = b.config(b.package()[3], diagnostic)
        change(d/'k1sr-receipt.json', lambda r:r.update(cfg))
        change(d/'c85r7-receipt.json', lambda r:r.update(version=cfg['release_version']))
        if diagnostic:
            change(d/'k1sr-receipt.json', lambda r:r.update(per_study=[
                dict(uid=str(i),ok=True,config_sha256=cfg['config_sha256']) for i in range(32)]))
        refresh(d)
        if not diagnostic:
            transport(d,kernel=b.SLUG)
    out = admit.prepare(**flat_case)
    assert 'ADMITTED_OFFLINE' in capsys.readouterr().out
    result = json.loads((out/'ordinary-visible-registration.json').read_bytes())
    assert result['tool_version'] == '1.0.6'
    assert result['kernel_id'] == b.SLUG
    assert result['version'] == b.RELEASE
    assert result['stage_registration_sha256'] == b.util.sha(stage/'registration.json')
    assert (stage/'kernel/release.ipynb').read_bytes() == outputs['kernel/release.ipynb']
