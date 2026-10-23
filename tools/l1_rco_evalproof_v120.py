"""Real two-study CPU evaluation equivalence for K0 v1.2.0. No labels."""
import ast
import importlib.util
import os
import sys
import time
from pathlib import Path


def proof(cache):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from tools import l1_rco_stage0_v120 as s
    from tools import l1_rco_stage0_v1 as old
    torch.set_num_threads(2)
    sys.path.insert(0, str(s.ROOT / s.CODE))
    spec = importlib.util.spec_from_file_location('l1_evalproof_trainer', s.WORK / 'compact_train_l1.py')
    ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
    reg = s.validate(s.read(s.WORK / 'registration.json'))
    weight = old.WORK / 'k0/models/E_prior_s0_f0.pt'
    net = ct.Net('convnext_small', str(s.ROOT / s.INIT), crop=352, in_chans=3, spatial=True, extent=False)
    net.load_deploy_state_dict(torch.load(weight, map_location='cpu', weights_only=True))
    net.eval()
    ids = reg['holdout'][:2]
    ds = ct.Studies(str(cache), ids, 64, 'eval', 9000, eval_stride=3, slices=3, phys_mm=3.5)
    rows = []
    start = time.perf_counter()
    for i in range(2):
        img, meta, mask, _ = ct.collate([ds[i]])
        img = img[..., 16:368, 16:368]
        with torch.no_grad():
            before = net(img, meta, mask, chunk=0)
            legacy = net(img, meta, mask, chunk=512)
            after = net(img, meta, mask, chunk=32)
        delta = float((before - after).abs().max())
        s.need(bool(torch.isfinite(after).all()), 'nonfinite eval proof')
        s.need(torch.equal(before, after) and torch.equal(legacy, after), 'CPU eval not bit-identical: ' + str(delta))
        rows.append({'windows': int(mask.sum()), 'equal': True, 'max_abs': delta})
        print('CPU evaluation study %d/2 PASS (%d windows)' % (i + 1, int(mask.sum())), flush=True)
    s.need(not torch.cuda.is_initialized(), 'CUDA initialized')
    result = {'status': 'PASS', 'version': s.VERSION, 'contract_id': reg['contract_id'],
              'studies': rows, 'weight_sha256': s.sha(weight), 'trainer_sha256': s.sha(s.WORK / 'compact_train_l1.py'),
              'cuda_initialized': False, 'seconds': time.perf_counter() - start,
              'scope': 'FP32 CPU complete studies, chunk=0/512/32; GPU bfloat16 and peak UNMEASURED'}
    s.write(s.WORK / 'eval-proof.json', result)
    return result


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.l1_rco_stage0_v120 import CACHE
    proof(CACHE)
