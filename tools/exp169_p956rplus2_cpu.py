"""P956-R+2 CPU replay: shared decode/windows, two strict loads, failure injection."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import time
import zipfile

try:
    from tools import exp169_release_p956rplus2 as release
    from tools import exp169_p956rplus_cpu as original
    from tools import exp152_own0949_release as helper
except ImportError:
    import exp169_release_p956rplus2 as release
    import exp169_p956rplus_cpu as original
    import exp152_own0949_release as helper

SCRATCH = release.ROOT / '.local/scratch/sol-4/p956rplus2'


def replay(output: Path):
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    weights = SCRATCH / 'fixture-weights'
    weights.mkdir(parents=True, exist_ok=True)
    seed44 = release.R44_SOURCE if release.R44_SOURCE.is_file() else release.R_SOURCE
    checkpoint_hashes = {}
    for name, source in ((release.R_NAME, release.R_SOURCE), (release.R44_NAME, seed44)):
        shutil.copyfile(source, weights / name)
        checkpoint_hashes[name] = dict(sha256=release.sha(weights / name))
    (weights / 'manifest.json').write_bytes(release.json_bytes(dict(checkpoints=checkpoint_hashes)))
    with (original.COMP / 'test.csv').open(encoding='utf-8-sig', newline='') as stream:
        ids = [row[release.UID] for row in csv.DictReader(stream)][:2]
    release.need(len(set(ids)) == 2, 'two studies required')
    with zipfile.ZipFile(original.DATA_ZIP) as archive:
        for name in archive.namelist():
            if name.startswith('test_series/') and name.endswith('.dcm') and any('/'+uid+'/' in name for uid in ids):
                release.need('..' not in Path(name).parts, 'unsafe DICOM path')
                archive.extract(name, SCRATCH / 'dicom')
    frame = pd.read_csv(original.COMP / 'test_series.csv',
        usecols=[release.UID, 'SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive'],
        dtype={release.UID: str, 'SeriesInstanceUID': str})
    records = {uid: group.to_dict('records') for uid, group in frame.groupby(release.UID, sort=False)}
    ns = dict(__name__='p956rplus2_cpu_serving', __file__=str(SCRATCH / 'public-serving.py'))
    source = (release.VISIBLE / 'exp155-assets/public-serving.py').read_text(encoding='utf-8')
    exec(compile(release.r_runtime_source(source), ns['__file__'], 'exec'), ns)

    @torch.inference_mode()
    def cpu_probs(model, windows, device):
        release.need(device == 'cpu', 'CPU only')
        logits = model(windows.unsqueeze(0)).float()
        release.need(bool(torch.isfinite(logits).all()), 'nonfinite CPU logits')
        return torch.sigmoid(logits)[0].numpy()

    ns['infer_probs'] = cpu_probs
    reader = original.cpu_reader(ns['_load_fastread']()).make_builder(
        ns['FASTREAD_VARIANT'], 384, ns['SLOTS'], hdr_threads=4, px_threads=4)
    release.bind_r(ns, helper, weights, original.COMP, output)
    binding = ns['_E152']
    models = [binding.checked_model(ns, weights / name, helper.ARCH, 384, 'cpu')
              for name in (release.R_NAME, release.R44_NAME)]
    release.need(all(res == 320 for _, res in models) and ns['K_EVAL'] == 94, 'serving drift')
    print('Both checkpoints strict-loaded on CPU (486 state tensors each)', flush=True)
    values, individual, fallback, inputs = [], [], [], {}
    inference_seconds = []
    decode_calls = 0
    for index, uid in enumerate(ids):
        vol, mask = reader(uid, records, str(SCRATCH / 'dicom/test_series'))
        decode_calls += 1
        release.need(bool(mask.any()), 'empty decoded study')
        plain = ns['gpu_windows'](vol, mask, 94, 320, 'cpu')
        mirror = ns['gpu_windows'](vol, mask, 94, 320, 'cpu', mirror=True)
        inputs[uid] = dict(volume_sha256=helper.array_sha(vol), mask_sha256=helper.array_sha(mask),
                           plain_sha256=helper.array_sha(plain), mirror_sha256=helper.array_sha(mirror))
        observed = []
        def infer(model, p, m, device):
            tick = time.monotonic()
            result = ns['infer_pair_probs'](model, p, m, device)
            inference_seconds.append(time.monotonic() - tick)
            observed.append(result.copy())
            return result
        combined = binding.infer_members(uid, models, plain, mirror, 'cpu', infer)
        np.testing.assert_array_equal(combined, .5 * (observed[0] + observed[1]))
        values.append(combined)
        individual.append(observed)
        # Execute actual successful inference in the survivor path, in both directions.
        for failed in (0, 1):
            def fail_one(model, p, m, device):
                if model is models[failed][0]:
                    raise RuntimeError('injected checkpoint failure')
                return ns['infer_pair_probs'](model, p, m, device)
            survived = binding.infer_members(uid + f'-fail{failed}', models, plain, mirror, 'cpu', fail_one)
            np.testing.assert_array_equal(survived, observed[1-failed])
            fallback.append(survived)
        def fail_both(*args):
            raise RuntimeError('injected both checkpoint failure')
        try:
            binding.infer_members(uid + '-both', models, plain, mirror, 'cpu', fail_both)
        except ValueError as exc:
            release.need('both R2 checkpoints failed' in str(exc), 'unexpected failure')
        else:
            raise AssertionError('both-failure must reach inherited R sentinel handling')
        del plain, mirror, vol, mask
        print(f'study {index+1}/2: decoded once, mean + both survivor directions finite', flush=True)
    r2 = np.asarray(values)
    prior = original.parent.read_json(original.parent.REPLAY / 'p956-receipt.json')
    s6_path = original.parent.REPLAY / 'p956-run/s6/predictions.csv'
    release.check_pin(s6_path, prior['provenance']['s6_raw_sha256'])
    all_ids, all_s6 = release.v5q._read_csv(s6_path)
    ours = all_s6[[all_ids.index(uid) for uid in ids]].copy()
    after = release.blend(ids, r2, ours)
    np.testing.assert_array_equal(after, original.parent.public_formula(r2, ours))
    for name, arr in [('r2.csv', r2), ('seed43.csv', np.asarray(individual)[:,0]),
                      ('seed44.csv', np.asarray(individual)[:,1]), ('s6.csv', ours), ('submission.csv', after)]:
        (output / name).write_bytes(release.csv_bytes(ids, arr))
    (output / 'inputs.json').write_bytes(release.json_bytes(inputs))
    report = dict(status='PASS_CPU_REPLAY', version=release.VERSION, studies=2, decode_calls=decode_calls,
        strict_load=True, state_tensors=[len(m.state_dict()) for m, _ in models],
        checkpoint_sha256=checkpoint_hashes, seed44_is_seed43_fixture=seed44 == release.R_SOURCE,
        native_resolution=320, k_eval=94, tta=['plain','anatomical mirror'], device='cpu', precision='float32',
        one_failure_exact=True, both_fail_raise=True, finite=bool(np.isfinite(r2).all()),
        s6_exact=True, public_formula_exact=True, member_failures=binding.member_failures,
        submission_sha256=release.sha(output / 'submission.csv'),
        pair_seconds=inference_seconds, seconds=round(time.monotonic()-started,3),
        scope='Two real DICOM studies; actual CPU forwards for mean and survivor paths. Authenticated unchanged S6 replay. No CUDA qualification.',
        files={p.name: release.sha(p) for p in output.iterdir() if p.is_file() and p.name != 'replay.json'})
    (output / 'replay.json').write_bytes(release.json_bytes(report))
    return report
