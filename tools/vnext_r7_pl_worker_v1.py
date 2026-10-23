"""vNext-R7-PL 1.0.0 dedicated child adapters; never edits frozen workers."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from tools import vnext_r7_pl_v1 as p


def checkpoint_metadata(ck, entry, receipt):
    expected = dict(arch=entry['arch'], lab=p.TARGETS, seed=entry['seed'])
    if entry['kind'] == 'CoAt':
        expected.update(res=384, inference_res=320, contract='exp143_raptor96_384_c140_s02',
            k_eval=94, norm='imagenet', window_policy='legacy', oai=False,
            checkpoint_kind='fixed-swa-bn', swa_epochs_1based=[14, 11, 12])
    p.need(all(ck.get(k) == v for k, v in expected.items()), 'PL checkpoint recipe')
    expected_pl = {k: receipt[k] for k in ('arm', 'teacher_sha256', 'target_sha256',
        'train_ids_sha256', 'l2_pos_weight_sha256', 'spec_sha256', 'registration_sha256',
        'clean_oof', 'exposure')}
    if receipt['arm'] == 'fill':
        expected_pl['silence_mask_sha256'] = receipt['silence_mask_sha256']
    p.need(ck.get('pl_provenance') == expected_pl, 'PL checkpoint provenance')


@contextmanager
def native_adapters(job, h):
    """Child-local function substitutions, restored even if a worker fails."""
    journals = [p.Journal(s['path'], s['ids'], s['identity']) for s in job['branches']]
    routes = {member: journal for journal in journals for member in journal.members}
    original = (h.c.Journal, h.dump, h.q_model, h.anchor_setup)
    class Dispatch:
        def __init__(self, *_):
            pass
        def record(self, uid, member, values=None, error=None, seconds=None):
            return routes[member].record(uid, member, values, error, seconds)
    def q_model(entry, device):
        import torch
        receipt = p.read(p.pin(entry['training_receipt']))
        checkpoint_metadata(torch.load(p.pin(entry), map_location='cpu', weights_only=False), entry, receipt)
        return original[2](entry, device)
    def anchor_setup(task, entries, device, failed=None):
        ns, models, reader = original[3](task, [e for e in entries if e['kind'] != 'CoAt'], device, failed)
        for entry in entries:
            if entry['kind'] != 'CoAt':
                continue
            try:
                import torch
                receipt = p.read(p.pin(entry['training_receipt']))
                ck = torch.load(p.pin(entry), map_location='cpu', weights_only=False)
                checkpoint_metadata(ck, entry, receipt)
                backbone = ns['build_backbone'](p.COAT_ARCH, pretrained=False)
                model = ns['RaptorClassifier'](backbone, F_dim=backbone.num_features)
                model.load_state_dict(ck['model'], strict=True)
                models[entry['id']] = model.eval().to(device)
            except Exception as exc:
                if failed is None:
                    raise
                failed[entry['id']] = f'{type(exc).__name__}: {exc}'
        return ns, models, reader
    h.c.Journal, h.dump, h.q_model, h.anchor_setup = Dispatch, p.dump, q_model, anchor_setup
    try:
        yield journals
    finally:
        h.c.Journal, h.dump, h.q_model, h.anchor_setup = original


def native(job, worker=None):
    from tools import hb116_vnext_release as h
    if worker is None:
        from tools.hb116_native_runtime import worker
    task = dict(job, identity={'dispatch': True}, entries=[dict(e, resolved=e['path']) for e in job['entries']])
    for e in task['entries']:
        if e['kind'] == 'MIL':
            e['spec_sha256'] = p.read(p.pin(e['training_receipt']))['spec_sha256']
    path = Path(job['output'])/'native-job.json'
    p.dump(path, task)
    with native_adapters(task, h) as journals:
        worker(path, job['device'])
        for journal in journals:
            journal.finish()


def r7(job, infer=None):
    from tools import exp169_release_c85r7codec_deadline_v122 as insurance
    from tools import hb127_vnextr7_release as base
    import numpy as np
    public = Path(job['output'])/'public'
    p.dump(public/'ids.json', job['ids'])
    # Preflight is pinned independently of the child's eventual provenance.
    _, identity = insurance.r7.verify_bundle(Path(job['bundle']), base.MANIFEST_SHA, Path(job['source']))
    p.dump(public/'pr7-run/r7-identity.json', identity)
    task = dict(job, output=str(public/'pr7-run/s_r7'), device='cpu' if job['device'] == 'cpu' else 'cuda')
    path = Path(job['output'])/'r7-job.json'
    p.dump(path, task)
    journal = p.Journal(**dict(directory=job['branches'][0]['path'], ids=job['ids'],
                               identity=job['branches'][0]['identity']))
    insurance.install_adapters(Path(job['output'])/'insurance-logs')
    with p.metadata.writers(insurance):
        try:
            (infer or base.r7_worker)(path)
        except Exception:
            # Only authenticated completed worker rows may survive operational failure.
            pass
        _, collector = p.metadata.collectors(insurance, job['ids'])
        ids, _, values, info = collector(public)
    p.need(ids == job['ids'], 'R7 cohort/order')
    for uid, value in zip(ids, np.asarray(values, np.float32)):
        journal.record(uid, 'S_R7', value, error=info['failures']['S_R7'].get(uid))
    journal.finish()


def codec(job):
    from tools import exp169_release_c85r7codec_deadline_v122 as insurance
    with p.metadata.writers(insurance):
        receipt = insurance.old.install_codecs(Path(job['runtime_root'])/'tools', Path(job['output']))
    p.dump(Path(job['output'])/'codec.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--job', type=Path)
    group.add_argument('--codec-job', type=Path)
    args = parser.parse_args()
    if args.codec_job:
        codec(p.read(args.codec_job))
    else:
        job = p.read(args.job)
        (native if job['family'] == 'NATIVE' else r7)(job)


if __name__ == '__main__':
    main()
