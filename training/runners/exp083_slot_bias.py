"""Explicit EXP083 GPU pilot/full runner. Importing never starts a phase."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def rng_fingerprint(torch):
    h = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return h.hexdigest()


def run(args):
    started = time.monotonic()
    sys.path.insert(0, str(ROOT))
    from tools import exp083_slot_bias as g
    from tools import exp082_r34_loss as old
    m = g.validate_registration(args.protocol, ROOT)
    seconds = g.validate_phase_authority(args.phase, args.protocol, m, args.pilot_receipt, args.root_decision)
    deadline = started + seconds
    old.install_gold_blocker(ROOT)
    out = Path(args.out)
    if out.exists():
        raise ValueError('Refusing to overwrite a phase directory')
    out.mkdir(parents=True)
    receipt = dict(schema='exp083-phase-v1', phase=args.phase, status='STARTED',
                   protocol_sha256=g.digest(args.protocol), budget_seconds=seconds,
                   started_unix=time.time(), gold_access='disabled', no_F2_score_evaluated=True)
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Dataset
        sys.path.insert(0, str(ROOT / 'training/src'))
        from raptor_common import Corpus, build_model, environment, seed_all, write_predictions
        from raptor_data import StudyWindows, collate
        from raptor_runtime import configure_runtime
        runtime = configure_runtime('fp16', 'cuda', policy='strict')
        env = environment()
        if runtime != m['expected_provenance']['eval_runtime']:
            raise ValueError('Runtime differs from saved B8')
        if any(env.get(k) != v for k, v in m['environment'].items()):
            raise ValueError('Environment differs from registration')
        seed_all(42)
        protocol_dir = Path(args.protocol).parent
        ordered = g.read(protocol_dir / 'ordered-ids.json')
        all_ids = ordered['train'] + ordered['eval']
        if (g.ids_hash(ordered['train']) != m['ordered_ids']['train_sha256']
                or g.ids_hash(ordered['eval']) != m['ordered_ids']['eval_sha256']
                or g.ids_hash(all_ids) != m['ordered_ids']['combined_sha256']):
            raise ValueError('Ordered UID mismatch')
        selected_ids = m['phases']['pilot_ids'] if args.phase == 'pilot' else all_ids
        indices = {u: i for i, u in enumerate(all_ids)}
        selected_indices = [indices[u] for u in selected_ids]
        centers = np.load(protocol_dir / 'centers.npy', allow_pickle=False)
        slots = np.load(protocol_dir / 'slots.npy', allow_pickle=False)
        corpus = Corpus(ROOT / m['corpus'])
        for u, i in zip(selected_ids, selected_indices):
            mask = corpus.mask(u)
            if g.centers_from_mask(mask) != centers[i].tolist() or g.slots_for_centers(centers[i], mask) != slots[i].tolist():
                raise ValueError('Registered actual centers/slots changed')
        checkpoint = torch.load(ROOT / m['checkpoint'], map_location='cpu', weights_only=False)
        model = build_model(m['architecture'], pretrained=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        model.eval().requires_grad_(False).to('cuda')
        encoder_before = g.state_hash({k: v for k, v in model.state_dict().items() if k.startswith('backbone.')})
        if encoder_before != m['encoder_state_sha256']:
            raise ValueError('Encoder differs from saved B8')
        initial_head = torch.load(protocol_dir / 'head-initial.pt', map_location='cpu', weights_only=True)
        if g.state_hash(initial_head) != m['saved_head_sha256']:
            raise ValueError('Saved head state changed')
        control_initial = g.make_head(initial_head, False).eval().to('cuda')
        treatment_initial = g.make_head(initial_head, True).eval().to('cuda')
        if {p.data_ptr() for p in control_initial.parameters()} & {p.data_ptr() for p in treatment_initial.parameters()}:
            raise ValueError('Initial arm parameters alias storage')
        dummy = {u: np.zeros(12, dtype=np.float32) for u in selected_ids}
        shape = (len(selected_ids), 24, m['cache']['shape'][2])
        features = np.lib.format.open_memmap(out / 'features.npy', mode='w+', dtype=np.float16, shape=shape)
        offset = 0; extract_started = time.monotonic()
        extraction_splits = [('train', ordered['train'][:8] if args.phase == 'pilot' else ordered['train']),
                             ('eval', ordered['eval'][:8] if args.phase == 'pilot' else ordered['eval'])]
        frozen_eval_probabilities, frozen_eval_ids, extraction_batches = [], [], []
        with torch.no_grad():
            # Separate loaders preserve original eval batch4 boundaries after odd train3475.
            for split_name, split_ids in extraction_splits:
                dataset = StudyWindows(ROOT / m['corpus'], split_ids, dummy, 224, 24,
                                       False, 'imagenet', 42, window_policy='within-slot-valid')
                loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2,
                                    collate_fn=collate, persistent_workers=False)
                for x, _, batch_ids in loader:
                    g.check_deadline(deadline)
                    if batch_ids != selected_ids[offset:offset + len(batch_ids)]:
                        raise ValueError('Feature row order mismatch')
                    batch_slots = torch.as_tensor(slots[[indices[u] for u in batch_ids]].astype(np.int64), device='cuda')
                    with torch.autocast('cuda', dtype=torch.float16):
                        encoded = model.encode(x.to('cuda'))
                        value = g.native_feature_array(encoded, (len(batch_ids), 24, shape[2]))
                        original_logits = model.head(encoded)
                        control_logits = control_initial(encoded, batch_slots)
                        treatment_logits = treatment_initial(encoded, batch_slots)
                    if (not torch.equal(original_logits, control_logits)
                            or not torch.equal(control_logits, treatment_logits)
                            or not bool(torch.isfinite(original_logits).all())):
                        raise ValueError('Zero-bias head logits are not exactly equal to saved B8')
                    # Preserve blank encodings too; reload written FP16 values before accepting cache.
                    features[offset:offset + len(batch_ids)] = value
                    reloaded = torch.from_numpy(np.array(features[offset:offset + len(batch_ids)], copy=True)).to('cuda')
                    if not torch.equal(encoded, reloaded):
                        raise ValueError('Native feature cache roundtrip changed values')
                    with torch.autocast('cuda', dtype=torch.float16):
                        reloaded_logits = model.head(reloaded)
                    if not torch.equal(original_logits, reloaded_logits):
                        raise ValueError('Native versus cache original-head logits changed')
                    if offset == 0:
                        receipt['head_dtypes'] = dict(control_initial.last_dtypes)
                        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                            model.att.train(); control_initial.train(); treatment_initial.train()
                            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                            with torch.autocast('cuda', dtype=torch.float16):
                                train_original = model.head(encoded)
                                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
                                train_control = control_initial(encoded, batch_slots)
                                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
                                train_treatment = treatment_initial(encoded, batch_slots)
                            if not torch.equal(train_original, train_control) or not torch.equal(train_control, train_treatment):
                                raise ValueError('Initial train-mode logits differ under replayed dropout RNG')
                            model.att.eval(); control_initial.eval(); treatment_initial.eval()
                        receipt['initial_train_logits_exact'] = True
                    if split_name == 'eval':
                        frozen_eval_probabilities.append(torch.sigmoid(reloaded_logits.float()).cpu().numpy())
                        frozen_eval_ids.extend(batch_ids)
                    extraction_batches.append(dict(split=split_name, size=len(batch_ids), uids_sha256=g.ids_hash(batch_ids)))
                    offset += len(batch_ids)
        if offset != len(selected_ids):
            raise ValueError('Incomplete feature cache')
        features.flush(); del features
        extraction_seconds = time.monotonic() - extract_started
        encoder_after = g.state_hash({k: v for k, v in model.state_dict().items() if k.startswith('backbone.')})
        if encoder_after != encoder_before:
            raise ValueError('Encoder changed during extraction')
        feature_sha = g.digest(out / 'features.npy')
        if frozen_eval_ids != extraction_splits[1][1]:
            raise ValueError('Frozen B8 common-cache eval row order changed')
        np.save(out / 'frozen-b8-eval-predictions.npy', np.concatenate(frozen_eval_probabilities), allow_pickle=False)
        g.write_new(out / 'extraction-batches.json', extraction_batches)
        receipt.update(feature_shape=list(shape), feature_dtype='float16', features_sha256=feature_sha,
                       selected_ids_sha256=g.ids_hash(selected_ids), initial_logits_exact=True,
                       native_cache_roundtrip_exact=True, native_cache_logits_exact=True,
                       disjoint_initial_head_storage=True, extraction_split_loaders=True,
                       frozen_b8_common_cache_predictions_sha256=g.digest(out / 'frozen-b8-eval-predictions.npy'),
                       extraction_batches_sha256=g.digest(out / 'extraction-batches.json'),
                       encoder_before_sha256=encoder_before, encoder_after_sha256=encoder_after,
                       extraction_seconds=extraction_seconds, runtime=runtime, environment=env,
                       blank_features='native normalized-image encoding preserved; no zero replacement/removal')
        del model, control_initial, treatment_initial, dataset, loader
        torch.cuda.empty_cache()
        g.check_deadline(deadline)
        if args.phase == 'pilot':
            receipt.update(pilot_ids=selected_ids, scope='Native dtype/initial parity/16-row operational pilot; not full-runtime/memory/performance proof')
        else:
            # Read training supervision only after feature cache and its parity are fixed.
            training_labels = old.read_training_labels(ROOT / m['labels'], ordered['train'], m['targets'])
            weights, _ = old.class_weights(training_labels, ordered['train'])
            if not np.array_equal(weights, np.asarray(m['head']['pos_weight'], dtype=np.float32)):
                raise ValueError('Common clipped loss vector changed')
            cache = np.load(out / 'features.npy', mmap_mode='r', allow_pickle=False)
            if cache.dtype != np.float16 or list(cache.shape) != m['cache']['shape']:
                raise ValueError('Unexpected shared cache contract')

            class Features(Dataset):
                def __init__(self, ids, supervised):
                    self.ids, self.supervised = ids, supervised
                def __len__(self):
                    return len(self.ids)
                def __getitem__(self, i):
                    uid = self.ids[i]; row = indices[uid]
                    y = training_labels[uid] if self.supervised else np.zeros(12, dtype=np.float32)
                    return (torch.from_numpy(np.array(cache[row], copy=True)),
                            torch.from_numpy(slots[row].astype(np.int64)),
                            torch.from_numpy(y), uid)

            arm_receipts = {}
            for arm, treatment in [('control', False), ('treatment', True)]:
                g.check_deadline(deadline)
                if g.digest(out / 'features.npy') != feature_sha:
                    raise ValueError('Shared feature cache changed between arms')
                head = g.make_head(initial_head, treatment).to('cuda')
                if g.shared_head_hash(head) != m['saved_head_sha256'] or bool(head.slot_bias.any()):
                    raise ValueError('Arm did not start from identical saved head/zero bias')
                shuffle_generator = torch.Generator().manual_seed(42)
                tl = DataLoader(Features(ordered['train'], True), batch_size=64, shuffle=True,
                                generator=shuffle_generator, drop_last=False, num_workers=0, pin_memory=True)
                params = [p for p in head.parameters() if p.requires_grad]
                if len(tl) != 55 or len(ordered['train']) % 64 != 19:
                    raise ValueError('Expected55batches/last19')
                opt = torch.optim.AdamW(params, lr=.001, weight_decay=.02, betas=(.9, .999), eps=1e-8, amsgrad=False)
                sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=.001, total_steps=1100, pct_start=.15,
                    anneal_strategy='cos', cycle_momentum=True, base_momentum=.85, max_momentum=.95,
                    div_factor=25., final_div_factor=10000., three_phase=False)
                scaler = torch.amp.GradScaler('cuda', init_scale=1024)
                if set(id(p) for group in opt.param_groups for p in group['params']) != set(map(id, params)):
                    raise ValueError('Optimizer does not own exactly this arm trainable parameters')
                lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(weights, device='cuda'))
                # Construction/imports/previous arm evaluation cannot affect either fit RNG.
                seed_all(42)
                trace, history, scale_history = [], [], []
                updates = skips = consecutive = 0
                arm_started = time.monotonic()
                for ep in range(20):
                    head.train(); total_loss = 0.
                    for batch_index, (x, slot_ids, y, batch_ids) in enumerate(tl):
                        g.check_deadline(deadline)
                        x, slot_ids, y = x.to('cuda'), slot_ids.to('cuda'), y.to('cuda')
                        opt.zero_grad(set_to_none=True)
                        before = rng_fingerprint(torch)
                        with torch.autocast('cuda', dtype=torch.float16):
                            logits = head(x, slot_ids)
                            loss = lossf(logits.float(), y)
                        after = rng_fingerprint(torch)
                        trace.append(dict(epoch=ep, batch=batch_index, uids_sha256=g.ids_hash(batch_ids), before=before, after=after))
                        if treatment:
                            i = len(trace) - 1
                            if trace[i] != control_trace[i]:
                                raise ValueError('Paired minibatch/dropout RNG trajectory differs')
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('Nonfinite head loss')
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 3.)
                        old_scale = scaler.get_scale()
                        scaler.step(opt); scaler.update()
                        scale_history.append(dict(attempt=len(trace), before=old_scale, after=scaler.get_scale()))
                        if scaler.get_scale() >= old_scale:
                            sched.step(); updates += 1; consecutive = 0
                        else:
                            skips += 1; consecutive += 1
                            if consecutive >= 10:
                                raise RuntimeError('Ten consecutive optimizer skips')
                        total_loss += float(loss.detach())
                    history.append(dict(epoch=ep, loss=total_loss / len(tl), attempts=len(trace), updates=updates, skips=skips,
                                        F2_metric=None, gold_metric=None))
                    print(json.dumps(dict(phase='head', arm=arm, epoch=ep + 1, updates=updates, skips=skips)), flush=True)
                if updates + skips != 1100 or skips / (updates + skips) > .01:
                    raise ValueError('Head update/skip contract failed')
                if not treatment:
                    control_trace = trace
                elif len(trace) != len(control_trace):
                    raise ValueError('RNG trace length mismatch')
                final_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                torch.save(dict(model=final_state, arm=arm, epoch=19, protocol_sha256=g.digest(args.protocol)), out / (arm + '-final.pt'))
                head.eval(); probabilities = []; returned_ids = []
                vl = DataLoader(Features(ordered['eval'], False), batch_size=4, shuffle=False, num_workers=0)
                with torch.no_grad():
                    for x, slot_ids, _, batch_ids in vl:
                        g.check_deadline(deadline)
                        with torch.autocast('cuda', dtype=torch.float16):
                            logits = head(x.to('cuda'), slot_ids.to('cuda'))
                        if not bool(torch.isfinite(logits).all()):
                            raise ValueError('Nonfinite final prediction')
                        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
                        returned_ids += batch_ids
                if returned_ids != ordered['eval']:
                    raise ValueError('Final OOF UID order changed')
                values = np.concatenate(probabilities)
                np.save(out / (arm + '-predictions.npy'), values, allow_pickle=False)
                write_predictions(out / (arm + '-predictions.csv'), returned_ids, values)
                g.write_new(out / (arm + '-history.json'), history)
                g.write_new(out / (arm + '-rng-trace.json'), trace)
                g.write_new(out / (arm + '-scale-history.json'), scale_history)
                arm_receipts[arm] = dict(attempts=updates + skips, updates=updates, skips=skips,
                    last_batch_size=len(ordered['train']) % 64, any_skips_require_review=bool(skips),
                    separate_optimizer_scheduler_scaler=True, elapsed_seconds=time.monotonic() - arm_started,
                    initial_shared_head_sha256=m['saved_head_sha256'], initial_slot_bias='allzero',
                    feature_sha256=feature_sha, final_state_sha256=g.state_hash(final_state),
                    predictions_sha256=g.digest(out / (arm + '-predictions.npy')),
                    rng_trace_sha256=g.digest(out / (arm + '-rng-trace.json')),
                    scale_history_sha256=g.digest(out / (arm + '-scale-history.json')),
                    history_sha256=g.digest(out / (arm + '-history.json')))
                del head, tl, vl, opt, sched, scaler, final_state
                torch.cuda.empty_cache()
            if g.digest(out / 'features.npy') != feature_sha:
                raise ValueError('Final shared cache hash changed')
            receipt.update(arms=arm_receipts, paired_rng_trace_exact=True,
                           requires_skip_review=any(v['skips'] for v in arm_receipts.values()),
                           equal_realized_update_budget=all(v['skips'] == 0 for v in arm_receipts.values()),
                           scope='Two fixed head fits and predictions; no F2 score or adoption decision')
        g.check_deadline(deadline)
        receipt.update(status='PASS', elapsed_seconds=time.monotonic() - started)
        g.write_new(out / 'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status='FAILED', elapsed_seconds=time.monotonic() - started,
                       exception_type=type(exc).__name__, error=str(exc))
        g.write_new(out / 'failure.json', receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--phase', choices=['pilot', 'full'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--pilot-receipt')
    parser.add_argument('--root-decision')
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in ('phase', 'status', 'elapsed_seconds', 'no_F2_score_evaluated')}, indent=2))


if __name__ == '__main__':
    main()
