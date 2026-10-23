"""Explicit EXP085 GPU worker. Import is passive; main is the only execution path."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_hash(torch):
    digest = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes())
    digest.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    sys.path.insert(0, str(ROOT))
    from tools import exp085_relative_position as g
    from tools import exp084_group_loss as old
    p, ordered = g.validate_protocol(args.protocol, ROOT, verify_inputs=True)
    phase_name = 'EXP085_RELATIVE_POSITION_G1_PILOT' if args.phase == 'pilot' else 'EXP085_RELATIVE_POSITION_G1'
    if Path(args.out).resolve() != (ROOT / 'artifacts/experiments' / phase_name).resolve():
        raise ValueError('Registered phase output path required')
    allowed_seconds = p['execution']['pilot_seconds']
    if args.phase == 'full':
        if not args.pilot_dir:
            raise ValueError('Full worker requires completed paired pilot')
        prior = g.read(Path(args.pilot_dir) / 'receipt.json')
        prior_resource = g.read(Path(args.pilot_dir).with_name(Path(args.pilot_dir).name + '_RESOURCE') / 'receipt.json')
        if (prior['protocol_sha256'] != g.sha(args.protocol) or prior_resource['status'] != 'COMPLETE'
                or prior_resource['child_exit_code'] != 0 or prior_resource['cleanup']['errors'] or prior_resource['cleanup']['survivors']):
            raise ValueError('Pilot protocol/resource completion mismatch')
        prior['supervised_elapsed_seconds'] = prior_resource['elapsed_seconds']
        g.pilot_cost_gate(prior, p)
        allowed_seconds = p['execution']['total_time_limit_seconds'] - prior_resource['elapsed_seconds']
    if not 0 < args.time_limit <= allowed_seconds:
        raise ValueError('Requested phase time exceeds registered remaining budget')
    old.install_gold_blocker(ROOT)
    sys.path.insert(0, str(ROOT / 'training/src'))
    from raptor_common import Corpus, LAB, build_model, environment, seed_all, write_predictions
    from raptor_runtime import configure_runtime
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.time_limit
    receipt = {'schema': g.SCHEMA, 'phase': args.phase, 'status': 'FAILED', 'protocol_sha256': g.sha(args.protocol),
               'no_metrics_evaluated': True, 'gold_access': 'disabled', 'retries': 0, 'arms': {}}
    try:
        if not torch.cuda.is_available() or not 0 < args.time_limit <= 7200:
            raise ValueError('GPU worker requires explicit CUDA and registered bounded timeout')
        old.worker_init(0)
        a = p['training']
        runtime = configure_runtime('fp16', 'cuda', policy='strict')
        env = environment()
        if runtime != p['expected_runtime']['runtime'] or runtime != p['expected_runtime']['eval_runtime']:
            raise ValueError('Strict runtime differs from registered local baseline')
        if any(env.get(key) != p['environment'].get(key) for key in old.ENV_KEYS):
            raise ValueError('Training environment differs from registered EXP084')
        if LAB != p['targets'] or a['res'] != 224 or a['k'] != 12 or a['k_eval'] != 24 or a['epochs'] != 8 or a['bs'] != 8 or a['accum'] != 1:
            raise ValueError('Fixed input/training contract changed')
        corpus = Corpus(g.contained(ROOT, p['paths']['corpus']))
        rows = old.read_rows(g.contained(ROOT, p['groups']['path']))
        train_ids, val_ids = old.validate_membership(corpus.ids.astype(str).tolist(), ordered, rows)
        labels = g.read_training_labels(g.contained(ROOT, p['paths']['labels']), train_ids, LAB)
        weights = old.class_weights(labels, train_ids)
        if not np.array_equal(weights, np.asarray(p['loss']['baseline_pos_weight'], dtype=np.float32)):
            raise ValueError('T-only clipped loss changed')
        inference_ids = train_ids[:p['execution']['pilot_eval_train_studies']] if args.phase == 'pilot' else val_ids
        # Full-V labels are never parsed by the runner; evaluation is external.
        dataset_labels = {**labels, **{uid: np.zeros(12, np.float32) for uid in val_ids}}
        checkpoint = torch.load(g.contained(ROOT, p['initial_checkpoint']['path']), map_location='cpu', weights_only=False)
        if any(key in checkpoint for key in ('optimizer', 'scheduler', 'scaler')):
            raise ValueError('Unexpected resumable state: registered experiment is weights-only adaptation')
        if (checkpoint['epoch'] != 7 or checkpoint['tag'] != 'EXP084_B_G1'
                or checkpoint['arch'] != a['arch'] or checkpoint['lab'] != LAB
                or checkpoint['provenance']['train_ids_ordered_sha256'] != p['ordered_ids']['train_sha256']
                or checkpoint['provenance']['eval_ids_ordered_sha256'] != p['ordered_ids']['val_sha256']
                or checkpoint['provenance']['protocol_sha256'] != p['parent_protocol']['sha256']):
            raise ValueError('Checkpoint architecture/epoch/ancestor contract differs')
        base_initial_hash = g.state_hash(checkpoint['model'])
        control_trace = None
        common_initial_hash = None
        full_attempts = (len(train_ids) // a['bs']) * p['execution']['full_epochs']
        if full_attempts != p['execution']['attempts_per_fit']:
            raise ValueError('Fixed realized full update budget changed')
        projected = {}
        for arm in g.ARMS:
            g.check_deadline(deadline)
            arm_start = time.monotonic()
            fit_deadline = min(deadline, arm_start + p['execution']['per_fit_seconds'])
            arm_out = out / arm
            arm_out.mkdir(exist_ok=False)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            seed_all(a['seed'])
            base = build_model(a['arch'], pretrained=False)
            base.load_state_dict(checkpoint['model'], strict=True)
            if g.state_hash(base.state_dict()) != base_initial_hash:
                raise ValueError('Base state not restored exactly')
            model = g.make_model(base, arm == 'treatment')
            initial_hash = g.state_hash(model.state_dict())
            if args.phase == 'full' and initial_hash != prior['arms'][arm]['initial_state_sha256']:
                raise ValueError('Full arm initialization differs from its paired pilot')
            if common_initial_hash is None:
                common_initial_hash = initial_hash
            elif initial_hash != common_initial_hash:
                raise ValueError('Paired expanded initial states differ')
            added_params = sum(t.numel() for n, t in model.named_parameters() if not n.startswith('base.'))
            if added_params != p['architecture']['extra_parameters']:
                raise ValueError('Unexpected architecture parameter count')
            model = model.to('cuda')
            if a['grad_ckpt']:
                model.base.backbone.set_grad_checkpointing(True)
            train_data = g.PositionWindows(corpus.dir, train_ids, dataset_labels, a['res'], a['k'], True, a['norm'], a['seed'], window_policy=a['window_policy'])
            generator = torch.Generator().manual_seed(a['seed'])
            loader = DataLoader(train_data, batch_size=a['bs'], shuffle=True, generator=generator,
                                num_workers=a['workers'], drop_last=True, pin_memory=True,
                                persistent_workers=a['workers'] > 0, collate_fn=g.collate, worker_init_fn=old.worker_init)
            eval_data = g.PositionWindows(corpus.dir, inference_ids, dataset_labels, a['res'], a['k_eval'], False, a['norm'], a['seed'], window_policy=a['window_policy'])
            eval_loader = DataLoader(eval_data, batch_size=4, shuffle=False, num_workers=min(a['workers'], 2),
                                     collate_fn=g.collate, worker_init_fn=old.worker_init)
            groups = g.optimizer_groups(model, a['bb_lr'], a['head_lr'])
            opt = torch.optim.AdamW(groups, weight_decay=a['wd'])
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a['bb_lr'], a['head_lr']],
                                                       total_steps=full_attempts, pct_start=.15)
            scaler = torch.amp.GradScaler('cuda', init_scale=a['amp_init_scale'])
            lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, device='cuda'))
            if set(id(t) for group in opt.param_groups for t in group['params']) != set(id(t) for t in model.parameters()):
                raise ValueError('Optimizer does not own every trainable parameter exactly')
            provenance = {'schema': g.SCHEMA, 'arm': arm, 'protocol_sha256': g.sha(args.protocol),
                          'initial_state_sha256': initial_hash, 'base_initial_state_sha256': base_initial_hash,
                          'initial_checkpoint_sha256': p['initial_checkpoint']['sha256'],
                          'train_ids_ordered_sha256': p['ordered_ids']['train_sha256'],
                          'eval_ids_ordered_sha256': p['ordered_ids']['val_sha256'],
                          'gold_access': 'disabled', 'metric_access': 'deferred_until_both_final',
                          'optimizer_state': 'fresh AdamW/OneCycleLR/GradScaler for each arm', 'runtime': runtime,
                          'extra_parameters': added_params, 'all_backbone_parameters_trainable': True,
                          'input_policy': 'Original StudyWindows output unchanged; deterministic center metadata replay',
                          'epoch_offset': p['execution']['epoch_offset'], 'environment': env}
            g.write_new(arm_out / 'provenance.json', provenance)
            seed_all(a['seed'])
            trace, history, updates = [], [], 0
            train_start = time.monotonic()
            warmup_seconds = None
            epochs = 1 if args.phase == 'pilot' else p['execution']['full_epochs']
            initial_exact = False
            for epoch in range(epochs):
                train_data.set_epoch(epoch + p['execution']['epoch_offset'])
                model.train()
                total_loss = 0.
                epoch_start = time.monotonic()
                for index, (pixels, slots, positions, targets, uids, centers) in enumerate(loader):
                    g.check_deadline(fit_deadline)
                    x, s, pos, y = (t.to('cuda', non_blocking=True) for t in (pixels, slots, positions, targets))
                    if updates == 0:
                        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                            initial_forward_gate = g.verify_initial_forward(model, x, s, pos)
                        initial_exact = True
                        model.train()
                    step = {'epoch': epoch, 'sampling_epoch': epoch + p['execution']['epoch_offset'], 'batch': index,
                            'uids_sha256': g.ids_hash(uids), 'centers_sha256': tensor_hash(centers),
                            'positions_sha256': tensor_hash(positions), 'rng_before': rng_hash(torch)}
                    if index == 0:
                        step['pixels_sha256'] = tensor_hash(pixels)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16):
                        loss = lossf(model(x, s, pos).float(), y)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Nonfinite training loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
                    scale_before = scaler.get_scale()
                    scaler.step(opt); scaler.update()
                    if scaler.get_scale() < scale_before:
                        raise RuntimeError('Optimizer overflow/skip; stop first failure without retry')
                    sched.step(); updates += 1
                    step['rng_after'] = rng_hash(torch)
                    trace.append(step)
                    if arm == 'treatment' and step != control_trace[len(trace)-1]:
                        raise ValueError('Paired pixel/metadata/minibatch/dropout RNG trajectory differs')
                    total_loss += float(loss.detach())
                    if args.phase == 'pilot' and updates == p['execution']['pilot_warmup_steps']:
                        torch.cuda.synchronize()
                        warmup_seconds = time.monotonic() - train_start
                    if args.phase == 'pilot' and updates == p['execution']['pilot_steps_per_arm']:
                        break
                history.append({'epoch': epoch, 'sampling_epoch': epoch + p['execution']['epoch_offset'],
                                'training_loss': total_loss / (index+1), 'updates': updates, 'skips': 0,
                                'seconds': time.monotonic()-epoch_start, 'validation_metric': None, 'gold_metric': None})
                print(json.dumps({'arm': arm, 'phase': args.phase, **history[-1]}), flush=True)
            torch.cuda.synchronize()
            train_seconds = time.monotonic() - train_start
            expected_updates = p['execution']['pilot_steps_per_arm'] if args.phase == 'pilot' else full_attempts
            if updates != expected_updates or not initial_exact:
                raise ValueError('Unexpected updates or unchecked original-B initialization')
            if arm == 'control':
                control_trace = trace
            elif len(trace) != len(control_trace):
                raise ValueError('Paired trace lengths differ')
            # Fixed-final inference only; pilot uses T16 and saves no predictions/checkpoint.
            eval_start = time.monotonic()
            first_eval_seconds = None
            first_eval_rows = None
            model.eval(); probabilities, returned_ids = [], []
            with torch.no_grad():
                for pixels, slots, positions, _, uids, _ in eval_loader:
                    g.check_deadline(fit_deadline)
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = model(pixels.to('cuda'), slots.to('cuda'), positions.to('cuda'))
                    values = torch.sigmoid(logits.float())
                    if tuple(values.shape) != (len(uids), 12) or not bool(torch.isfinite(values).all()):
                        raise ValueError('Invalid fixed-final inference')
                    if args.phase == 'full':
                        probabilities.append(values.cpu().numpy())
                    returned_ids += uids
                    if first_eval_seconds is None:
                        torch.cuda.synchronize()
                        first_eval_seconds = time.monotonic() - eval_start
                        first_eval_rows = len(uids)
            torch.cuda.synchronize()
            eval_seconds = time.monotonic() - eval_start
            if returned_ids != inference_ids:
                raise ValueError('Final inference UID order differs')
            if args.phase == 'full':
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save({'model': state, 'epoch': 7, 'adaptation_epochs': 8, 'arm': arm, 'schema': g.SCHEMA,
                            'arch': a['arch'], 'res': 224, 'lab': LAB, 'provenance': provenance}, arm_out / 'model_final.pt')
                write_predictions(arm_out / 'oof_final.csv', returned_ids, np.concatenate(probabilities))
                del state
            g.write_new(arm_out / 'history.json', history)
            g.write_new(arm_out / 'trace.json', trace)
            arm_receipt = {'arm': arm, 'phase': args.phase, 'status': 'PASS', 'protocol_sha256': g.sha(args.protocol),
                           'initial_state_sha256': initial_hash, 'original_B_forward_exact': initial_exact,
                           'initial_forward_gate': initial_forward_gate,
                           'last_attention_dtypes': model.last_attention_dtypes,
                           'train': len(train_ids), 'val': len(val_ids), 'inference_rows': len(returned_ids),
                           'updates': updates, 'skips': 0, 'selected_epoch_1based': epochs,
                           'training_seconds': train_seconds, 'inference_seconds': eval_seconds,
                           'pilot_train_warmup_seconds': warmup_seconds,
                           'first_eval_batch_seconds': first_eval_seconds, 'first_eval_batch_rows': first_eval_rows,
                           'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                           'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
                           'elapsed_seconds': time.monotonic() - arm_start,
                           'trace_sha256': g.sha(arm_out / 'trace.json'), 'weak_oof_macro_auc': None,
                           'gold_access': 'disabled', 'metric_access': 'deferred_until_both_final',
                           'checkpoint_saved': args.phase == 'full', 'predictions_saved': args.phase == 'full'}
            if args.phase == 'full':
                arm_receipt['checkpoint_sha256'] = g.sha(arm_out / 'model_final.pt')
                arm_receipt['predictions_sha256'] = g.sha(arm_out / 'oof_final.csv')
            else:
                setup = max(0., arm_receipt['elapsed_seconds'] - train_seconds - eval_seconds)
                train_rate = (train_seconds - warmup_seconds) / (updates - p['execution']['pilot_warmup_steps'])
                eval_rate = (eval_seconds - first_eval_seconds) / (len(inference_ids) - first_eval_rows)
                if train_rate <= 0 or eval_rate <= 0:
                    raise ValueError('Invalid pilot steady timing')
                estimate = (setup + warmup_seconds + train_rate * (full_attempts - p['execution']['pilot_warmup_steps'])
                            + first_eval_seconds + eval_rate * (len(val_ids) - first_eval_rows))
                projected[arm] = p['execution']['projection_safety_factor'] * estimate
                arm_receipt['pilot_steady_train_seconds_per_update'] = train_rate
                arm_receipt['pilot_steady_eval_seconds_per_study'] = eval_rate
            g.check_deadline(fit_deadline)
            g.write_new(arm_out / 'summary.json', arm_receipt)
            receipt['arms'][arm] = arm_receipt
            del model, base, opt, sched, scaler, lossf, loader, eval_loader, train_data, eval_data, groups
            del loss, logits, values, x, s, pos, y
            gc.collect(); torch.cuda.empty_cache()
        receipt.update(status='PASS', paired_initialization_exact=True, paired_trace_exact=True,
                       conservative_full_seconds_by_arm=projected, elapsed_seconds=time.monotonic()-started,
                       pilot_state_discarded=args.phase == 'pilot', no_validation_target_values_converted_or_scored=True,
                       validation_csv_rows_scanned_for_uid_only=True)
        g.check_deadline(deadline)
        g.write_new(out / 'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status='FAILED', elapsed_seconds=time.monotonic()-started,
                       exception_type=type(exc).__name__, error=str(exc))
        g.write_new(out / 'failure.json', receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--phase', choices=['pilot', 'full'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--time-limit', required=True, type=float)
    parser.add_argument('--pilot-dir')
    result = run(parser.parse_args())
    print(json.dumps({'status': result['status'], 'phase': result['phase']}))


if __name__ == '__main__':
    main()
