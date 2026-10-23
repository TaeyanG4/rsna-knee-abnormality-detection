"""EXP114 (H-B-39): resumable training checkpoints for the Colab T4 runner, stored as directories of fixed-size parts.

Layout: <ckpt_dir>/<tag>/epoch_NNN/ (NNN = completed epochs, 001..020) holding part_000, part_001, ... and manifest.json (schema, the run
identity, sha256 of every part and of the whole payload). A checkpoint is written to epoch_NNN.tmp-<pid>/ and renamed when complete, so a
listed epoch_NNN directory is complete unless it was damaged later (every read re-checks the hashes). <ckpt_dir>/<tag>/latest.json points at
the newest one. Parts exist because the Colab CLI moves a whole file per request through the Jupyter contents API (base64 JSON, no
chunking; colab_cli/contents.py), so a ~290 MB ResNet50 + AdamW checkpoint is moved as ~32 MiB pieces that can be retried one by one.

The payload (torch.save) holds model, optimizer, OneCycleLR and GradScaler state, completed epochs, the global step (optimizer updates,
skipped FP16 steps included as in the runner's counter), the FP16 skip count, the epoch history, the train.log text so far, the sessions of
the trajectory, the RNG states (python, numpy, torch CPU, every CUDA device, the DataLoader shuffle generator) and, after the final epoch,
the model_best payload. The pure-file helpers (parts, manifests, listing, pruning, split/join of single files) need no torch, so the local
poller (tools/colab/sync_exp114.py) imports this module as well; the torch helpers import torch only when called.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import time
from pathlib import Path

SCHEMA = 'exp114-ckpt-v1'
FILE_SCHEMA = 'exp114-file-parts-v1'
PART_BYTES = 32 * 2 ** 20
EPOCH_RE = re.compile(r'^epoch_(\d{3})$')
PAYLOAD_KEYS = ('model', 'optimizer', 'scheduler', 'scaler', 'completed_epochs', 'global_step', 'optimizer_updates', 'skipped', 'history',
                'rng', 'best', 'train_log', 'sessions', 'identity')
RNG_KEYS = ('python', 'numpy', 'torch_cpu', 'torch_cuda', 'loader_generator')


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def epoch_dir_name(completed_epochs):
    return f'epoch_{int(completed_epochs):03d}'


def write_json_atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ------------------------------------------------------------------ parts (no torch)
def _write_part_dir(chunks, dest, header, part_bytes):
    """Write an iterable of byte chunks (each <= part_bytes) as part_NNN files plus manifest.json into dest (atomic rename)."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    tmp = dest.with_name(dest.name + f'.tmp-{os.getpid()}')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    whole, parts, total = hashlib.sha256(), [], 0
    for i, chunk in enumerate(chunks):
        name = f'part_{i:03d}'
        (tmp / name).write_bytes(chunk)
        whole.update(chunk); total += len(chunk)
        parts.append(dict(name=name, bytes=len(chunk), sha256=sha256_bytes(chunk)))
    manifest = dict(header, total_bytes=total, sha256=whole.hexdigest(), part_bytes=int(part_bytes), parts=parts,
                    written_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (tmp / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, dest)
    return manifest


def write_parts(data, dest, header, part_bytes=PART_BYTES):
    if part_bytes < 1:
        raise ValueError('part_bytes must be positive')
    chunks = [data[i:i + part_bytes] for i in range(0, len(data), part_bytes)] or [b'']
    return _write_part_dir(chunks, dest, header, part_bytes)


def split_file(src, dest, header=None, part_bytes=PART_BYTES):
    """A single file as a part directory (for moving model_best.pt through the Colab CLI)."""
    src = Path(src)

    def chunks():
        with src.open('rb') as f:
            while True:
                chunk = f.read(part_bytes)
                if not chunk:
                    break
                yield chunk
    return _write_part_dir(chunks(), dest, dict(header or {}, schema=FILE_SCHEMA, file=src.name), part_bytes)


def read_manifest(d):
    return read_json(Path(d) / 'manifest.json')


def verify_parts(d, manifest=None):
    """Problems of one part directory (empty list = every part present with its sha256 and the whole payload hash matches)."""
    d = Path(d)
    try:
        m = manifest or read_manifest(d)
    except (OSError, ValueError) as e:
        return [f'{d.name}: manifest unreadable ({type(e).__name__})']
    problems, whole, total = [], hashlib.sha256(), 0
    for part in m.get('parts') or []:
        p = d / part['name']
        if not p.is_file():
            problems.append(f"{d.name}: {part['name']} missing"); continue
        data = p.read_bytes()
        if len(data) != part['bytes'] or sha256_bytes(data) != part['sha256']:
            problems.append(f"{d.name}: {part['name']} differs from the manifest"); continue
        whole.update(data); total += len(data)
    if not problems and (total != m.get('total_bytes') or whole.hexdigest() != m.get('sha256')):
        problems.append(f'{d.name}: payload size/hash differs from the manifest')
    if not m.get('parts'):
        problems.append(f'{d.name}: no parts listed')
    return problems


def read_parts(d):
    problems = verify_parts(d)
    if problems:
        raise ValueError('; '.join(problems))
    m = read_manifest(d)
    return b''.join((Path(d) / part['name']).read_bytes() for part in m['parts'])


def join_parts(d, dest):
    """Reassemble a part directory into dest (atomic), checking every hash."""
    data = read_parts(d)
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f'.tmp-{os.getpid()}')
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return sha256_bytes(data)


def complete_epochs(tag_dir):
    """[(completed_epochs, path)] of the epoch directories that carry a manifest, oldest first (hashes are not checked here)."""
    tag_dir = Path(tag_dir)
    if not tag_dir.is_dir():
        return []
    found = []
    for d in tag_dir.iterdir():
        m = EPOCH_RE.match(d.name)
        if m and d.is_dir() and (d / 'manifest.json').is_file():
            found.append((int(m.group(1)), d))
    return sorted(found)


def prune(tag_dir, keep):
    """Keep the newest `keep` complete epoch directories (keep <= 0 keeps all); always remove stale *.tmp-* leftovers of dead writers."""
    tag_dir = Path(tag_dir)
    removed = []
    if not tag_dir.is_dir():
        return removed
    for d in tag_dir.iterdir():
        if '.tmp-' in d.name and d.is_dir():
            pid = d.name.rsplit('.tmp-', 1)[1]
            if pid != str(os.getpid()):
                shutil.rmtree(d, ignore_errors=True); removed.append(d.name)
    if keep > 0:
        for _, d in complete_epochs(tag_dir)[:-keep]:
            shutil.rmtree(d, ignore_errors=True); removed.append(d.name)
    return removed


# ------------------------------------------------------------------ RNG and training state (torch imported lazily)
def capture_rng(shuffle_generator, cuda):
    import random
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if cuda else None,
                loader_generator=shuffle_generator.get_state())


def restore_rng(state, shuffle_generator, cuda):
    import random
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if cuda:
        if state.get('torch_cuda') is None:
            raise ValueError('checkpoint has no CUDA RNG state (written by a CPU run); refusing to resume on CUDA')
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    shuffle_generator.set_state(state['loader_generator'])


def training_payload(model, opt, sched, scaler, completed_epochs, optimizer_updates, skipped, history, best, shuffle_generator, cuda, log_path,
                     sessions, identity):
    """Everything an uninterrupted run carries from the end of one epoch into the next (tensors moved to CPU)."""
    log_path = Path(log_path)
    return dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, optimizer=opt.state_dict(),
                scheduler=sched.state_dict(), scaler=scaler.state_dict(), completed_epochs=int(completed_epochs),
                global_step=int(optimizer_updates), optimizer_updates=int(optimizer_updates), skipped=int(skipped),
                history=list(history), rng=capture_rng(shuffle_generator, cuda), best=best,
                train_log=log_path.read_text(encoding='utf-8') if log_path.is_file() else '', sessions=list(sessions), identity=dict(identity))


def save_training_checkpoint(ckpt_dir, tag, completed_epochs, payload, identity, keep=3, part_bytes=PART_BYTES):
    """torch.save(payload) as <ckpt_dir>/<tag>/epoch_NNN/ parts + manifest; update latest.json; prune to `keep`. Returns (dir, manifest)."""
    import torch
    buf = io.BytesIO()
    torch.save(payload, buf)
    tag_dir = Path(ckpt_dir) / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    dest = tag_dir / epoch_dir_name(completed_epochs)
    if dest.exists():  # only an older, superseded copy of the same epoch (e.g. one that failed verification) can be here
        shutil.rmtree(dest)
    history = payload.get('history') or []
    header = dict(schema=SCHEMA, identity=dict(identity), tag=tag, completed_epochs=int(completed_epochs), global_step=payload['global_step'],
                  optimizer_updates=payload['optimizer_updates'], skipped=payload['skipped'], has_best=payload.get('best') is not None,
                  epoch_loss=history[-1].get('loss') if history else None, sessions=len(payload.get('sessions') or []))
    manifest = write_parts(buf.getvalue(), dest, header, part_bytes)
    write_json_atomic(tag_dir / 'latest.json', dict(dir=dest.name, completed_epochs=int(completed_epochs), sha256=manifest['sha256'],
                                                    total_bytes=manifest['total_bytes'], written_at=manifest['written_at']))
    prune(tag_dir, keep)
    return dest, manifest


def find_resume(ckpt_dir, tag, identity, log=print):
    """Newest intact checkpoint of this run as dict(path, manifest, payload), or None. A damaged directory is skipped (older one used); a
    checkpoint of a different run identity (code, protocol, labels, ids, schedule, ...) stops the run instead of being mixed in."""
    import torch
    tag_dir = Path(ckpt_dir) / tag
    for n, d in reversed(complete_epochs(tag_dir)):
        problems = verify_parts(d)
        if problems:
            log(f'checkpoint {d} skipped: {problems[:3]}')
            continue
        manifest = read_manifest(d)
        got = manifest.get('identity') or {}
        diff = {k: [got.get(k), v] for k, v in identity.items() if got.get(k) != v}
        if manifest.get('schema') != SCHEMA or diff:
            raise SystemExit(f'checkpoint {d} belongs to a different run identity ({manifest.get("schema")}; {json.dumps(diff)[:1500]}); '
                             'move the checkpoint directory aside to start this fold again')
        payload = torch.load(io.BytesIO(read_parts(d)), map_location='cpu', weights_only=False)
        missing = [k for k in PAYLOAD_KEYS if k not in payload]
        if missing or payload['completed_epochs'] != n:
            raise SystemExit(f'checkpoint {d} payload incomplete: missing {missing}, epochs {payload.get("completed_epochs")} vs {n}')
        return dict(path=str(d), manifest=manifest, payload=payload)
    return None


def restore_training_state(state, model, opt, sched, scaler, loader, shuffle_generator, cuda, seed):
    """Load a checkpoint payload into a freshly built run and put every RNG stream where the uninterrupted run had it after that epoch.

    The DataLoader detail: an uninterrupted run with persistent workers draws the iterator's base seed from loader.generator once, before
    epoch 0, and afterwards only the sampler draws (two permutations per epoch) from the same generator object; a new process creates a
    new persistent iterator at its first epoch, so loader.generator is pointed at a separate generator for that one base-seed draw (worker
    seeds do not matter: window sampling and jitter are sha256(seed, epoch, uid)-deterministic). Without persistent workers every epoch
    creates a new iterator in both runs, so the shared generator is kept. Returns (history, optimizer_updates, skipped, start_epoch, best).
    """
    import torch
    model.load_state_dict(state['model'])
    opt.load_state_dict(state['optimizer'])
    sched.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    if loader.persistent_workers:
        loader.generator = torch.Generator().manual_seed(int(seed))
    restore_rng(state['rng'], shuffle_generator, cuda)
    return list(state['history']), int(state['optimizer_updates']), int(state['skipped']), int(state['completed_epochs']), state.get('best')


def session_record(start_epoch, environment, resumed_from=None):
    keys = ('torch', 'cuda', 'cudnn', 'gpu', 'timm', 'numpy', 'driver', 'platform', 'hostname', 'python')
    return dict(started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), start_epoch=int(start_epoch), resumed_from=resumed_from,
                environment={k: (environment or {}).get(k) for k in keys})
