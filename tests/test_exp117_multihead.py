"""CPU-only checks for EXP117 (H-B-47 label-source heads; tools/exp117_multihead.py, training/runners/exp117_multihead.py).

Rules: the runner is the EXP107 loop with only the approved ordered changes (difflib opcodes on the normalised source, so extra,
reordered or re-indented statements fail); head 0 of the multi-head model computes exactly the EXP113 model;
the masked summed loss equals per-head BCEWithLogitsLoss when every source is present and ignores absent cells; each head's pos_weight is
the EXP099 rule on its own labels (dread = EXP113's vector, relabel_v1 = EXP113's relabel_v1 registration); the served output is the
per-target head rank mean; the protocol equals EXP113 except the documented fields; queue entries are valid gpu_queue jobs (F0 only
by default, continue_on_failure false, F1-F4 only after a recorded F0 screen PASS); assembly pools
the served and every head's OOF. One CPU smoke (2 optimizer updates, --device cpu, subprocess). No GPU, no CUDA context.
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'src'))
from tools import exp117_multihead as g  # noqa: E402
from tools import exp113_radr50_local as e113  # noqa: E402
from tools import exp107_kfold_radr50 as e107  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402

UID = g.UID
T = list(g.T)
REAL = ROOT / g.PROTOCOL / 'manifest.json'
BASE = ROOT / g.BASE_PROTOCOL / 'manifest.json'
WEIGHTS = ROOT / e113.WEIGHTS_LOCAL
CORPUS = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
QUEUE = ROOT / g.QUEUE_FILE
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP117 protocol not prepared')
weights = pytest.mark.skipif(not WEIGHTS.exists(), reason='RadImageNet weights not present locally')
corpus = pytest.mark.skipif(not (CORPUS / 'manifest.json').exists(), reason='64-slot cache not present')
REPLACED_EXP107_LINES = {
    'from tools import exp107_kfold_radr50 as guard', 'import torch.nn as nn',
    'all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in oof_ids}}',
    'log(f"{a.tag} | {a.arch} RadImageNet | {a.input_policy} {a.norm} | fold {a.fold} seed {a.seed} mode {a.mode} | train {len(train_ids)} | val {len(oof_ids)} | "',
    'f"steps {steps} | Gold and metrics disabled")', 'pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=dev)',
    'model = built.to(dev)', 'init, init_matches, init_mismatched = init_record(model)', 'lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)',
    'if logits.shape != (len(uids), len(LAB)) or not torch.isfinite(logits).all():',
    'if logits.shape != (len(batch_uids), 12) or not torch.isfinite(logits).all():', 'loss = lossf(logits.float(), y)',
    'torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=a.amp != "fp16")', 'optimizer_updates += 1; tot += loss.item()',
    '"optimizer_updates": optimizer_updates, "fp16_skips": skipped}',
    'history.append(rec); log(f"ep{ep} loss {rec[\'loss\']:.4f} | fixed-final | {rec[\'epoch_seconds\']}s | peak {peak:.1f} GiB | skips {skipped}")',
    'write_predictions(out / "oof_best.csv", U, P)', 'log(f"K{a.k_eval} OOF written ({len(U)} rows, {time.time() - t_eval:.0f}s)")',
    'write_predictions(out / protocol["extra_eval"]["file"], U36, P36)',
    '"fold": a.fold, "label_set": guard.LABEL_SET, "mode": a.mode, "oof_files": {"k24": "oof_best.csv", f"k{k_extra}": protocol["extra_eval"]["file"]},',
    '"caveat": "weak-label fold OOF for ensembling / teacher pseudo-labels; one seed; remote runs are independent trajectories"}'}
# The approved EXP107 -> EXP117 runner change, in order: (opcode, EXP107 line index, EXP117 line index, EXP107 lines, EXP117 lines) of
# difflib.SequenceMatcher over _normalized() sources. Any extra, missing, reordered or re-indented statement changes this tuple.
APPROVED_EXP107_TO_EXP117_HUNKS = (
    ('insert', 11, 11, (
    ), (
        "REGISTERED_ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1'}",
        'def supply_registered_env(environ=None):',
        '    environ = os.environ if environ is None else environ',
        '    supplied = [k for k in REGISTERED_ENV if k not in environ]',
        '    for k in supplied:',
        '        environ[k] = REGISTERED_ENV[k]',
        '    return supplied',
    )),
    ('insert', 51, 58, (
    ), (
        '    supply_registered_env()',
    )),
    ('replace', 53, 61, (
        '    from tools import exp107_kfold_radr50 as guard',
    ), (
        '    from tools import exp117_multihead as guard',
    )),
    ('delete', 63, 71, (
        '    import torch.nn as nn',
    ), (
    )),
    ('insert', 69, 76, (
    ), (
        '    heads = [h["name"] for h in protocol["heads"]]',
    )),
    ('insert', 75, 83, (
    ), (
        '    head_labels = {}',
        '    for h in protocol["heads"]:',
        '        path = ROOT / h["path"]',
        '        if sha256_file(path) != h["sha256"]:',
        '            raise SystemExit(f"head label file changed: {h[\'name\']}")',
        '        head_labels[h["name"]], semantic = load_label_csv(path)',
        '        if semantic != h["semantic_sha256"]:',
        '            raise SystemExit(f"head label semantic hash mismatch: {h[\'name\']}")',
        '    if protocol["heads"][0]["semantic_sha256"] != label_hash:',
        '        raise SystemExit("the dread head is not the --labels file")',
    )),
    ('insert', 88, 106, (
    ), (
        '    head_pw = {n: guard.head_pos_weight(head_labels[n], train_ids) for n in heads}',
        '    for n in heads:',
        '        if not np.array_equal(head_pw[n], np.asarray(fold["head_pos_weight"][n], np.float32)):',
        '            raise SystemExit(f"{n} head pos_weight differs from the manifest fold vector")',
        '    if not np.array_equal(head_pw["dread"], pos_weight_np):',
        '        raise SystemExit("dread head pos_weight differs from the EXP113 vector")',
    )),
    ('replace', 94, 118, (
        '    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in oof_ids}}',
    ), (
        '    stacked = guard.stack_sources(head_labels, heads, train_ids)',
        '    absent = {n: int(sum(1 for u in train_ids if u not in head_labels[n])) for n in heads}',
        '    all_labels = {**stacked, **{u: np.zeros(len(LAB) * len(heads), np.float32) for u in oof_ids}}',
    )),
    ('insert', 109, 135, (
    ), (
        '    multihead_registered = protocol["multihead"]["multihead_initial_state_sha256"]',
    )),
    ('insert', 124, 151, (
    ), (
        '        multi = guard.build_multihead(built, len(heads))',
        '        multi_init, _ = guard.state_hashes(multi)',
    )),
    ('insert', 128, 157, (
    ), (
        '        if a.seed == protocol["seed"] and multi_init != multihead_registered and not relaxed:',
        '            raise SystemExit("multi-head initial state differs from the registration")',
    )),
    ('insert', 131, 162, (
    ), (
        '                              heads=heads, head_pos_weight={n: [round(float(x), 6) for x in v] for n, v in head_pw.items()}, absent_rows=absent,',
    )),
    ('insert', 132, 164, (
    ), (
        '                              multihead_initial_state_sha256=multi_init, multihead_initial_state_matches_registered=multi_init == multihead_registered,',
    )),
    ('replace', 157, 190, (
        '    log(f"{a.tag} | {a.arch} RadImageNet | {a.input_policy} {a.norm} | fold {a.fold} seed {a.seed} mode {a.mode} | train {len(train_ids)} | val {len(oof_ids)} | "',
        '        f"steps {steps} | Gold and metrics disabled")',
        '    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=dev)',
    ), (
        '    log(f"{a.tag} | {a.arch} RadImageNet x{len(heads)} heads {heads} | {a.input_policy} {a.norm} | fold {a.fold} seed {a.seed} mode {a.mode} | "',
        '        f"train {len(train_ids)} | val {len(oof_ids)} | steps {steps} | Gold and metrics disabled")',
        '    pos_weight = torch.tensor(np.stack([head_pw[n] for n in heads]), dtype=torch.float32, device=dev)',
    )),
    ('replace', 165, 198, (
        '    model = built.to(dev)',
        '    init, init_matches, init_mismatched = init_record(model)',
    ), (
        "    init, init_matches, init_mismatched = init_record(built)  # EXP113's trunk + head state (enforced below)",
        '    model = guard.build_multihead(built, len(heads)).to(dev)',
        '    multi_init, _ = guard.state_hashes(model)',
    )),
    ('insert', 169, 203, (
    ), (
        '                  "multihead_initial_state_sha256": multi_init, "multihead_initial_state_matches_registered": multi_init == multihead_registered,',
    )),
    ('insert', 173, 208, (
    ), (
        '                  "heads": heads, "head_pos_weight": {n: head_pw[n].tolist() for n in heads}, "head_label_files": {h["name"]: h["sha256"] for h in protocol["heads"]},',
        '                  "absent_training_rows": absent, "loss": protocol["multihead"]["loss"], "served_output": protocol["multihead"]["serving"],',
    )),
    ('insert', 184, 221, (
    ), (
        '    if a.seed == protocol["seed"] and multi_init != multihead_registered and not relaxed:',
        '        raise SystemExit("multi-head initial state differs from the registration")',
    )),
    ('delete', 197, 236, (
        '    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)',
    ), (
    )),
    ('replace', 210, 248, (
        '            if logits.shape != (len(uids), len(LAB)) or not torch.isfinite(logits).all():',
    ), (
        '            if logits.shape != (len(uids), len(heads), len(LAB)) or not torch.isfinite(logits).all():',
    )),
    ('insert', 214, 252, (
    ), (
        '    def write_outputs(P, U, served_name, suffix):',
        '        for h, n in enumerate(heads):',
        '            write_predictions(out / f"oof_head_{n}{suffix}.csv", U, P[:, h])',
        '        write_predictions(out / served_name, U, guard.rank_mean(P))',
    )),
    ('insert', 218, 260, (
    ), (
        '        head_tot, norms, clipped = np.zeros(len(heads)), [], 0',
    )),
    ('replace', 222, 265, (
        '                if logits.shape != (len(batch_uids), 12) or not torch.isfinite(logits).all():',
    ), (
        '                if logits.shape != (len(batch_uids), len(heads), 12) or not torch.isfinite(logits).all():',
    )),
    ('replace', 224, 267, (
        '                loss = lossf(logits.float(), y)',
    ), (
        '                loss, head_losses = guard.masked_multihead_bce(logits.float(), y, pos_weight)',
    )),
    ('replace', 229, 272, (
        '            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=a.amp != "fp16")',
    ), (
        '            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=a.amp != "fp16")',
        '            norm_value = float(total_norm)',
        '            if math.isfinite(norm_value):',
        '                norms.append(norm_value); clipped += norm_value > 3.0',
    )),
    ('replace', 238, 284, (
        '            optimizer_updates += 1; tot += loss.item()',
    ), (
        '            optimizer_updates += 1; tot += loss.item(); head_tot += head_losses.detach().float().cpu().numpy()',
    )),
    ('replace', 244, 290, (
        '               "optimizer_updates": optimizer_updates, "fp16_skips": skipped}',
        '        history.append(rec); log(f"ep{ep} loss {rec[\'loss\']:.4f} | fixed-final | {rec[\'epoch_seconds\']}s | peak {peak:.1f} GiB | skips {skipped}")',
    ), (
        '               "optimizer_updates": optimizer_updates, "fp16_skips": skipped,',
        '               "head_loss": {n: float(head_tot[h] / max(1, len(tl))) for h, n in enumerate(heads)},',
        '               "grad_norm_preclip_mean": float(np.mean(norms)) if norms else None, "clipped_fraction": clipped / max(1, len(norms))}',
        '        history.append(rec); log(f"ep{ep} loss {rec[\'loss\']:.4f} | fixed-final | {rec[\'epoch_seconds\']}s | peak {peak:.1f} GiB | skips {skipped} | "',
        '                                 f"clipped {rec[\'clipped_fraction\']:.2f}")',
    )),
    ('insert', 251, 300, (
    ), (
        '            best.update(model_class="exp117 MultiHeadRaptor (tools/exp117_multihead.build_multihead)", heads=heads,',
        '                        serving=protocol["multihead"]["serving"])',
    )),
    ('replace', 264, 315, (
        '    write_predictions(out / "oof_best.csv", U, P)',
        '    log(f"K{a.k_eval} OOF written ({len(U)} rows, {time.time() - t_eval:.0f}s)")',
    ), (
        '    write_outputs(P, U, "oof_best.csv", "")',
        '    log(f"K{a.k_eval} OOF written ({len(U)} rows, {len(heads)} heads + rank mean, {time.time() - t_eval:.0f}s)")',
    )),
    ('replace', 270, 321, (
        '    write_predictions(out / protocol["extra_eval"]["file"], U36, P36)',
    ), (
        '    write_outputs(P36, U36, protocol["extra_eval"]["file"], f"_k{k_extra}")',
    )),
    ('replace', 278, 329, (
        '               "fold": a.fold, "label_set": guard.LABEL_SET, "mode": a.mode, "oof_files": {"k24": "oof_best.csv", f"k{k_extra}": protocol["extra_eval"]["file"]},',
        '               "caveat": "weak-label fold OOF for ensembling / teacher pseudo-labels; one seed; remote runs are independent trajectories"}',
    ), (
        '               "fold": a.fold, "label_set": guard.LABEL_SET, "mode": a.mode, "heads": heads,',
        '               "oof_files": {"k24": "oof_best.csv", f"k{k_extra}": protocol["extra_eval"]["file"],',
        '                             "heads": {n: [f"oof_head_{n}.csv", f"oof_head_{n}_k{k_extra}.csv"] for n in heads}},',
        '               "caveat": "weak-label fold OOF; served = per-target head rank mean within the fold\'s validation set; one seed"}',
    )),
    ('insert', 282, 335, (
    ), (
        '    names += [f"oof_head_{n}{s}.csv" for n in heads for s in ("", f"_k{k_extra}")]',
    )),
)


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def guard_env(monkeypatch):
    for k, v in {**g.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    monkeypatch.delenv(e107.WEIGHTS_ENV, raising=False)
    return monkeypatch


def runner():
    return load_path('exp117_runner', ROOT / g.RUNNER)


def parsed(argv):
    return runner().build_parser().parse_args(argv[2:])


def _mutate(argv, flag, value=None):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag] + ([str(value)] if value is not None else [])
    return argv


def without_thread_env():
    return {k: v for k, v in e113.local_env().items() if k not in g.THREAD_ENV}


def _normalized(path):
    """Source after the module docstring: trailing '# EXP107' / '# EXP117' markers and trailing spaces removed, blank lines dropped,
    indentation KEPT (moving a statement into or out of a block is a change)."""
    text = Path(path).read_text(encoding='utf-8').split('"""', 2)[2]
    return [x for x in (re.sub(r'\s*#\s*EXP1(07|17)\b.*$', '', line).rstrip() for line in text.splitlines()) if x.strip()]


def ordered_hunks(old, new):
    """Every non-equal difflib opcode as (opcode, old index, new index, old lines, new lines), in order."""
    return tuple((tag, i1, j1, tuple(old[i1:i2]), tuple(new[j1:j2]))
                 for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if tag != 'equal')


def _first_difference(got, want):
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return i, a, b
    return min(len(got), len(want)), got[len(want):len(want) + 1], want[len(got):len(got) + 1]


def _removed_set(old, new):
    """The check this test replaced: the set of stripped EXP107 lines absent from EXP117 (blind to order and to added lines)."""
    present = {y.strip() for y in new}
    return {x.strip() for x in old if x.strip() not in present}


# ------------------------------------------------------------------ names and the runner text
def test_names_are_new_and_never_collide():
    assert g.PROTOCOL not in (e113.PROTOCOL, e107.PROTOCOL) and g.OOF_DIR not in (e113.OOF_DIR, e107.OOF_DIR) and g.VERSION == '1.0.0'
    for k in g.FOLDS:
        for mode in ('full', 'smoke'):
            new = g.tag_for(k, mode)
            for old in (e113.tag_for(k, mode), e107.tag_for(k, mode), f'EXP116_TEACHER_FILL_F{k}'):
                assert new != old and old not in new and new not in old
    assert g.HEAD_NAMES == ('dread', 'relabel_v1', 'steven_v4', 'pilkwang_v2') and g.HEADS[0][1] == g.DREAD
    assert 'oof_head_pilkwang_v2_k36.csv' in g.RUN_FILES and set(e113.RUN_FILES) <= set(g.RUN_FILES)


def test_runner_is_the_exp107_loop_with_only_the_approved_ordered_changes():
    old, new = _normalized(ROOT / e107.RUNNER), _normalized(ROOT / g.RUNNER)
    got = ordered_hunks(old, new)
    assert got == APPROVED_EXP107_TO_EXP117_HUNKS, _first_difference(got, APPROVED_EXP107_TO_EXP117_HUNKS)
    assert {x.strip() for h in got for x in h[3]} == REPLACED_EXP107_LINES  # the 21 replaced/deleted EXP107 lines, now in their places
    assert sum(len(h[3]) for h in got) == 21 and sum(len(h[4]) for h in got) == 75 and len(new) - len(old) == 54
    shared = load_path('exp107_parser_117', ROOT / e107.RUNNER).build_parser()
    assert [(a.dest, a.required, a.choices) for a in runner().build_parser()._actions] == [(a.dest, a.required, a.choices) for a in shared._actions]
    text = (ROOT / g.RUNNER).read_text(encoding='utf-8')
    names = set(re.findall(r'guard\.([A-Za-z_]+)', text))
    assert all(hasattr(g, n) for n in names), sorted(n for n in names if not hasattr(g, n))
    for name in ('worker_init', 'read_rows', 'install_gold_blocker', 'uid_hash', 'read', 'canonical_order', 'validate_fold_membership',
                 'pos_weight', 'fold_contract', 'state_hashes', 'mismatched_keys'):
        assert getattr(g, name) is getattr(e107, name), name


def test_runner_supplies_only_missing_thread_env():
    env = {'OMP_NUM_THREADS': '4'}
    assert runner().supply_registered_env(env) == ['MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'HF_HUB_OFFLINE']
    assert env['OMP_NUM_THREADS'] == '4'


@pytest.mark.parametrize('mutation', ['extra_optimizer_update', 'reordered_statements', 'moved_out_of_block'])
def test_ordered_check_rejects_what_the_set_check_missed(mutation):
    old, new = _normalized(ROOT / e107.RUNNER), _normalized(ROOT / g.RUNNER)
    m = list(new)
    find = lambda text: next(i for i, x in enumerate(m) if text in x)
    if mutation == 'extra_optimizer_update':
        i = find('scaler.step(opt); scaler.update()')
        m.insert(i + 1, m[i])
    elif mutation == 'reordered_statements':
        i, j = find('scaler.scale(loss).backward()'), find('scaler.unscale_(opt)')
        m[i], m[j] = m[j], m[i]
    else:
        i = find('sched.step()')
        m[i] = m[i][4:]
    assert _removed_set(old, m) == REPLACED_EXP107_LINES  # the former set comparison accepts every one of these
    assert ordered_hunks(old, m) != APPROVED_EXP107_TO_EXP117_HUNKS


# ------------------------------------------------------------------ pure helpers
def test_stack_sources_and_head_pos_weight():
    a = {'u1': np.full(12, 0.9, np.float32), 'u2': np.full(12, 0.1, np.float32), 'u3': np.full(12, 0.2, np.float32)}
    b = {'u1': np.full(12, 0.6, np.float32), 'u3': np.full(12, 0.4, np.float32)}
    st = g.stack_sources({'a': a, 'b': b}, ['a', 'b'], ['u1', 'u2'])
    assert st['u1'].dtype == np.float32 and st['u1'].shape == (24,) and np.allclose(st['u1'][12:], 0.6)
    assert np.isnan(st['u2'][12:]).all() and np.allclose(st['u2'][:12], 0.1)
    assert np.array_equal(g.head_pos_weight(a, ['u1', 'u2', 'u3']), e107.pos_weight(a, ['u1', 'u2', 'u3']))
    assert np.array_equal(g.head_pos_weight(b, ['u1', 'u2', 'u3']), e107.pos_weight(b, ['u1', 'u3']))
    with pytest.raises(ValueError):
        g.head_pos_weight(b, ['u2'])


def test_rank_mean():
    rng = np.random.default_rng(0)
    P = rng.random((40, 4, 12))
    R = g.rank_mean(P)
    assert R.shape == (40, 12) and R.min() > 0 and R.max() <= 1
    same = np.repeat(P[:, :1], 4, axis=1)
    assert np.allclose(g.rank_mean(same)[:, 3], k99.avg_rank(P[:, 0, 3]) / 40)
    warped = P.copy(); warped[:, 2] = P[:, 2] ** 5  # monotone per-head rescaling (e.g. a different pos_weight) changes nothing
    assert np.array_equal(g.rank_mean(warped), R)


def test_masked_loss_equals_per_head_bce_and_ignores_absent_cells():
    import torch
    torch.manual_seed(0)
    B, H, C = 6, 4, 12
    logits = torch.randn(B, H, C, requires_grad=True)
    y = torch.rand(B, H * C)
    pw = torch.rand(H, C) * 3 + 1
    total, per = g.masked_multihead_bce(logits, y, pw)
    for h in range(H):
        ref = torch.nn.BCEWithLogitsLoss(pos_weight=pw[h])(logits[:, h], y.view(B, H, C)[:, h])
        assert torch.allclose(per[h], ref, atol=1e-6)
    assert torch.allclose(total, per.sum())
    y2 = y.clone().view(B, H, C); y2[:2, 1] = float('nan'); y2[:, 3] = float('nan')
    total2, per2 = g.masked_multihead_bce(logits, y2.view(B, H * C), pw)
    ref1 = torch.nn.BCEWithLogitsLoss(pos_weight=pw[1])(logits[2:, 1], y.view(B, H, C)[2:, 1])
    assert torch.allclose(per2[1], ref1, atol=1e-6) and float(per2[3]) == 0.0 and torch.isfinite(total2)
    total2.backward()
    assert torch.count_nonzero(logits.grad[:, 3]) == 0 and torch.count_nonzero(logits.grad[:2, 1]) == 0 and torch.count_nonzero(logits.grad[2:, 1]) > 0


def test_multihead_head0_is_the_base_model_and_heads_are_independent_copies():
    import torch
    from raptor_common import build_model, seed_all
    seed_all(1)
    base = build_model('resnet18', pretrained=False).eval()
    multi = g.build_multihead(base, 4).eval()
    x = torch.randn(2, 3, 3, 64, 64)
    with torch.no_grad():
        out, ref = multi(x), base(x)
    assert out.shape == (2, 4, 12) and torch.equal(out[:, 0], ref) and all(torch.equal(out[:, h], ref) for h in range(4))
    keys = multi.state_dict().keys()
    assert {k.split('.')[0] for k in keys} == {'backbone', 'heads'} and len([k for k in keys if k.startswith('heads.')]) == 4 * 8
    assert multi.backbone is base.backbone and multi.heads[0].clsW is not base.clsW and multi.heads[0].clsW is not multi.heads[1].clsW
    heads = [p for n, p in multi.named_parameters() if not n.startswith('backbone.')]
    assert sum(p.numel() for p in heads) == 4 * sum(p.numel() for n, p in base.named_parameters() if not n.startswith('backbone.'))


# ------------------------------------------------------------------ the registration
@prepared
def test_manifest_equals_exp113_except_documented_fields():
    p, p113 = g.read(REAL), g.read(BASE)
    differing = {k for k in set(p) | set(p113) if p.get(k) != p113.get(k)}
    assert differing <= set(g.CHANGED_FIELDS), differing - set(g.CHANGED_FIELDS)
    for key in ('common_args', 'seed', 'recipe', 'resource_args', 'smoke_allowed', 'training_code_sha256', 'environment', 'corpus_manifest_sha256',
                'paths', 'label_set', 'fold_protocol', 'total_steps', 'extra_eval', 'init_weights', 'input', 'initial_state_sha256', 'initial_state'):
        assert p[key] == p113[key], key
    for k in g.FOLDS:
        new, old = p['folds'][str(k)], p113['folds'][str(k)]
        assert {x for x in set(new) | set(old) if new.get(x) != old.get(x)} <= set(g.FOLD_CHANGED_FIELDS)
        assert new['head_pos_weight']['dread'] == old['pos_weight'] == new['pos_weight']
    assert [h['name'] for h in p['heads']] == list(g.HEAD_NAMES) and all(h['g1_missing'] == 0 for h in p['heads'])
    assert p['heads'][0]['sha256'] == p113['label_set']['sha256'] and p['schema'] == g.SCHEMA and p['plan_item'] == 'H-B-47'
    assert p['multihead']['multihead_initial_state_sha256'] and p['local']['runner'] == g.RUNNER and p['local']['loop_source'] == e107.RUNNER
    for path, expected in p['source_bindings'].items():
        assert g.sha256(ROOT / path) == expected, path
    assert {g.RUNNER, g.TOOL, *[x for _, x in g.HEADS]} <= set(p['source_bindings'])


@prepared
def test_head_pos_weight_is_each_sources_own_rule():
    p = g.read(REAL)
    reg = ROOT / e113.LABEL_SETS_DIR / 'relabel_v1.json'
    for name, path in g.HEADS:
        labels, _ = k99._load_labels(ROOT / path)
        for k in g.FOLDS:
            train = g.read(ROOT / p['folds'][str(k)]['ordered_ids']['path'])['train']
            assert np.array_equal(e107.pos_weight(labels, train), np.asarray(p['folds'][str(k)]['head_pos_weight'][name], np.float32)), (name, k)
            if name == 'relabel_v1' and reg.exists():
                assert p['folds'][str(k)]['head_pos_weight'][name] == g.read(reg)['pos_weight'][str(k)]


@prepared
@weights
@pytest.mark.parametrize('fold', [0, 4])
def test_runner_argv_round_trip_validates(guard_env, fold):
    p = g.read(REAL)
    argv = g.runner_argv(p, fold)
    assert argv[1] == str(ROOT / g.RUNNER) and argv[argv.index('--labels') + 1] == str(ROOT / g.DREAD)
    args = parsed(argv)
    for k, v in p['common_args'].items():
        assert getattr(args, k) == v and type(getattr(args, k)) is type(v), k
    got, spec, ordered, out_root = g.validate_arguments(args, ROOT)
    assert spec['tag'] == g.tag_for(fold) and got == p and out_root == (ROOT / 'artifacts' / 'experiments').resolve()


@prepared
@weights
@pytest.mark.parametrize('flag, value, message', [
    ('--seed', '43', 'seed changed'), ('--tag', 'EXP113_RADR50_LOCAL_F0', 'tag'), ('--tag', 'EXP117_MULTIHEAD_F2', 'tag'),
    ('--protocol', str(BASE), 'wrong protocol'), ('--head-lr', '0.01', 'frozen argument changed: head_lr'),
    ('--device', 'cpu', 'frozen argument changed: device'), ('--labels', str(ROOT / g.HEADS[1][1]), 'input path differs: labels'),
    ('--workers', '99', 'workers'), ('--output-root', str(ROOT / 'artifacts' / 'other'), 'artifacts/experiments only')])
def test_validate_arguments_rejects(guard_env, flag, value, message):
    argv = _mutate(g.runner_argv(g.read(REAL), 0), flag, value)
    with pytest.raises(ValueError, match=message):
        g.validate_arguments(parsed(argv), ROOT)


@prepared
@weights
def test_relax_env_smoke_rules_and_exp113_guard_refusal(guard_env, tmp_path):
    p = g.read(REAL)
    args = parsed(g.runner_argv(p, 0))
    guard_env.setenv('KNEE_REMOTE_RELAX_ENV', '1')
    with pytest.raises(ValueError, match='KNEE_REMOTE_RELAX_ENV'):
        g.validate_arguments(args, ROOT)
    guard_env.delenv('KNEE_REMOTE_RELAX_ENV')
    with pytest.raises(ValueError, match='wrong protocol'):
        e113.validate_arguments(args, ROOT)
    smoke = parsed(g.runner_argv(p, 0, 'smoke', workers=0, output_root=tmp_path))
    assert (smoke.device, smoke.amp, smoke.tag) == ('cpu', 'none', 'EXP117_MULTIHEAD_F0_SMOKE')
    assert g.validate_arguments(smoke, ROOT)[3] == tmp_path.resolve()


def _check_entries(entries, folds, after):
    assert [e['tag'] for e in entries] == [f'EXP117_MULTIHEAD_F{k}' for k in folds]
    assert [e.get('after') for e in entries] == [after] + [f'EXP117_MULTIHEAD_F{k}' for k in folds[:-1]]
    for k, e in zip(folds, entries):
        assert set(e) == {'tag', 'command', 'after', 'continue_on_failure'} and e['continue_on_failure'] is False
        cmd = e['command']
        assert all(isinstance(c, str) for c in cmd) and Path(cmd[1]) == ROOT / g.RUNNER and cmd[cmd.index('--fold') + 1] == str(k)
        assert cmd[cmd.index('--device') + 1] == 'cuda' and cmd[cmd.index('--workers') + 1] == '12' and not {'--output-root', '--validate-only'} & set(cmd)


def _screen_pass(path, **over):
    body = {**dict(verdict='PASS', tag='EXP117_MULTIHEAD_F0', protocol_sha256=g.sha256(REAL)), **over}
    path.write_text(json.dumps(body), encoding='utf-8')
    return path


def test_parse_folds():
    assert g.parse_folds('0') == [0] and g.parse_folds('1-4') == [1, 2, 3, 4] and g.parse_folds('4,1,2,3') == [1, 2, 3, 4]
    for bad in ('', '5', '0-5', '1,1', 'x', '2-1'):
        with pytest.raises(SystemExit):
            g.parse_folds(bad)


@prepared
def test_queue_defaults_to_f0_only_without_failure_continuation():
    entries = g.queue_entries(after='PREV')
    _check_entries(entries, [0], 'PREV')
    assert g.queue_entries([0], None)[0].get('after') is None


@prepared
def test_f1_to_f4_need_a_recorded_f0_screen_pass(tmp_path):
    for folds in ([0, 1, 2, 3, 4], [0, 1]):
        with pytest.raises(SystemExit, match='queued alone'):
            g.queue_entries(folds, 'PREV', screen_pass=str(_screen_pass(tmp_path / 'ok.json')))
    with pytest.raises(SystemExit, match='not found'):
        g.queue_entries([1, 2, 3, 4], 'EXP117_MULTIHEAD_F0')
    with pytest.raises(SystemExit, match='not found'):
        g.queue_entries([1, 2, 3, 4], 'EXP117_MULTIHEAD_F0', screen_pass=str(tmp_path / 'missing.json'))
    for name, over, message in [('fail.json', dict(verdict='FAIL'), 'PASS'), ('lower.json', dict(verdict='pass'), 'PASS'),
                                ('tag.json', dict(tag='EXP113_RADR50_LOCAL_F0'), 'not the EXP117_MULTIHEAD_F0 screen'),
                                ('proto.json', dict(protocol_sha256='0' * 64), 'not the current manifest')]:
        with pytest.raises(SystemExit, match=message):
            g.queue_entries([1, 2, 3, 4], 'EXP117_MULTIHEAD_F0', screen_pass=str(_screen_pass(tmp_path / name, **over)))
    (tmp_path / 'junk.json').write_text('{not json', encoding='utf-8')
    with pytest.raises(SystemExit, match='not readable JSON'):
        g.queue_entries([1, 2, 3, 4], 'EXP117_MULTIHEAD_F0', screen_pass=str(tmp_path / 'junk.json'))
    with pytest.raises(SystemExit, match='F0 .the screen itself. is queued without it'):
        g.queue_entries([0], 'PREV', screen_pass=str(_screen_pass(tmp_path / 'ok.json')))
    entries = g.queue_entries([1, 2, 3, 4], 'EXP117_MULTIHEAD_F0', screen_pass=str(_screen_pass(tmp_path / 'ok.json')))
    _check_entries(entries, [1, 2, 3, 4], 'EXP117_MULTIHEAD_F0')


@prepared
def test_queue_json_cli(tmp_path, capsys):
    out = tmp_path / 'f0.json'
    g.main(['queue-json', '--after', 'PREV', '--out', str(out)])
    saved = json.loads(out.read_text(encoding='utf-8'))
    _check_entries(saved, [0], 'PREV')
    assert b'\r\n' not in out.read_bytes()
    with pytest.raises(SystemExit):
        g.main(['queue-json', '--folds', '1-4', '--after', 'EXP117_MULTIHEAD_F0', '--out', str(tmp_path / 'rest.json')])
    assert not (tmp_path / 'rest.json').exists()
    g.main(['queue-json', '--folds', '1-4', '--after', 'EXP117_MULTIHEAD_F0', '--screen-pass', str(_screen_pass(tmp_path / 'ok.json')),
            '--out', str(tmp_path / 'rest.json')])
    _check_entries(json.loads((tmp_path / 'rest.json').read_text(encoding='utf-8')), [1, 2, 3, 4], 'EXP117_MULTIHEAD_F0')


@prepared
def test_saved_queue_entries_match_the_protocol():
    if not QUEUE.exists():
        pytest.skip('queue-entries.json not written')
    saved = json.loads(QUEUE.read_text(encoding='utf-8'))
    assert saved == g.queue_entries([0], saved[0].get('after')) and saved[0]['after'] == g.DEFAULT_AFTER == 'EXP113_RADR50_LOCAL_F4'
    assert len(saved) == 1 and saved[0]['continue_on_failure'] is False
    assert not (ROOT / g.QUEUE_FILE_REST).exists()  # F1-F4 are not generated before a recorded F0 screen pass


# ------------------------------------------------------------------ verification and assembly (synthetic workspace)
@pytest.fixture()
def mini(tmp_path):
    uids = [f'u{i:02d}' for i in range(15)]
    y = np.array([i % 2 for i in range(15)])
    (tmp_path / Path(g.DREAD).parent).mkdir(parents=True)
    pd.DataFrame({UID: uids, **{t: np.where(y == 1, 0.95, 0.05) for t in T}}).to_csv(tmp_path / g.DREAD, index=False)
    folds = {k: dict(train=[u for u in uids if u not in uids[3 * k:3 * k + 3]], val=uids[3 * k:3 * k + 3]) for k in range(5)}
    proto = tmp_path / g.PROTOCOL; spec = {}
    for k, v in folds.items():
        (proto / f'fold{k}').mkdir(parents=True)
        (proto / f'fold{k}' / 'ordered-ids.json').write_text(json.dumps(v), encoding='utf-8')
        spec[str(k)] = dict(ordered_ids=dict(path=f'{g.PROTOCOL}/fold{k}/ordered-ids.json', train_sha256=g.uid_hash(v['train']), val_sha256=g.uid_hash(v['val'])),
                            n_train=12, n_val=3, steps=20, max_fp16_skips=0, tag=g.tag_for(k), pos_weight=[1.0] * 12,
                            head_pos_weight={n: [1.0 + h] * 12 for h, n in enumerate(g.HEAD_NAMES)})
    manifest = dict(schema=g.SCHEMA, seed=42, common_args=e107.EXPECTED_RECIPE, folds=spec, initial_state_sha256='init',
                    multihead=dict(multihead_initial_state_sha256='minit'), label_set=dict(semantic_sha256='dsem'),
                    source_bindings={g.RUNNER: 'runner-sha', g.TOOL: 'guard-sha'}, tags=dict(prefix=g.TAG_PREFIX))
    (proto / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return tmp_path, folds


def _fake_run(root, runs, k, folds, **overrides):
    p = g.read(root / g.PROTOCOL / 'manifest.json')
    tag = g.tag_for(k); run = runs / tag; run.mkdir(parents=True)
    val = folds[k]['val']
    good = pd.DataFrame({UID: val, **{t: [0.9 if int(u[1:]) % 2 else 0.1 for u in val] for t in T}})
    bad = pd.DataFrame({UID: val, **{t: [0.1 if int(u[1:]) % 2 else 0.9 for u in val] for t in T}})
    for name in g.RUN_FILES:
        if name.endswith('.csv'):
            (bad if 'pilkwang' in name else good).to_csv(run / name, index=False)
    prov = dict(fold=k, seed=42, label_set=g.LABEL_SET, label_semantic_sha256='dsem', protocol_sha256=g.sha256(root / g.PROTOCOL / 'manifest.json'),
                eval_ids_ordered_sha256=g.uid_hash(val), train_ids_ordered_sha256=g.uid_hash(folds[k]['train']), training_mode='full',
                checkpoint_selection='fixed-final', arch='resnet50', input_policy='center3', norm='radimagenet', remote_relaxed=False,
                initial_state_sha256='init', initial_state_matches_registered=True, multihead_initial_state_sha256='minit',
                multihead_initial_state_matches_registered=True, environment_matches_baseline=True, runner_sha256='runner-sha', guard_sha256='guard-sha',
                heads=list(g.HEAD_NAMES), head_pos_weight=p['folds'][str(k)]['head_pos_weight'])
    prov.update(overrides)
    (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    (run / 'summary.json').write_text(json.dumps(dict(provenance=prov, optimizer_updates=20, skipped_updates=0, selected_epoch_1based=20, train=12, val=3,
                                                      args=dict(tag=tag), total_seconds=1)), encoding='utf-8')
    (run / 'history.json').write_text('[]', encoding='utf-8'); (run / 'model_best.pt').write_bytes(b'w')
    names = [n for n in g.RUN_FILES if n != 'output-inventory.json']
    (run / 'output-inventory.json').write_text(json.dumps({n: dict(sha256=g.sha256(run / n)) for n in names}), encoding='utf-8')


def test_assemble_pools_served_and_every_head(mini):
    root, folds = mini
    runs = root / 'artifacts' / 'experiments'
    for k in folds:
        _fake_run(root, runs, k, folds)
    body = g.assemble(root=root)
    m = body['metrics']
    assert set(m) == {'served', *g.HEAD_NAMES} and m['served']['k24']['pooled']['macro12'] == 1.0 and m['pilkwang_v2']['k36']['pooled']['macro12'] == 0.0
    out = root / g.OOF_DIR
    assert {f'oof_head_{n}_k24.csv' for n in g.HEAD_NAMES} | {'oof_k24.csv', 'oof_k36.csv'} <= {x.name for x in out.iterdir()}
    with pytest.raises(SystemExit):
        g.assemble(root=root)


@pytest.mark.parametrize('damage, message', [(dict(multihead_initial_state_matches_registered=False), 'multihead_initial_state'),
                                             (dict(heads=['dread']), 'heads'), (dict(guard_sha256='x'), 'guard_sha256'),
                                             (dict(head_pos_weight={n: [9.0] * 12 for n in g.HEAD_NAMES}), 'head pos_weight')])
def test_verify_run_refuses(mini, damage, message):
    root, folds = mini
    runs = root / 'runs'
    for k in folds:
        _fake_run(root, runs, k, folds, **(damage if k == 1 else {}))
    p = g.read(root / g.PROTOCOL / 'manifest.json')
    problems = g.verify_run(runs / g.tag_for(1), p, 1, g.sha256(root / g.PROTOCOL / 'manifest.json'))
    assert problems and any(message in x for x in problems), problems
    with pytest.raises(SystemExit):
        g.assemble(runs_dir=runs, root=root)


# ------------------------------------------------------------------ the runner on CPU (subprocess; no CUDA context)
@prepared
@weights
@corpus
def test_validate_only_on_cpu():
    p = g.read(REAL)
    fold = 0
    argv = g.runner_argv(p, fold, 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=without_thread_env(), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    out = json.loads(r.stdout)
    spec = p['folds'][str(fold)]
    assert out['status'] == 'VALIDATED_NO_TRAINING' and out['initial_state_matches_registered'] is True
    assert out['multihead_initial_state_matches_registered'] is True and out['heads'] == list(g.HEAD_NAMES)
    assert out['head_pos_weight'] == {n: [round(float(x), 6) for x in v] for n, v in spec['head_pos_weight'].items()}
    assert out['absent_rows'] == {n: 0 for n in g.HEAD_NAMES} and out['steps'] == spec['steps']


@prepared
@weights
@corpus
def test_cpu_smoke_two_optimizer_updates(tmp_path):
    p = g.read(REAL)
    fold = 1
    argv = g.runner_argv(p, fold, 'smoke', workers=2, output_root=tmp_path, smoke=dict(train=16, val=2, epochs=1))
    r = subprocess.run(argv, cwd=ROOT, env=without_thread_env(), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    run = tmp_path / g.tag_for(fold, 'smoke')
    s = g.read(run / 'summary.json'); prov = s['provenance']
    assert s['optimizer_updates'] == 2 and s['skipped_updates'] == 0 and (s['train'], s['val']) == (16, 2)
    h = s['history'][0]
    assert np.isfinite(h['loss']) and set(h['head_loss']) == set(g.HEAD_NAMES) and abs(sum(h['head_loss'].values()) - h['loss']) < 1e-4
    assert prov['cuda_initialized'] is False and prov['initial_state_matches_registered'] and prov['multihead_initial_state_matches_registered']
    assert prov['head_pos_weight'] == p['folds'][str(fold)]['head_pos_weight'] and prov['actual_pos_weight'] == p['folds'][str(fold)]['pos_weight']
    heads = {n: pd.read_csv(run / f'oof_head_{n}.csv', dtype={UID: str}) for n in g.HEAD_NAMES}
    served = pd.read_csv(run / 'oof_best.csv', dtype={UID: str})
    P = np.stack([heads[n][T].to_numpy() for n in g.HEAD_NAMES], 1)
    assert np.allclose(served[T].to_numpy(), g.rank_mean(P)) and list(served.columns) == [UID] + T
    inv = g.read(run / 'output-inventory.json')
    assert set(inv) == set(g.RUN_FILES) - {'output-inventory.json'} and all(g.sha256(run / n) == v['sha256'] for n, v in inv.items())
