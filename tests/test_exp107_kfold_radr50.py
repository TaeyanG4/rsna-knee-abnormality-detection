"""CPU-only checks for EXP107 (RadImageNet ResNet50, centre slice x3; tools/exp107_kfold_radr50.py, training/runners/exp107_kfold_radr50.py,
training/src/raptor_radimagenet.py, tools/remote_run_v3_exp107.py, tools/remote_bundle_v3_exp107.py).

The protocol must equal the EXP101 seed-42 registration in folds, pos_weight, contract and every recipe value except the registered
changes (arch, init, input policy, normalisation); the RadImageNet checkpoint must load strictly; the centre-slice input must be exactly the
centre plane of the EXP101 triplet; the runner's loop must be the EXP101 loop; output names must never collide. No GPU, no training.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'src'))
from tools import exp107_kfold_radr50 as e  # noqa: E402
from tools import exp101_kfold_arch as b  # noqa: E402
from tools import exp101_kfold_rn34_s43 as s43  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
from tools import remote_bundle_v3_exp107 as bundle  # noqa: E402
import raptor_radimagenet as rad  # noqa: E402
from raptor_common import Corpus, windows_from, select_centers  # noqa: E402
from raptor_data import StudyWindows  # noqa: E402

REAL = ROOT / e.PROTOCOL / 'manifest.json'
BASE = ROOT / e.BASE_PROTOCOL / 'manifest.json'
WEIGHTS = ROOT / e.WEIGHTS_LOCAL
CORPUS = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP107 protocol not prepared')
weights = pytest.mark.skipif(not WEIGHTS.exists(), reason='RadImageNet weights not present locally')
corpus = pytest.mark.skipif(not (CORPUS / 'manifest.json').exists(), reason='64-slot cache not present')


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def runner():
    return load_path('exp107_runner', ROOT / e.RUNNER)


@pytest.fixture()
def guard_env(monkeypatch):
    for k, v in {**e.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    monkeypatch.delenv(e.WEIGHTS_ENV, raising=False)
    return monkeypatch


# ------------------------------------------------------------------ names
def test_names_never_collide_with_exp101_runs():
    assert e.PROTOCOL not in (b.PROTOCOL, s43.PROTOCOL) and e.OOF_DIR not in (b.OOF_DIR, s43.OOF_DIR) and e.KERNEL not in (b.KERNEL, s43.KERNEL)
    for k in e.FOLDS:
        for mode in ('full', 'smoke'):
            new = e.tag_for(k, mode)
            for old in (b.tag_for(k, mode), s43.tag_for(k, mode)):
                assert new != old and old not in new and new not in old  # `kernels output --file-pattern <tag>` picks only this family
    assert e.tag_for(3) == 'EXP107_RADR50_DREAD_F3' and e.tag_for(0, 'smoke') == 'EXP107_RADR50_DREAD_F0_SMOKE'
    assert e.PROTOCOL == 'artifacts/experiments/EXP107_KFOLD_RADR50_PROTOCOL_V1' and e.KERNEL == 'taeyangg4/knee-run-v3-exp107-radr50-5fold'
    assert bundle.DATASET == 'knee-runners-v3-exp107' and bundle.PACKAGE.name == 'KNEE_RUNNERS_V3_EXP107'


def test_recipe_is_rn_clip_with_only_the_registered_changes():
    assert set(e.EXPECTED_RECIPE) == set(b.EXPECTED_RECIPE) | {'input_policy'}
    differing = {k for k in e.EXPECTED_RECIPE if e.EXPECTED_RECIPE[k] != b.EXPECTED_RECIPE.get(k)}
    assert differing == set(e.RECIPE_CHANGES) == {'arch', 'norm', 'pretrained', 'input_policy'}
    for k in ('res', 'k', 'k_eval', 'epochs', 'fixed_epoch', 'bs', 'accum', 'amp', 'amp_init_scale', 'eval_amp', 'bb_lr', 'head_lr', 'wd', 'window_policy'):
        assert e.EXPECTED_RECIPE[k] == b.EXPECTED_RECIPE[k], k
    assert e.EXPECTED_RECIPE['bb_lr'] == 5e-4 and e.EXPECTED_RECIPE['head_lr'] == 1e-3 and e.EXPECTED_RECIPE['wd'] == 0.02 and e.EXPECTED_RECIPE['epochs'] == 20


# ------------------------------------------------------------------ RadImageNet key mapping and strict load
def _timm_r50():
    import timm
    return timm.create_model('resnet50', pretrained=False, num_classes=0, global_pool='avg')


def _as_checkpoint(model):
    """A RadImageNet-layout state dict (backbone.<torchvision index>.*) built from a timm resnet50."""
    inverse = {v: k for k, v in rad.STAGE_MAP.items()}
    out = {}
    for k, v in model.state_dict().items():
        head, rest = k.split('.', 1)
        out[f'backbone.{inverse[head]}.{rest}'] = v.clone()
    return out


def test_key_map_covers_timm_resnet50_exactly():
    m = _timm_r50()
    ck = _as_checkpoint(m)
    assert len(ck) == rad.RADIMAGENET_R50_KEYS
    mapped = rad.map_radimagenet_state(ck)
    assert set(mapped) == set(m.state_dict())
    report = rad.strict_load_backbone(_timm_r50(), mapped)
    assert report['missing'] == report['unexpected'] == report['shape_mismatch'] == [] and report['params'] == rad.RADIMAGENET_R50_PARAMS


@pytest.mark.parametrize('bad', ['backbone.8.weight', 'fc.weight', 'backbone.2.x', 'module.backbone.0.weight'])
def test_key_map_refuses_unknown_keys(bad):
    with pytest.raises(ValueError):
        rad.map_radimagenet_state({bad: 0})


def test_strict_load_refuses_missing_and_nonfinite():
    import torch
    mapped = rad.map_radimagenet_state(_as_checkpoint(_timm_r50()))
    dropped = dict(mapped); dropped.pop('layer4.2.bn3.running_var')
    with pytest.raises(RuntimeError, match='missing'):
        rad.strict_load_backbone(_timm_r50(), dropped)
    broken = dict(mapped); broken['conv1.weight'] = broken['conv1.weight'].clone(); broken['conv1.weight'][0, 0, 0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite'):
        rad.strict_load_backbone(_timm_r50(), broken)
    extra = dict(mapped); extra['fc.weight'] = torch.zeros(12, 2048)
    with pytest.raises(RuntimeError, match='unexpected'):
        rad.strict_load_backbone(_timm_r50(), extra)


@weights
def test_real_checkpoint_loads_strictly_and_matches_the_anchor_encoder():
    import torch
    import torchvision
    assert rad.file_sha256(WEIGHTS) == rad.RADIMAGENET_R50_SHA256 == e.RADIMAGENET_SHA256
    torch.manual_seed(0)
    model, report = rad.build_radimagenet_model('resnet50', WEIGHTS)
    assert report['source_keys'] == report['target_keys'] == 318 and not report['missing'] and not report['unexpected'] and report['byte_equal_after_load']
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)

    class AnchorEncoder(torch.nn.Module):  # anchor_source.py _RadEncoder
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(*list(torchvision.models.resnet50(weights=None).children())[:-2])

    anchor = AnchorEncoder(); anchor.load_state_dict(state, strict=True); anchor.eval(); model.eval()
    x = (torch.rand(2, 1, 64, 64) * 2 - 1).expand(-1, 3, -1, -1).contiguous()
    with torch.no_grad():
        assert torch.equal(anchor.backbone(x).mean(dim=(2, 3)), model.backbone(x))
    with pytest.raises(ValueError):
        rad.build_radimagenet_model('resnet34', WEIGHTS)
    with pytest.raises(RuntimeError, match='sha256'):
        rad.build_radimagenet_model('resnet50', WEIGHTS, '0' * 64)


# ------------------------------------------------------------------ centre-slice input
def test_center3_is_the_centre_plane_of_the_triplet_synthetic():
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 256, size=(10, 336, 336), dtype=np.uint8)
    centers = [1, 4, 8, -1, 5]
    c3, tri = rad.center3_windows_from(vol, centers, 224), windows_from(vol, centers, 224)
    assert c3.shape == tri.shape == (5, 3, 224, 224) and c3.dtype == np.float32
    for ch in range(3):
        assert np.array_equal(c3[:, ch], tri[:, 1])
    assert not c3[3].any()
    same = rad.center3_windows_from(vol, [4], 336)
    assert np.array_equal(same[0, 2], vol[4].astype(np.float32) / 255.0)


def test_normalisation():
    x = np.array([0.0, 0.25, 1.0], np.float32)
    assert np.array_equal(rad.normalise(x, 'radimagenet'), np.array([-1.0, -0.5, 1.0], np.float32)) and rad.normalise(x, 'radimagenet').dtype == np.float32
    assert np.allclose(rad.normalise(np.full(3, 127 / 255, np.float32), 'radimagenet'), 127 / 127.5 - 1, rtol=0, atol=1e-6)  # anchor uint8/127.5-1
    with pytest.raises(ValueError):
        rad.normalise(x, 'other')


@corpus
@pytest.mark.parametrize('train', [True, False])
def test_dataset_uses_the_exp101_centres_and_jitter(train):
    c = Corpus(CORPUS)
    ids = [str(u) for u in c.ids[[3, 777, 2500]]]
    labels = {u: np.zeros(12, np.float32) for u in ids}
    old = StudyWindows(CORPUS, ids, labels, 224, 12, train, 'none', 42, window_policy='within-slot-valid')
    new = rad.CenterSliceWindows(CORPUS, ids, labels, 224, 12, train, 'none', 42, window_policy='within-slot-valid')
    norm = rad.CenterSliceWindows(CORPUS, ids, labels, 224, 12, train, 'radimagenet', 42, window_policy='within-slot-valid')
    for ds in (old, new, norm):
        ds.set_epoch(3)
    for i in range(len(ids)):
        xo, _, uo = old[i]; xn, _, un = new[i]; xr, _, _ = norm[i]
        assert uo == un and xo.shape == xn.shape == (12, 3, 224, 224)
        for ch in range(3):
            assert np.array_equal(xn[:, ch].numpy(), xo[:, 1].numpy())
        assert np.allclose(xr.numpy(), xn.numpy() * 2 - 1, atol=1e-6) and float(xr.min()) >= -1 and float(xr.max()) <= 1


# ------------------------------------------------------------------ the real registration
@prepared
def test_manifest_differs_from_exp101_only_in_registered_fields():
    p, p42 = e.read(REAL), e.read(BASE)
    assert e.sha256(BASE) == e.BASE_MANIFEST_SHA256
    differing = {k for k in set(p) | set(p42) if p.get(k) != p42.get(k)}
    assert differing <= set(e.CHANGED_FIELDS) and {'common_args', 'initial_state_sha256', 'folds', 'tags', 'init_weights'} <= differing
    for key in ('resource_args', 'smoke_allowed', 'training_code_sha256', 'environment', 'corpus_manifest_sha256', 'excluded_uids', 'fold_protocol',
                'paths', 'label_set', 'total_steps', 'pos_weight_rule', 'contract_rule', 'extra_eval', 'seed'):
        assert p[key] == p42[key], key
    assert p['common_args'] == e.EXPECTED_RECIPE and p['schema'] == e.SCHEMA and p['plan_item'] == 'H-B-24'
    for k in e.FOLDS:
        new, old = p['folds'][str(k)], p42['folds'][str(k)]
        assert {x for x in set(new) | set(old) if new.get(x) != old.get(x)} == {'tag'} and new['tag'] == e.tag_for(k)
        assert new['steps'] == (new['n_train'] // 8) * 20 and new['max_fp16_skips'] == new['steps'] // 100
    assert p['recipe']['changes'] == [*p42['recipe']['changes'], *e.CHANGE_NOTES]
    assert p['init_weights']['sha256'] == e.RADIMAGENET_SHA256 and p['init_weights']['kaggle_dataset'] == e.WEIGHTS_DATASET
    assert p['init_weights']['load']['missing'] == [] and p['init_weights']['load']['unexpected'] == [] and p['init_weights']['load']['source_keys'] == 318
    assert p['remote']['kernel'] == e.KERNEL and p['remote']['docker_image'] == e.DOCKER_IMAGE and p['evaluation']['oof_dir'] == e.OOF_DIR


@prepared
def test_bindings_hold_and_extend_exp101():
    p, p42 = e.read(REAL), e.read(BASE)
    assert {k: p['source_bindings'][k] for k in p42['source_bindings']} == p42['source_bindings']
    assert set(p['source_bindings']) - set(p42['source_bindings']) == {e.BASE_PROTOCOL + '/manifest.json', e.RUNNER, e.TOOL, e.RAD_MODULE,
                                                                       e.PROTOCOL + '/' + e.KEYS_FILE}
    for path, expected in p['source_bindings'].items():
        assert e.sha256(ROOT / path) == expected, path


@prepared
def test_pos_weight_recomputed_from_labels():
    p = e.read(REAL)
    labels, _ = k99._load_labels(ROOT / e.DREAD)
    for k in e.FOLDS:
        train = e.read(ROOT / p['folds'][str(k)]['ordered_ids']['path'])['train']
        assert np.array_equal(e.pos_weight(labels, train), np.asarray(p['folds'][str(k)]['pos_weight'], np.float32))


@prepared
@weights
def test_initial_state_backbone_is_the_checkpoint_bytes():
    import torch
    p = e.read(REAL)
    keys = e.read(ROOT / p['initial_state']['keys_file'])
    assert keys['combined_sha256'] == p['initial_state_sha256'] and len(keys['keys']) == p['initial_state']['keys'] == 326
    mapped = rad.map_radimagenet_state(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    backbone = {k: v for k, v in keys['keys'].items() if k.startswith('backbone.')}
    assert len(backbone) == 318
    for name, tensor in mapped.items():
        assert backbone['backbone.' + name][0] == hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(), name


# ------------------------------------------------------------------ argument validation through the runner parser
def parsed(runner, argv):
    return runner.build_parser().parse_args(argv[2:])


def _mutate(argv, flag, value=None):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag] + ([str(value)] if value is not None else [])
    return argv


@prepared
@weights
@pytest.mark.parametrize('fold', [0, 4])
def test_runner_argv_round_trip_validates(guard_env, runner, fold):
    p = e.read(REAL)
    argv = e.runner_argv(p, fold)
    assert argv[1] == str(ROOT / e.RUNNER) and argv[argv.index('--init-weights') + 1] == str(WEIGHTS)
    args = parsed(runner, argv)
    for k, v in p['common_args'].items():
        assert getattr(args, k) == v and type(getattr(args, k)) is type(v), k
    assert args.workers == 4 and args.seed == 42 and args.tag == e.tag_for(fold)
    got, spec, ordered, out_root = e.validate_arguments(args, ROOT)
    assert spec['tag'] == args.tag and len(ordered['train']) == spec['n_train'] and out_root == (ROOT / 'artifacts' / 'experiments').resolve()


@prepared
@weights
@pytest.mark.parametrize('flag, value, message', [
    ('--seed', '43', 'seed changed'), ('--tag', 'EXP101_RN34_DREAD_F0', 'tag'), ('--tag', 'EXP107_RADR50_DREAD_F1', 'tag'),
    ('--protocol', str(BASE), 'wrong protocol'), ('--bb-lr', '0.0003', 'frozen argument changed: bb_lr'), ('--arch', 'resnet34.a1_in1k', 'frozen argument changed: arch'),
    ('--device', 'cpu', 'frozen argument changed: device'), ('--workers', '99', 'workers'), ('--labels', 'storage/labels/other.csv', 'input path differs: labels'),
    ('--init-weights', str(ROOT / e.RUNNER), 'init weights'), ('--output-root', '/tmp/elsewhere', 'remote runs')])
def test_validate_arguments_rejects(guard_env, runner, flag, value, message):
    argv = _mutate(e.runner_argv(e.read(REAL), 0), flag, value)
    with pytest.raises(ValueError, match=message):
        e.validate_arguments(parsed(runner, argv), ROOT)


@prepared
@weights
def test_exp101_guard_refuses_this_protocol(guard_env, runner):
    with pytest.raises(ValueError, match='wrong protocol'):
        b.validate_arguments(parsed(runner, e.runner_argv(e.read(REAL), 0)), ROOT)


def test_runner_parser_accepts_only_the_registered_input_and_init(runner):
    base = e.runner_argv({'seed': 42, 'resource_args': {'workers': {'default': 4}}, 'common_args': e.EXPECTED_RECIPE,
                          'paths': {'corpus': 'c', 'labels': 'l', 'eval_labels': 'l'}}, 0, weights='w.pt')
    parsed(runner, base)
    for flag, value in (('--norm', 'imagenet'), ('--input-policy', 'triplet'), ('--pretrained', 'timm')):
        with pytest.raises(SystemExit):
            parsed(runner, _mutate(base, flag, value))


def test_runner_loop_is_the_exp101_loop():
    """From the optimizer to the end, the EXP107 runner equals the EXP101 runner once the marked EXP107 lines are removed."""
    marker = '    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]'
    ours = (ROOT / e.RUNNER).read_text(encoding='utf-8').split(marker, 1)[1]
    shared = (ROOT / b.RUNNER).read_text(encoding='utf-8').split(marker, 1)[1]
    ours = '\n'.join(line for line in ours.splitlines() if '# EXP107' not in line)
    assert ours == '\n'.join(shared.splitlines())


def test_verify_run_reports_missing_files(tmp_path):
    if not REAL.exists():
        pytest.skip('EXP107 protocol not prepared')
    problems = e.verify_run(tmp_path / e.tag_for(0), e.read(REAL), 0, e.sha256(REAL))
    assert problems and 'missing' in problems[0]


# ------------------------------------------------------------------ remote entry and bundle
def test_remote_entry_registers_the_job_and_replaces_only_the_weight_step():
    rr = load_path('remote_run_v3_exp107_test', ROOT / e.REMOTE_ENTRY)
    m = rr.load_base()
    assert m.JOBS[rr.JOB] == dict(tool='exp107_kfold_radr50', folds=[0, 1, 2, 3, 4]) and 'exp101:rn34-5fold' in m.JOBS
    assert m.TOOLS_DATASET == bundle.DATASET and m.fetch_weights.__name__ == 'fetch_weights' and m.fetch_weights.__module__ != m.__name__
    assert rr.JOB in bundle.JOBS


def test_locate_weights_picks_the_registered_hash(tmp_path):
    rr = load_path('remote_run_v3_exp107_test2', ROOT / e.REMOTE_ENTRY)
    (tmp_path / 'a').mkdir(); (tmp_path / 'b' / 'x').mkdir(parents=True)
    (tmp_path / 'a' / 'ResNet50.pt').write_bytes(b'other'); (tmp_path / 'b' / 'x' / 'ResNet50.pt').write_bytes(b'right')
    want = hashlib.sha256(b'right').hexdigest()
    path, hashes = rr.locate_weights(want, roots=(str(tmp_path),))
    assert Path(path) == tmp_path / 'b' / 'x' / 'ResNet50.pt' and len(hashes) == 2
    with pytest.raises(RuntimeError, match='no ResNet50.pt'):
        rr.locate_weights('0' * 64, roots=(str(tmp_path),))


@prepared
def test_bundle_ships_code_and_protocol_but_no_weights():
    files = bundle.bundle_files()
    for f in (e.RAD_MODULE, e.RUNNER, e.TOOL, e.REMOTE_ENTRY, 'tools/remote_run_v3.py', 'tools/exp101_kfold_arch.py', 'training/src/raptor_data.py',
              e.PROTOCOL + '/manifest.json', e.PROTOCOL + '/' + e.KEYS_FILE, e.DREAD):
        assert f in files, f
    assert not any(f.endswith('.pt') or f.startswith(('storage/weights', 'storage/competition', 'storage/gold')) for f in files)
    for f in bundle.CODE_FILES:
        assert (ROOT / f).is_file(), f


def test_kernel_metadata_and_push_rules():
    meta = bundle.kernel_metadata('taeyangg4', 'exp107:radr50-5fold')
    assert meta['id'] == e.KERNEL and meta['docker_image'] == e.DOCKER_IMAGE and meta['machine_shape'] == 'NvidiaTeslaT4' and meta['is_private']
    assert meta['dataset_sources'] == ['taeyangg4/knee-runners-v3-exp107', 'taeyangg4/knee-raptor-corpus-336', 'marwanmath/resnet-50-radimagenet-marwan']
    argv = bundle.push_argv(ROOT / 'kaggle' / 'packages' / 'X')
    assert '--timeout' not in argv and argv[argv.index('--accelerator') + 1] == 'NvidiaTeslaT4' and Path(argv[argv.index('-p') + 1]).is_absolute()
    script = bundle.kernel_script('exp107:radr50-5fold', 'taeyangg4')
    assert 'remote_run_v3_exp107.py' in script and "'0,1,2,3,4'" in script and "'11.0'" in script and max(map(len, script.splitlines())) < 400
    compile(script, 'run.py', 'exec')
