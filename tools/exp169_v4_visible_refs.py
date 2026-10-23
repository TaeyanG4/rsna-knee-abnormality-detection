"""Independent, local three-study S43 CPU/fp32 and C4 GPU references.

Run from the project root with the system GPU Python. Outputs are create-only;
the staged release, visible bundles, and training environments are read-only.
The C4 command consumes the local GPU and must use the project's GPU queue.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

from tools import exp169_release_v4 as v4
from tools import exp169_v2_local_refs as local

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/releases/EXP169_V4_A4_B4_s43_stage_v411'
B172 = ROOT / v4.B172
OUT = ROOT / 'artifacts/experiments/EXP169_RELEASE_V2/local-refs/v4-visible'
COMPETITION = local.COMPETITION
DICOM = local.DICOM_ROOT
B172_INPUT_ANCHOR = ROOT / 'artifacts/experiments/EXP169_RELEASE_V2/local-refs/b172-own-anchor'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _own_folder(visible):
    """The downloaded A4/B4 bundles give S42 different folder names."""
    candidate = local.read_json(visible / v4.RECEIPT).get('candidate')
    need(candidate in ('A4', 'B4') and visible.name == 'visible-' + candidate.lower(),
         'Visible bundle candidate/path differs')
    return 'exp169-own42' if candidate == 'A4' else 'exp169-trio'


def _established_inputs(ids, provenance, expected, anchor=B172_INPUT_ANCHOR):
    """Bind visible records and the current DICOM tree to an earlier local run."""
    anchor = Path(anchor)
    receipt = local.read_json(anchor.parent / (anchor.name + '.local-receipt.json'))
    records_path = anchor / 'inputs.json'
    need(receipt.get('status') == 'COMPLETE_LOCAL_OWN_ANCHOR_REFERENCE' and
         receipt.get('ids') == ids and
         receipt.get('files', {}).get('inputs.json') == local.sha(records_path),
         'B172 independent input receipt differs')
    need(provenance == receipt.get('inputs'),
         'Current source CSV/DICOM tree differs from B172 independent input receipt')
    need(expected == local.read_json(records_path),
         'Visible decoded inputs differ from B172 independent input records')


def _inputs(stage, competition, dicom, visible):
    ids, provenance = local.visible_inputs(competition, dicom)
    expected = local.read_json(visible / _own_folder(visible) / 'inputs.json')
    need(set(expected) == set(ids), 'Visible input ID roster differs')
    _established_inputs(ids, provenance, expected)
    return ids, provenance, expected


def _staged_code(code, manifest, *, require_all_loaded=False):
    """Validate frozen B172 sources and every imported transitive module."""
    expected = manifest['source_code']
    names = {'exp169_partner_bundle_release.py', 'exp169_release_v2.py',
             *(name + '.py' for name in local.RUNTIME_MODULES)}
    need(set(expected) == names, 'B172 staged source inventory differs')
    for name, digest in expected.items():
        need(local.sha(code / name) == digest, 'B172 staged source hash differs: ' + name)
    loaded = local.staged_module_files(code)
    for name, item in loaded.items():
        need(item['sha256'] == expected[name + '.py'], 'Loaded B172 source hash differs: ' + name)
    need({'exp169_partner_bundle_release', 'exp169_release_v2'} <= set(loaded),
         'B172 staged scorer/wrapper not loaded')
    if require_all_loaded:
        need({name.removesuffix('.py') for name in names} == set(loaded),
             'B172 staged transitive runtime inventory incomplete')
    return loaded


def _load_b172_runtime(b172):
    """Temporarily hide v4's packaging import while loading frozen B172 code."""
    alias = 'tools.exp169_partner_bundle_release'
    package = sys.modules['tools']
    prior_module = sys.modules.pop(alias, None)
    prior_attr = getattr(package, 'exp169_partner_bundle_release', None)
    if hasattr(package, 'exp169_partner_bundle_release'):
        delattr(package, 'exp169_partner_bundle_release')
    try:
        return local.load_staged_partner(b172)
    finally:
        sys.modules.pop(alias, None)
        if prior_module is not None:
            sys.modules[alias] = prior_module
        if prior_attr is not None:
            package.exp169_partner_bundle_release = prior_attr


def _publish(out, raw_source, ids, input_records, arm, device, evidence):
    """Publish only after a complete independent run and input equality."""
    need(not out.exists(), 'Fresh reference output required: ' + str(out))
    if arm == 'S43':
        actual_inputs = local.read_json(raw_source.parent / 'inputs.json')
        need(actual_inputs == input_records, 'Local decoded inputs differ from visible input records')
    with np.load(raw_source, allow_pickle=False) as z:
        need(z['ids'].tolist() == ids, 'Reference raw IDs/order differ')
        if arm == 'S43':
            need(set(z.files) == {'ids', 'targets', 'own'} and z['targets'].tolist() == local.TARGETS,
                 'S43 raw contract differs')
        else:
            need(set(z.files) == {'ids', 'l2', 'dino', 'convnext'} and
                 z['convnext'].shape == (5, 3, 12), 'C4 raw contract differs')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + '.tmp-', dir=out.parent) as scratch:
        pending = Path(scratch) / 'complete'
        pending.mkdir()
        shutil.copyfile(raw_source, pending / 'raw.npz')
        receipt = dict(schema='exp169-v4-local-visible-reference-v1',
                       status='COMPLETE_LOCAL_REFERENCE', source='independent-serving-run',
                       candidate_arm=arm, device=device, ids=ids,
                       raw_sha256=local.sha(pending / 'raw.npz'),
                       visible_input_sha256=local.sha(evidence['visible_inputs_path']),
                       input_records_sha256=local.text_sha(json.dumps(input_records, sort_keys=True)),
                       tool='tools/exp169_v4_visible_refs.py', tool_sha256=local.sha(__file__),
                       **{k: v for k, v in evidence.items() if k != 'visible_inputs_path'})
        local.write_json(pending / 'local-receipt.json', receipt)
        need(not out.exists(), 'Fresh reference output required: ' + str(out))
        os.replace(pending, out)
    return receipt


def _install_s43_cpu_bind(module):
    """Patch the staged bind in its live globals so v4's S43 binding is visible."""
    bind_source, bind_patch = local.patch_source(inspect.getsource(module.bind),
                                                local.B174_BIND_PATCHES)
    scope = module.__dict__
    scope['_EXP169_LOCAL_DEVICE'] = 'cpu'
    exec(compile(bind_source, '<s43-local-cpu-bind>', 'exec'), scope)
    need(module.bind.__globals__ is scope, 'S43 bind detached from staged module globals')
    return bind_patch


def run_s43(stage=STAGE, competition=COMPETITION, dicom=DICOM, visible=None, out_root=OUT,
            dry_run=False):
    stage, out_root = Path(stage), Path(out_root)
    visible = Path(visible or stage / 'visible-a4')
    t0 = time.monotonic()
    ids, inputs, expected = _inputs(stage, competition, dicom, visible)
    assets = stage / 's43/assets'
    manifest = local.read_json(assets / 'release-manifest.json')
    name = v4.S43_NAME
    need(manifest['schema'] == 'exp169-s43-release-v4' and manifest['checkpoint'] == name and
         local.sha(assets / name) == manifest['checkpoint_sha256'] and
         local.sha(assets / 'public-serving.py') == manifest['serving_sha256'],
         'S43 stage asset binding differs')
    out = out_root / 's43-own'
    need(not out.exists(), 'Fresh S43 reference required')
    with tempfile.TemporaryDirectory(prefix='exp169-v4-s43-') as td:
        tmp = Path(td)
        copied = tmp / 'assets'
        shutil.copytree(assets, copied)
        need(all(local.sha(copied / p) == local.sha(assets / p) for p in
                 (name, 'public-serving.py', local.EXP152_FILE, 'release-manifest.json')),
             'S43 asset copy differs')
        module = local._load_module('exp152_own0949_release', copied / local.EXP152_FILE)
        # source_sha256 identifies the embedded public source, not the helper
        # file itself (whose byte hash is different in the frozen stage).
        need(module.SOURCE_SHA == manifest['source_sha256'],
             'S43 staged helper source binding differs')
        bind_patch = _install_s43_cpu_bind(module)
        serving, serving_patch = local.patch_source(
            (copied / 'public-serving.py').read_text(encoding='utf-8'),
            local.SERVING_PATCHES + local.B174_CPU_PATCHES)
        compile(serving, '<s43-local-cpu-serving>', 'exec')
        if dry_run:
            return dict(status='DRY_RUN_OK', ids=ids, asset_sha256=manifest['checkpoint_sha256'],
                        serving_patch=serving_patch, bind_patch=bind_patch)
        import torch
        module.runtime_info = lambda: local.b174_cpu_runtime(torch)
        view = local.competition_view(tmp / 'competition', competition, dicom)
        need(local.tree_digest(view / 'test_series') == inputs['dicom_tree'], 'DICOM copy differs')
        work = tmp / 'work'
        work.mkdir()
        ns = {'__name__': 'exp169_local_s43', '__file__': str(copied / 'public-serving.py'),
              '_EXP169_LOCAL_POOL': local._single_thread_pool}
        exec(compile(serving, '<s43-local-cpu-serving>', 'exec'), ns)
        fastread = local.install_fastread(ns)
        ns['find_test_root'] = lambda: str(view)
        v4.bind_s43(ns, manifest, {name: copied / name,
                    'public-serving.py': copied / 'public-serving.py'}, work, module)
        started = time.monotonic()
        ns['main']()
        seconds = time.monotonic() - started
        local.check_anchor_folder(work,
                                  local.read_json(visible / 'exp169-s43-own/binding.json'), ids)
        need(local.read_json(work / 'runtime.json')['device'] == 'cpu', 'CPU runtime not recorded')
        out_root.mkdir(parents=True, exist_ok=True)
        return _publish(out, work / 'raw.npz', ids, expected, 'S43', 'cpu-fp32', dict(
            visible_inputs_path=visible / _own_folder(visible) / 'inputs.json',
            stage_manifest_sha256=local.sha(assets / 'release-manifest.json'),
            checkpoint_sha256=manifest['checkpoint_sha256'],
            patches=dict(serving=serving_patch, bind=bind_patch, fastread=fastread),
            input_source=inputs, inference_seconds=seconds, wall_seconds=time.monotonic()-t0))


def _c4_local_function(base, staged_v2):
    """The v4 C4 transformations followed by the v2 one-GPU scheduling patch."""
    loader = inspect.getsource(base._load_models)
    loader = staged_v2.replace_once(loader,
        "need(ck.get('source_sha256') == entry['source_sha256'],",
        "need((entry.get('format') == 'authenticated-original' and sha(path) == entry['source_sha256']) or ck.get('source_sha256') == entry['source_sha256'],")
    v4.v2 = staged_v2  # this standalone process binds v4 metadata validation to staged v2
    scope = {**base.__dict__, 'model_meta': v4.c4_model_meta}
    exec(compile(loader, '<v4-c4-local-loader>', 'exec'), scope)
    source = inspect.getsource(base.partner_scores)
    for old, new in (
        ('center3 = center3_windows_from(volume, centers, 224)',
         "center3 = center3_windows_from(volume, centers, 224)\n                c4 = center3.copy()\n                odd = list(range(1, len(centers), 2))\n                c4[odd] = center3_windows_from(volume[:, 48:288, 48:288], [centers[j] for j in odd], 224)"),
        ("normalize_input(center3, 'imagenet'),",
         "normalize_input(center3, 'imagenet'), normalize_input(c4, 'imagenet'),"),
        ('i, rad, img, triplet, flags = job', 'i, rad, img, mixed_img, triplet, flags = job'),
        ("x = triplet if family == 'dino' else rad if family == 'l2' else img",
         "x = triplet if family == 'dino' else rad if family == 'l2' else mixed_img"),
    ):
        source = staged_v2.replace_once(source, old, new)
    source, patches = local.patch_source(source, local.PARTNER_PATCHES)
    scope['_load_models'] = scope['_load_models']
    exec(compile(source, '<v4-c4-local-one-gpu>', 'exec'), scope)
    return scope['partner_scores'], patches, local.text_sha(source), local.text_sha(loader)


def run_c4(stage=STAGE, b172=B172, competition=COMPETITION, dicom=DICOM,
           visible=None, out_root=OUT, min_free_gib=8., dry_run=False):
    stage, b172, out_root = Path(stage), Path(b172), Path(out_root)
    visible = Path(visible or stage / 'visible-b4')
    t0 = time.monotonic()
    ids, inputs, expected = _inputs(stage, competition, dicom, visible)
    identity = local.stage_identity(b172)
    need(identity['candidate'] == 'B172', 'B172 stage identity differs')
    code_manifest = local.read_json(stage / 'code/assets/release-manifest.json')
    c4_archive = stage / 'c4/dataset' / v4.C4_ARCHIVE
    need(local.sha(c4_archive) == code_manifest['c4_archive_sha256'] and
         local.sha(b172 / 'dataset/exp169-assets.zip.bin') == code_manifest['reused']['b172_archive_sha256'],
         'C4/B172 archive binding differs')
    out = out_root / 'c4-partner'
    need(not out.exists(), 'Fresh C4 reference required')
    with tempfile.TemporaryDirectory(prefix='exp169-v4-c4-') as td:
        tmp = Path(td)
        work = tmp / 'partner'
        shutil.copytree(b172 / 'assets', work)
        need(local.sha(work / 'release-manifest.json') == identity['manifest_sha256'],
             'B172 asset copy differs')
        c4 = tmp / 'c4'
        c4.mkdir()
        with zipfile.ZipFile(c4_archive) as archive:
            names = archive.namelist()
            need(len(names) == len(set(names)) and all(not Path(n).is_absolute() and
                 '..' not in Path(n).parts for n in names), 'Unsafe C4 archive member')
            archive.extractall(c4)
        need(local.sha(c4 / 'release-manifest.json') == code_manifest['c4_manifest_sha256'],
             'C4 extracted manifest differs')
        base, staged_v2, code = _load_b172_runtime(b172)
        _staged_code(code, local.read_json(work / 'release-manifest.json'))
        staged_v2.verify_assets(work, local.read_json(work / 'release-manifest.json'))
        # overlay_c4 authenticates B172 through v4.v2; bind it to the staged
        # wrapper before overlaying, not only when constructing the scorer.
        v4.v2 = staged_v2
        manifest = v4.overlay_c4(work, c4, code_manifest['c4_manifest_sha256'])
        scorer, patches, scorer_sha, loader_sha = _c4_local_function(base, staged_v2)
        if dry_run:
            return dict(status='DRY_RUN_OK', ids=ids, stage=identity,
                        scorer_sha256=scorer_sha, loader_sha256=loader_sha, patches=patches)
        import torch
        free = local.require_free_vram(torch, min_free_gib)
        view = local.competition_view(tmp / 'competition', competition, dicom)
        need(local.tree_digest(view / 'test_series') == inputs['dicom_tree'], 'DICOM copy differs')
        staged_v2.reset_backend(torch)
        backend = local.t4_effective_backend(torch)
        started = time.monotonic()
        scores, metrics = scorer(work, view, ids, manifest, started + 3600)
        seconds = time.monotonic() - started
        staged_modules = _staged_code(code, local.read_json(b172 / 'assets/release-manifest.json'),
                                      require_all_loaded=True)
        local.check_partner_arrays({'ids': np.asarray(ids), **scores}, n=3)
        raw = tmp / 'raw.npz'
        with raw.open('xb') as stream:
            np.savez(stream, ids=np.asarray(ids), **scores)
        out_root.mkdir(parents=True, exist_ok=True)
        return _publish(out, raw, ids, expected, 'C4', 'local-cuda-fp16', dict(
            visible_inputs_path=visible / _own_folder(visible) / 'inputs.json',
            b172_manifest_sha256=identity['manifest_sha256'],
            c4_manifest_sha256=code_manifest['c4_manifest_sha256'],
            c4_archive_sha256=code_manifest['c4_archive_sha256'],
            scorer_sha256=scorer_sha, loader_sha256=loader_sha,
            patches=patches, staged_modules=staged_modules,
            backend=backend, vram_before=free, metrics=metrics, input_source=inputs,
            inference_seconds=seconds, wall_seconds=time.monotonic()-t0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=('s43', 'c4'))
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--visible', type=Path)
    parser.add_argument('--out-root', type=Path, default=OUT)
    parser.add_argument('--min-free-gib', type=float, default=8.)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    result = (run_s43(args.stage, visible=args.visible, out_root=args.out_root, dry_run=args.dry_run)
              if args.arm == 's43' else run_c4(args.stage, visible=args.visible,
              out_root=args.out_root, min_free_gib=args.min_free_gib, dry_run=args.dry_run))
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
