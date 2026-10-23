"""CPU-only HB151 fixtures using the unchanged own25q3 fast path; exact CPU model loading (no inference)."""
from __future__ import annotations
import ast
import contextlib
import csv
import io
import os
from pathlib import Path
from unittest.mock import patch

from tools import hb151_killtest_v101 as b
from tools.c85r7_own25q2_v1114 import VISIBLE_IDS
from tools.c85r7_own25q_v1111 import HEADER

MARKERS = [None, '', '0', 'False', '1', 'false', 'true', ' ', '\n']
COHORTS = [[], VISIBLE_IDS[:2], VISIBLE_IDS, ['other-a', 'other-b', 'other-c'], VISIBLE_IDS + ['fourth']]
FAULTS = ['open', 'enter', 'header', 'rows', 'close', 'unlink-transient', 'unlink-persistent', 'print']
BAD_MOUNTS = ['absent', 'ambiguous', 'missing-series', 'missing-manifest', 'bad-override', 'list-error', 'read-error']
BAD_CSV = ['wrong_header\nfoo\n', 'StudyInstanceUID,StudyInstanceUID\nfoo,foo\n',
           'StudyInstanceUID\n"unclosed', '', 'StudyInstanceUID\n\n', '\ufffd\n']


def run_case(area, *, ids=None, manifest_ids=None, marker=None, layout='flat',
             raw_manifest=None, malformation=None, fault=None, preexisting=False,
             real_manifest=False):
    """Execute actual cell zero and actual guard; stop after original import statement."""
    ids = VISIBLE_IDS if ids is None else ids
    manifest_ids = VISIBLE_IDS if manifest_ids is None else manifest_ids
    root = area / 'input'
    root.mkdir(parents=True, exist_ok=True)
    middle = {'flat':'', 'nested':'competitions', 'dataset':'datasets/owner', 'override':'override'}[layout]
    comp = root / middle / 'rsna-knee-abnormality-detection'
    series = comp / 'test_series'
    if malformation not in ['absent', 'bad-override']:
        comp.mkdir(parents=True, exist_ok=True)
        if malformation != 'missing-series':
            series.mkdir(exist_ok=True)
            for uid in ids:
                (series / uid).mkdir(exist_ok=True)
            (series / 'ignored-file').write_text('not a study', encoding='utf-8')
        if malformation != 'missing-manifest':
            if real_manifest:
                (comp / 'test.csv').write_bytes((b.ROOT / 'storage/competition/test.csv').read_bytes())
            elif raw_manifest is not None:
                (comp / 'test.csv').write_text(raw_manifest, encoding='utf-8')
            else:
                with (comp / 'test.csv').open('w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['StudyInstanceUID'])
                    w.writerows([[uid] for uid in manifest_ids])
    if malformation == 'ambiguous':
        other = root / 'competitions/rsna-knee-abnormality-detection'
        other.mkdir(parents=True, exist_ok=True)
        (other / 'test.csv').write_bytes((comp / 'test.csv').read_bytes())
    work = area / 'working'
    work.mkdir(exist_ok=True)
    output = work / 'submission.csv'
    if preexisting:
        output.write_bytes(b'existing submission must survive')
    ns = {}
    log = io.StringIO()
    unlink_calls = 0
    actual_open, actual_unlink, actual_iterdir, actual_writer = Path.open, Path.unlink, Path.iterdir, csv.writer

    class Stream:
        def __init__(self, stream):
            self.stream = stream
        def write(self, value):
            return self.stream.write(value)
        def __enter__(self):
            if fault == 'enter':
                self.stream.close()
                raise RuntimeError('injected enter failure')
            return self.stream
        def __exit__(self, *args):
            assert ns['FAST'] is False, 'FAST committed before close'
            self.stream.close()
            if fault == 'close':
                raise OSError('injected close failure')

    class Writer:
        def __init__(self, stream):
            self.stream = stream
            self.writer = actual_writer(stream)
        def writerow(self, row):
            assert ns['FAST'] is False, 'FAST committed before writing'
            if fault == 'header':
                self.stream.write('partial')
                raise RuntimeError('injected header failure')
            return self.writer.writerow(row)
        def writerows(self, rows):
            assert ns['FAST'] is False, 'FAST committed before writing'
            if fault in ['rows', 'unlink-transient', 'unlink-persistent']:
                self.writer.writerow(next(iter(rows)))
                raise OSError('injected rows failure')
            return self.writer.writerows(rows)

    def mapped(value, *parts):
        if str(value) == '/kaggle/input':
            return root
        if str(value) == '/kaggle/working/submission.csv':
            return output
        return Path(value, *parts)

    def opened(path, *args, **kwargs):
        if path == comp / 'test.csv' and malformation == 'read-error':
            raise PermissionError('injected manifest read failure')
        if path == output:
            assert ns['FAST'] is False, 'FAST committed before open'
            if fault == 'open':
                raise PermissionError('injected open failure')
            return Stream(actual_open(path, *args, **kwargs))
        return actual_open(path, *args, **kwargs)

    def unlinked(path, *args, **kwargs):
        nonlocal unlink_calls
        if path == output:
            unlink_calls += 1
            if fault == 'unlink-persistent' or (fault == 'unlink-transient' and unlink_calls <= 2):
                raise PermissionError('injected sharing violation')
        return actual_unlink(path, *args, **kwargs)

    def listed(path):
        if path == series and malformation == 'list-error':
            raise RuntimeError('injected directory enumeration failure')
        return actual_iterdir(path)

    def printed(*args, **kwargs):
        if fault == 'print':
            raise BrokenPipeError('injected print failure')
        print(*args, **kwargs)

    with patch.dict(os.environ, {}, clear=False), patch('pathlib.Path', mapped), \
         patch.object(Path, 'open', opened), patch.object(Path, 'unlink', unlinked), \
         patch.object(Path, 'iterdir', listed), patch('csv.writer', Writer), \
         patch('time.sleep'), contextlib.redirect_stdout(log):
        os.environ.pop('KAGGLE_IS_COMPETITION_RERUN', None)
        os.environ.pop('C85_COMPETITION', None)
        if marker is not None:
            os.environ['KAGGLE_IS_COMPETITION_RERUN'] = marker
        if layout == 'override' or malformation in ['missing-manifest', 'bad-override']:
            os.environ['C85_COMPETITION'] = str(comp)
        ns['print'] = printed
        # Deliberately no exception catcher: any escaped exception fails this case.
        exec(''.join(b.read(b.OUT / 'kernel/release.ipynb')['cells'][0]['source']), ns)
        first_reached = False
        for cell in b.read(b.OUT / 'kernel/release.ipynb')['cells'][1:]:
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']))
            assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If)
            tree.body[0].body = tree.body[0].body[:1]
            exec(compile(tree, '<original-first-statement-only>', 'exec'), ns)
            first_reached = 'Path' in ns
        assert first_reached == (not ns['FAST'])
        assert 'BLOB' not in ns and 'exp169_release_c85r7codec' not in ns
    expected = (not marker and sorted(ids) == VISIBLE_IDS and
                len(manifest_ids) == 3 and set(manifest_ids) == set(VISIBLE_IDS) and
                raw_manifest is None and malformation is None and fault is None and not preexisting)
    assert ns['FAST'] is bool(expected), (str(area), ns['FAST'], expected, log.getvalue())
    if ns['FAST']:
        with output.open(newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        assert rows == [HEADER] + [[uid] + ['0.5'] * 12 for uid in VISIBLE_IDS]
        assert 'FAST_PATH_VISIBLE' in log.getvalue()
    elif preexisting:
        assert output.read_bytes() == b'existing submission must survive'
    elif fault == 'unlink-persistent':
        assert output.exists() and unlink_calls == 40
        assert 'cleanup=PermissionError' in log.getvalue()
    else:
        assert not output.exists(), 'partial output was not removed'
    if not ns['FAST'] and fault != 'print':
        assert 'FAST_PATH_DISABLED' in log.getvalue()
    return dict(status='PASS', FAST=ns['FAST'], error=None, original_first_statement=first_reached,
                output_exists=output.exists(), unlink_calls=unlink_calls,
                diagnostic=log.getvalue().strip())


def case_specs():
    cases = []
    for m, marker in enumerate(MARKERS):
        for c, ids in enumerate(COHORTS):
            cases.append((f'm{m}-c{c}', dict(ids=ids, marker=marker, real_manifest=True)))
    cases += [(name, dict(layout=name, real_manifest=True)) for name in ['nested', 'dataset', 'override']]
    cases += [(f'mount-{name}', dict(malformation=name, real_manifest=True)) for name in BAD_MOUNTS]
    cases += [(f'csv-{i}', dict(raw_manifest=raw)) for i, raw in enumerate(BAD_CSV)]
    cases += [(f'manifest-{i}', dict(manifest_ids=ids)) for i, ids in enumerate([
        [], VISIBLE_IDS[:2], ['a','b','c'], [VISIBLE_IDS[0]] * 3, VISIBLE_IDS + ['fourth']])]
    cases += [(f'fault-{name}', dict(fault=name, real_manifest=True)) for name in FAULTS]
    cases += [('existing-visible', dict(preexisting=True, real_manifest=True)),
              ('existing-malformed', dict(preexisting=True, raw_manifest='StudyInstanceUID\n"unclosed')),
              ('reversed', dict(ids=list(reversed(VISIBLE_IDS)), manifest_ids=list(reversed(VISIBLE_IDS))))]
    return cases


def run_smoke(area):
    import shutil
    # Fixed fixture directories only; never remove input/raw data or stage artifacts.
    assert area.resolve().is_relative_to((b.SCRATCH / 'fast-cases').resolve())
    for name, _ in case_specs():
        fixture = (area / name).resolve()
        assert fixture.is_relative_to(area.resolve())
        if fixture.exists():
            shutil.rmtree(fixture)
    results = [dict(case=name, **run_case(area / name, **kwargs)) for name, kwargs in case_specs()]
    return dict(status='PASS', cases=results,
                limitation='Local real test.csv copied unchanged; authenticated real visible UIDs in synthetic directory fixtures (local test_series absent). No DICOM/GPU inference. Persistent OS denial of deletion is reported, bounded, nonfatal; removal cannot be guaranteed while the filesystem forbids it.')


def loader_smoke(area):
    """Execute actual guarded extraction/hash/verify_assets; stop before serving exec."""
    import builtins
    import shutil
    import sys
    assert area.resolve().is_relative_to(b.SCRATCH.resolve())
    if area.exists():
        shutil.rmtree(area)
    area.mkdir(parents=True)
    tree = ast.parse(b.source(b.read(b.OUT/'kernel/release.ipynb')['cells'][5]))
    body = tree.body[0].body
    stop = next(i for i,n in enumerate(body) if isinstance(n, ast.Expr) and
                isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'exec')
    tree.body[0].body = body[:stop]
    original_import = builtins.__import__
    frozen = None
    def imported(name, *args, **kwargs):
        if name == 'exp152_own0949_release':
            import importlib.util
            spec = importlib.util.spec_from_file_location('_hb151_extracted', area/'exp152-assets'/b.RUNTIME_NAME)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return original_import(name, *args, **kwargs)
    def mapped(value, *parts):
        if str(value) in ['/kaggle/input/'+b.DATASET.split('/')[1], '/kaggle/input/datasets/'+b.DATASET]:
            return b.OUT/'dataset'
        if str(value) == '/kaggle/working/exp152-assets':
            return area/'exp152-assets'
        if str(value) == '/kaggle/working':
            return area
        return Path(value, *parts)
    ns = {'FAST':False}
    paths = sys.path[:]
    try:
        with patch('pathlib.Path', mapped), patch('builtins.__import__', imported):
            exec(compile(tree, '<actual-parent-through-verify-assets>', 'exec'), ns)
    finally:
        sys.path[:] = paths
    assert ns['_files'][b.NAME].is_file()
    assert b.sha(ns['_files'][b.NAME]) == b.read(b.OUT/'build-receipt.json')['checkpoint_sha256']
    assert 'main' not in ns and not (area/'submission.csv').exists()
    exact = exact_model_loader(ns['_e152'], ns['_files'][b.NAME], ns['_manifest'])
    return dict(status='PASS', exact_model_loader=exact, archive_extract_and_verify_assets=True,
                original_import_launch_unchanged=True, stopped_before_serving_exec=True,
                checkpoint_is_placeholder=b.read(b.OUT/'build-receipt.json')['placeholder'])


def placeholder_export():
    """Completed CPU fixture, never read the running fit output; marked non-release."""
    import torch
    torch.set_num_threads(2)
    area = b.SCRATCH/'placeholder-v101'
    area.mkdir(parents=True, exist_ok=True)
    protocol = area/'protocol.json'
    b.write(protocol, b.encoded(dict(run_dirs=dict(local=b.RUN_ID), fixture='CPU_ONLY_NOT_TRAINED')))
    ck = b.runtime().safe_load(b.ROOT/'.local/scratch/astra-b4/H-B-151/v3/cpu-checkpoint.pt')
    ck.update(exposure='EXPOSED', clean_oof=False, hb151_killtest_placeholder=True)
    ck['provenance'] = dict(protocol_sha256=b.sha(protocol), run_dir=b.RUN_ID, seed=42)
    checkpoint = area/b.NAME
    # Reuse identical completed fixture bytes on reruns, without trusting arbitrary existing bytes.
    temporary = checkpoint.with_suffix('.partial')
    torch.save(ck, temporary)
    b.retry(lambda:b.os.replace(temporary, checkpoint))
    csha=b.sha(checkpoint)
    identity=dict(sha256=csha,protocol_sha256=b.sha(protocol),run_dir=b.RUN_ID,
                  exposure='EXPOSED',clean_oof=False,serving_name=b.NAME,placeholder=True)
    b.write(checkpoint.with_suffix('.sha256'),(csha+'  '+b.NAME+'\n').encode())
    b.write(checkpoint.with_suffix('.json'),b.encoded({k:v for k,v in identity.items() if k != 'run_dir'}))
    b.write(area/'summary.json',b.encoded(dict(status='COMPLETE_TRAINING_NO_EVALUATION',
        identity=ck['provenance'],model_sha256=csha,exposure='EXPOSED',clean_oof=False)))
    # Hardlink the synthetic canonical checkpoint; no additional 300 MB copy.
    canonical=area/'model_best.pt'
    if canonical.exists(): b.retry(canonical.unlink)
    b.os.link(checkpoint,canonical)
    return checkpoint,identity


def exact_model_loader(runtime, checkpoint, manifest):
    """Execute unchanged serving load_model -> staged checked_model on real CoAtNet CPU."""
    import copy
    import torch
    import timm
    torch.set_num_threads(2)
    tree=ast.parse((b.PARENT/'assets/public-serving.py').read_text(encoding='utf-8'))
    names={'build_backbone','RaptorClassifier','load_model'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==3
    ns=dict(torch=torch,nn=torch.nn,timm=timm,INPUT_RES=384,_E152=runtime,
            _E152_FILES={b.NAME:checkpoint},_E152_MANIFEST=copy.deepcopy(manifest))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<unchanged-serving-model-definitions>','exec'),ns)
    ns['_E152_MANIFEST']['checkpoints'][b.NAME]['sha256']='0'*64
    try:
        ns['load_model'](checkpoint,runtime.ARCH,384,torch.device('cpu'))
    except ValueError as exc:
        assert 'Bound checkpoint SHA mismatch' in str(exc)
    else: raise AssertionError('wrong SHA accepted')
    ns['_E152_MANIFEST']=copy.deepcopy(manifest)
    for key in ['protocol_sha256','run_dir']:
        bad=copy.deepcopy(manifest)
        if key=='protocol_sha256': bad[key]='0'*64
        else: bad['hb151_export'][key]='old/S42'
        ns['_E152_MANIFEST']=bad
        try: ns['load_model'](checkpoint,runtime.ARCH,384,torch.device('cpu'))
        except ValueError as exc:
            assert ('protocol SHA mismatch' if key=='protocol_sha256' else 'run mismatch') in str(exc)
        else: raise AssertionError('wrong provenance accepted')
    ns['_E152_MANIFEST']=copy.deepcopy(manifest)
    model,res=ns['load_model'](checkpoint,runtime.ARCH,384,torch.device('cpu'))
    assert res==320 and not model.training and next(model.parameters()).device.type=='cpu'
    assert runtime.state_schema(model.state_dict())==manifest['own_state_schema']
    count=sum(p.numel() for p in model.parameters())
    del model
    assert not torch.cuda.is_initialized()
    return dict(status='PASS', wrong_sha='REJECT',right_sha='ACCEPT',wrong_protocol='REJECT',
                wrong_run_dir='REJECT',strict_state_load=True,parameters=count,device='cpu',
                path='public-serving.load_model -> extracted exp152_own0949_release.checked_model',
                model_forward=False)


def fixture_bind_smoke():
    """Run bind/validate and reject final binding of a synthetic finished export."""
    from tools import hb151_build_v3 as fit
    checkpoint,identity=placeholder_export()
    out=b.SCRATCH/'bound-fixture-v101'
    with patch.object(fit,'PROTOCOL',checkpoint.parent/'protocol.json'), \
         patch.object(fit,'RUN',checkpoint.parent),patch.object(b,'OUT',out):
        result=b.bind(checkpoint,fixture=True)
        assert result['status']=='PASS_CPU_SCOPE' and result['placeholder']
        assert b.validate()['status']=='PASS_CPU_SCOPE'
        try: b.bind(checkpoint)
        except ValueError as exc: assert 'placeholder' in str(exc)
        else: raise AssertionError('fixture admitted as real')
    return dict(status='PASS',completed_export_bind='PASS',validate='PASS',final_placeholder='REJECT')
