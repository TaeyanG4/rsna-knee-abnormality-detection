"""v1.2.2 metadata crash boundaries. Frozen source files remain byte-identical.

Changelog 2026-10-13 v1.2.2: require bound worker status evidence; retain sticky
journal failures. Atomic commits and hard identity checks remain unchanged.
"""
from __future__ import annotations
import ast
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import threading
import io

VERSION = '1.2.2'


def atomic_write(path, data, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise ValueError('immutable artifact differs: ' + str(path))
        return len(data)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return len(data)


def write_once(path, data):
    return atomic_write(path, data, immutable=True)


def atomic_text(path, data, encoding='utf-8', errors=None, newline=None):
    # Explicit newline='' or '\n' preserves the original serializer bytes.
    return atomic_write(path, data.encode(encoding or 'utf-8', errors or 'strict'))


@contextmanager
def atomic_open(path, mode='w', encoding='utf-8', **kwargs):
    stream = io.StringIO()
    try:
        yield stream
        atomic_text(path, stream.getvalue(), encoding)
    finally:
        stream.close()


def atomic_json(value, path, **kwargs):
    return atomic_text(path, json.dumps(value, **kwargs))


class AtomicWrites(ast.NodeTransformer):
    """Only I/O calls change; model/preprocessing expressions are retained."""
    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'json' and node.func.attr == 'dump'
                and len(node.args) == 2 and isinstance(node.args[1], ast.Call)
                and isinstance(node.args[1].func, ast.Name) and node.args[1].func.id == 'open'):
            return ast.copy_location(ast.Call(ast.Name('_metadata_json', ast.Load()),
                [node.args[0], node.args[1].args[0]], node.keywords), node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('write_text', 'write_bytes'):
            name = '_metadata_text' if node.func.attr == 'write_text' else '_metadata_bytes'
            return ast.copy_location(ast.Call(ast.Name(name, ast.Load()), [node.func.value, *node.args], node.keywords), node)
        return node

    def visit_With(self, node):
        if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == 'dump'
               for statement in node.body for x in ast.walk(statement)):
            for item in node.items:
                call = item.context_expr
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'open'
                        and len(call.args) > 1 and isinstance(call.args[1], ast.Constant) and call.args[1].value == 'w'):
                    call.func.id = '_metadata_open'
        return self.generic_visit(node)


def atomic_source(source):
    tree = AtomicWrites().visit(ast.parse(source))
    imports = ast.parse('from release_metadata_v122 import atomic_text as _metadata_text, atomic_write as _metadata_bytes, atomic_json as _metadata_json, atomic_open as _metadata_open').body
    pos = 1 if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else 0
    while pos < len(tree.body) and isinstance(tree.body[pos], ast.ImportFrom) and tree.body[pos].module == '__future__':
        pos += 1
    tree.body[pos:pos] = imports
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def journal_text(directory):
    directory = Path(directory)
    rows = directory / 'journal-rows'
    if rows.exists():
        try:
            records = [p.read_text(encoding='utf-8') for p in sorted(rows.glob('*.json'))]
            for record in records:
                if not isinstance(json.loads(record), dict):
                    raise MetadataUnavailable('journal record shape')
            return '\n'.join(record.rstrip('\r\n') for record in records) + ('\n' if records else '')
        except (OSError, UnicodeError, ValueError) as exc:
            raise MetadataUnavailable('journal record unavailable') from exc
    path = directory / 'predictions.jsonl'
    return path.read_text(encoding='utf-8') if path.exists() else ''


class MetadataUnavailable(Exception):
    """Missing, truncated or structurally malformed metadata; not a pin mismatch."""


def metadata(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError) as exc:
        raise MetadataUnavailable(str(path)) from exc
    if path.name == 'ids.json':
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise MetadataUnavailable(str(path))
    elif not isinstance(value, dict) or not value:
        raise MetadataUnavailable(str(path))
    return value


def shaped(value, exemplar):
    """Missing fields/types are interrupted metadata; present wrong values remain hard."""
    if type(value) is not type(exemplar):
        raise MetadataUnavailable('metadata type')
    if isinstance(exemplar, dict):
        for key, child in exemplar.items():
            if key not in value:
                raise MetadataUnavailable('metadata missing field: ' + key)
            shaped(value[key], child)
    return value


def optional_json(path):
    try:
        value = metadata(path)
        for key in ('fallback', 'failures', 'files', 'worker_exit_codes', 'fallback_reasons', 'provenance'):
            if key in value and not isinstance(value[key], dict):
                return {}
        if 'workers' in value and (not isinstance(value['workers'], list) or
                                  not all(isinstance(w, dict) for w in value['workers'])):
            return {}
        return value
    except MetadataUnavailable:
        return {}


def telemetry(checks, name, check):
    try:
        value = check()
        checks[name] = True if value is None else value
    except (OSError, ValueError, KeyError, TypeError, AttributeError, AssertionError, MetadataUnavailable) as exc:
        checks[name] = f'{type(exc).__name__}: {exc}'


def branch_metadata(directory, names):
    """Optional receipts may be absent during inference, but never trusted when corrupt."""
    for name in names:
        path = Path(directory) / name
        if path.exists():
            value = metadata(path)
            for key in ('fallback', 'failures', 'files', 'worker_exit_codes'):
                if key in value and not isinstance(value[key], dict):
                    raise MetadataUnavailable('malformed ' + str(path))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_binding(worker):
    """Bind final status to the assigned cohort, output bytes and source identity."""
    worker = Path(worker)
    return {name: digest(worker / relative) for name, relative in (
        ('assignment', 'ids.json'), ('predictions', 'predictions.csv'),
        ('provenance', '../../provenance.json'))}


# Only these two immutable recorded S6 workers are allowed without a v122 binding.
# Pins cover receipt, assignment AND predictions; enabled only for the explicit
# historical arithmetic stand-in, never for actual R7 inference.
LEGACY_WORKERS = {
    ('cb8c3f7661c0d3c6a1b58075dc2027ddfe9d016506829c59182c0dbba165ffaa',
     'c483108be0d7e35a05aa71c3f7d720c85059b7f258b7f016f5aec5e25df6039d',
     '5d84c8a8f6c66fc9afb63629ac53fce77b887ec9ad55eb94d68076acd63220ce'),
    ('253114c3dd245b22b86eb7e52db33eee4b5d75990a8526c7301009f6c316d797',
     'a202c53ef5a0b4f3c0f8754e901b20ecdb5d2e2f0c7262de062481f71ca0f197',
     'd218bea43fd83f0078e9e27d592a26d2554bd517d90cbfcbb8dfed11a4e8ada6'),
}


def worker_status(worker, part, *, legacy=False):
    """No absent field means success. A committed stale binding remains fatal."""
    value = shaped(metadata(worker / 'receipt.json'),
                   {'status': '', 'studies': 0, 'scored': 0, 'fallback': {}})
    if (value['status'] != 'PASS' or value['studies'] != len(part)
            or value['scored'] != len(part) - len(value['fallback'])
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in value['fallback'].items())):
        raise MetadataUnavailable('incomplete worker status')
    if not set(value['fallback']) <= set(part):
        raise ValueError('worker receipt fallback cohort mismatch')
    if 'publication_binding' in value:
        expected = worker_binding(worker)
        shaped(value['publication_binding'], expected)
        if value['publication_binding'] != expected:
            raise ValueError('worker receipt committed identity/output mismatch')
    elif not (legacy and tuple(digest(worker / name) for name in
                  ('receipt.json', 'ids.json', 'predictions.csv')) in LEGACY_WORKERS):
        raise MetadataUnavailable('worker receipt lacks publication binding')
    return value


def read_journal(m, directory, ids, members, expected_identity):
    import numpy as np
    identity = shaped(metadata(directory / 'journal-identity.json'), expected_identity)
    m.need(identity == expected_identity, 'journal checkpoint/source identity')
    assigned = m.v5q.validate_ids(metadata(directory / 'ids.json'))
    m.need(set(assigned) == set(ids), 'journal cohort')
    receipt = optional_json(directory / 'receipt.json')
    if 'journal_identity' in receipt:
        shaped(receipt['journal_identity'], expected_identity)
        m.need(receipt['journal_identity'] == expected_identity, 'journal receipt identity')
    if 'assignment' in receipt:
        m.need(receipt['assignment'] == assigned, 'journal receipt assignment')
    values = {name: np.full((len(ids), 12), .5) for name in members}
    failures = {name: dict.fromkeys(ids, 'MissingPrediction: child did not finish study') for name in members}
    failed = {name: set() for name in members}
    for line in journal_text(directory).splitlines():
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise MetadataUnavailable('corrupt journal record') from exc
        if (not isinstance(row, dict) or not isinstance(row.get('uid'), str)
                or not isinstance(row.get('member'), str) or not {'values', 'error'} & row.keys()):
            raise MetadataUnavailable('journal record shape')
        uid, member = row['uid'], row['member']
        m.need(uid in ids and member in members, 'journal UID/member')
        # A later success row cannot erase an explicit failure (including rows
        # containing both values and error). MissingPrediction alone is provisional.
        if 'error' in row:
            failed[member].add(uid)
            failures[member][uid] = str(row['error'])
            values[member][ids.index(uid)] = .5
            continue
        if 'values' in row:
            try:
                value = m.p.RBinding.prediction(row['values'])
            except (ValueError, TypeError):
                failed[member].add(uid)
                failures[member][uid] = 'InvalidPrediction'
                values[member][ids.index(uid)] = .5
            else:
                if uid not in failed[member]:
                    values[member][ids.index(uid)] = value
                    failures[member].pop(uid, None)
        else:
            failures[member][uid] = str(row['error'])
    return values, failures


@contextmanager
def writers(m):
    """Scoped adapters in root/children; never mutate frozen files or another release."""
    if getattr(m, '_metadata_writers_active', False):
        yield
        return
    m._metadata_writers_active = True
    saved = []
    absent = object()
    def set_(obj, name, value):
        saved.append((obj, name, getattr(obj, name, absent)))
        setattr(obj, name, value)
    try:
        for module in (m, m.old, m.base, m.c85, m.p):
            set_(module, 'write_once', write_once)
            if hasattr(module, 'telemetry'):
                set_(module, 'telemetry', telemetry)
            if hasattr(module, 'optional_json'):
                set_(module, 'optional_json', optional_json)
        original_codec = m.old.codec_receipt
        def codec_receipt(path):
            rec = original_codec(path)
            if rec.get('status') == 'ACTIVE' and (not isinstance(rec.get('hook_dir'), str)
                    or not isinstance(rec.get('entry'), str)):
                return {'status': 'INACTIVE_CORRUPT_METADATA'}
            return rec
        set_(m.old, 'codec_receipt', codec_receipt)
        hook_source = m.codecs.hook_source
        def synced_hook(*args, **kwargs):
            source = hook_source(*args, **kwargs)
            anchor = 'json.dump(record, handle, sort_keys=True)'
            m.need(source.count(anchor) == 1, 'codec metadata boundary drift')
            return source.replace(anchor, anchor + '\n                handle.flush()\n                _hb126_os.fsync(handle.fileno())')
        set_(m.codecs, 'hook_source', synced_hook)
        p = m.p
        class Journal:
            def __init__(self, directory, ids, identity):
                self.directory, self.lock = Path(directory), threading.Lock()
                write_once(self.directory / 'ids.json', p.json_bytes(ids))
                write_once(self.directory / 'journal-identity.json', p.json_bytes(identity))
                (self.directory / 'journal-rows').mkdir(exist_ok=True)
                self.sequence = 0
            def record(self, uid, member, values=None, error=None, seconds=None):
                import numpy as np
                row = dict(uid=uid, member=member)
                if error is None:
                    row['values'] = np.asarray(p.RBinding.prediction(values)).tolist()
                else:
                    row['error'] = str(error)
                if seconds is not None:
                    row['seconds'] = seconds
                with self.lock:
                    self.sequence += 1
                    write_once(self.directory / 'journal-rows' / f'{self.sequence:09d}.json',
                               (json.dumps(row, allow_nan=False) + '\n').encode())
                return values
        set_(m.c85, 'Journal', Journal)
        def finish_receipt(binding, work, seconds):
            write_once(binding.work / 'receipt.json', p.json_bytes(dict(
                status='RECORDED', seconds=seconds, members=binding.members,
                failures=binding.failures,
                journal_identity=metadata(binding.work / 'journal-identity.json'),
                assignment=metadata(binding.work / 'ids.json'))))
        set_(m.c85.ServingBinding, 'finish_receipt', finish_receipt)
        set_(m.r7, '_metadata_text', atomic_text)
        set_(m.r7, '_metadata_bytes', atomic_write)
        for name in ('diagnostic_source', 'write_provenance', 'infer_studies'):
            scope = dict(m.r7.__dict__, _metadata_text=atomic_text, _metadata_bytes=atomic_write)
            tree = AtomicWrites().visit(ast.parse(inspect.getsource(getattr(m.r7, name))))
            exec(compile(ast.fix_missing_locations(tree), '<v122-' + name + '>', 'exec'), scope)
            fn = scope[name]
            # Keep original globals live so infer_studies sees the patched helpers.
            import types
            globals_ = m.r7.__dict__
            set_(m.r7, name, types.FunctionType(fn.__code__, globals_, name, fn.__defaults__, fn.__closure__))
            getattr(m.r7, name).__kwdefaults__ = fn.__kwdefaults__
        original = m.r7.diagnostic_source
        def diagnostic_source(*a, **kw):
            runtime = original(*a, **kw)
            for name in ('probe_dual.py', 'compact_infer.py', 'px_cache.py', 'knee_anchor_bank_v2.py'):
                path = runtime / name
                if not path.exists():
                    continue
                source = path.read_text(encoding='utf-8')
                if name == 'compact_infer.py':
                    anchor = '    (out / "receipt.json").write_text('
                    m.need(source.count(anchor) == 1, 'worker receipt boundary drift')
                    source = source.replace(anchor,
                        '    from release_metadata_v122 import worker_binding\n'
                        '    receipt["publication_binding"] = worker_binding(out)\n' + anchor)
                atomic_text(path, atomic_source(source))
            atomic_text(runtime / 'runtime_sha256.json', json.dumps(
                {p.name: m.s6.sha256(p) for p in sorted(runtime.glob('*.py'))}, indent=2) + '\n')
            return runtime
        set_(m.r7, 'diagnostic_source', diagnostic_source)
        yield
    finally:
        for obj, name, value in reversed(saved):
            if value is absent:
                delattr(obj, name)
            else:
                setattr(obj, name, value)
        m._metadata_writers_active = False


def guarded(source, start, end, fallback):
    begin, finish = source.index(start), source.index(end, source.index(start))
    body = source[begin:finish]
    return source[:begin] + '    try:\n' + ''.join('    ' + line if line.strip() else line for line in body.splitlines(True)) + \
        '    except (MetadataUnavailable, OSError, UnicodeError, KeyError, TypeError, AttributeError):\n' + ''.join('        ' + line + '\n' for line in fallback.splitlines()) + source[finish:]


def collectors(m, cohort):
    """Compile only guarded metadata boundaries around the original authenticated math."""
    import numpy as np
    c, b = m.c85, m.base
    scope = dict(c.__dict__, read_json=metadata, optional_json=optional_json, telemetry=telemetry,
                 MetadataUnavailable=MetadataUnavailable, shaped=shaped, cohort=cohort,
                 journal_text=journal_text, branch_metadata=branch_metadata)
    def manifest(path):
        value = shaped(metadata(path), {'candidate': '', 'generator_version': '', 'entries': [],
                                        'anchor_binding': {'checkpoint_sha256': {m.p.R_NAME: ''}}})
        if not value['entries']:
            raise MetadataUnavailable('manifest roster missing')
        for entry in value['entries']:
            shaped(entry, {'family': '', 'seed': 0, 'fold': 0})
        return c.manifest(path)
    scope['manifest'] = manifest
    scope['read_journal'] = lambda *a, **kw: read_journal(m, *a, **kw)
    source = inspect.getsource(c.authenticated_a172)
    source = source.replace("read_json(trio / 'binding.json') == m['anchor_binding']", "shaped(read_json(trio / 'binding.json'), m['anchor_binding']) == m['anchor_binding']")
    source = source.replace("    elif (trio / 'raw.npz').exists():",
        "    elif (trio / 'raw.npz').exists():\n"
        "        shaped(read_json(trio / 'receipt.json'), {'status': '', 'files': {'raw.npz': ''}})\n"
        "        if tr['status'] != 'COMPLETE_INFERENCE': raise MetadataUnavailable('own raw status')\n"
        "        p.check_pin(trio / 'raw.npz', tr['files']['raw.npz'])")
    source = source.replace("m = manifest(mpath) if mpath.exists() else None", "try:\n        m = manifest(mpath) if mpath.exists() else None\n    except MetadataUnavailable:\n        m = None")
    source = source.replace("need(m is not None, 'own manifest identity missing')", "if m is None: raise MetadataUnavailable('own manifest identity missing')")
    source = source.replace("need(m is not None, 'Q manifest identity missing')", "if m is None: raise MetadataUnavailable('Q manifest identity missing')")
    source = source.replace("ids = read_json(directory / 'ids.json') if (directory / 'ids.json').exists() else []", "ids = cohort")
    source = source.replace("    if (trio / 'journal-identity.json').exists():", "    branch_metadata(trio, ['receipt.json', 'binding.json'])\n    if (trio / 'journal-identity.json').exists() or (trio / 'journal-rows').exists():")
    source = guarded(source, "    branch_metadata(trio", '    q = np.full',
        "ids = cohort\nvalues = {name: np.full((len(ids), 12), .5) for name in ['EXP143v2', 'EXP155']}\nfailures = {name: dict.fromkeys(ids, 'MetadataUnavailable: own') for name in values}")
    source = source.replace("    q = np.full", "    need(set(ids) == set(cohort), 'own cohort mismatch')\n    q = np.full")
    source = source.replace("    if rawpath.exists()", "    branch_metadata(directory, ['exp169-executed-candidate.json'])\n    if rawpath.exists()")
    source = source.replace("not (directory / 'q-journal/journal-identity.json').exists()", "not (directory / 'q-journal').exists()")
    source = source.replace("elif (directory / 'q-journal/journal-identity.json').exists():", "elif (directory / 'q-journal').exists():")
    source = guarded(source, '    branch_metadata(directory', '    if arrays is not None:', 'arrays = None')
    exec(compile(source, '<v122-a172>', 'exec'), scope)
    a = scope['authenticated_a172']
    from types import SimpleNamespace
    ps = dict(b.__dict__, read_json=metadata, optional_json=optional_json, cohort=cohort, telemetry=telemetry,
              MetadataUnavailable=MetadataUnavailable, shaped=shaped, branch_metadata=branch_metadata,
              worker_status=worker_status, legacy_stand_in=m.base.STAND_IN,
              c85=SimpleNamespace(**dict(c.__dict__, read_journal=scope['read_journal'])))
    source = inspect.getsource(b.authenticated_pr7)
    source = source.replace('except (ValueError, KeyError, TypeError) as exc:', 'except (ValueError, KeyError, TypeError, AttributeError) as exc:')
    begin, end = source.index('    if (directory / "ids.json")'), source.index('    r, ours =')
    source = source[:begin] + '    ids = cohort\n' + source[end:]
    source = source.replace('    if (work / "r/journal-identity.json").exists():', '    branch_metadata(work / "r", ["receipt.json"])\n    if (work / "r/journal-identity.json").exists() or (work / "r/journal-rows").exists():')
    source = source.replace('        need(rr.get("status") == "PASS_R", "R raw output lacks its receipt")',
        '        shaped(rr, {"status": "", "raw_sha256": ""})\n'
        '        if rr["status"] != "PASS_R": raise MetadataUnavailable("R raw status")')
    source = source.replace('        p.check_pin(work / "r/raw.npz", rr["raw_sha256"])',
        '        shaped(rr, {"failure_count": 0})\n'
        '        if rr["failure_count"] != len(rr.get("failures", {})):\n'
        '            raise MetadataUnavailable("R raw failure status incomplete")\n'
        '        p.check_pin(work / "r/raw.npz", rr["raw_sha256"])')
    source = guarded(source, '    branch_metadata(work / "r"', '    rf.update(',
                     "r = np.full((len(ids), 12), .5)\nrf = dict.fromkeys(ids, 'MetadataUnavailable: R')")
    source = source.replace('        for key, value in [("dataset_slug"',
        '        shaped(expected, {"manifest_sha256": "", "file_sha256": {}, "bundle_id": "", "dataset_slug": ""})\n        shaped(prov, {"dataset_slug": "", "bundle_id": "", "source_sha256": {}, "anchor_weights_sha256": "", "manifest_sha256": "", "file_sha256": {}})\n        for key, value in [("dataset_slug"')
    source = source.replace('    if prov_path.exists() or workers:', '    branch_metadata(work / "s_r7", ["identity.json", "invocation.json", "r7_runtime/runtime_sha256.json", "controller/submission_receipt.json"])\n    if prov_path.exists() or workers:')
    source = source.replace('            wr = optional_json(worker / "receipt.json")',
        '            try:\n'
        '                wr = worker_status(worker, part, legacy=prov.get("stand_in") == legacy_stand_in)\n'
        '            except (MetadataUnavailable, OSError):\n'
        '                sf.update(dict.fromkeys(part, "MetadataUnavailable: worker status"))\n'
        '                continue')
    source = guarded(source, '    branch_metadata(work / "s_r7"', '    telemetry(checks, "controller_crosscheck"',
        "ours = np.full((len(ids), 12), .5)\nsf = dict.fromkeys(ids, 'MetadataUnavailable: S_R7')\nidentity = {}")
    exec(compile(source, '<v122-pr7>', 'exec'), ps)
    return a, ps['authenticated_pr7']
