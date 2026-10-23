"""Diagnostic-only quarantine of prediction directories (EXP116 / H-B-46, D-B-024), CPU only, standard library only.

A directory whose predictions must never become decision evidence carries DIAGNOSTIC_ONLY.json (schema diagnostic-only-marker-v1,
decision_eligible false, reason, discoveries). "Decision evidence" covers a ruler or comparison, a gate, an ensemble-weight fit, a stacking
input and a teacher / pseudo-label source.

Consumers that accept arbitrary CSV paths call require_decision_eligible(path, consumer) BEFORE reading anything. A file is refused when
its own directory, or any ancestor directory below the workspace root, holds
  - DIAGNOSTIC_ONLY.json (whatever its content: the marker alone quarantines), or
  - an assembly / registration record (result.json or manifest.json) whose top-level decision_eligible, or evaluation.decision_eligible,
    is false;
a record of that name that cannot be read as JSON is refused too (eligibility unverifiable, fail closed).

  python tools/diagnostic_quarantine.py check PATH [PATH ...]   # prints the verdict per path; exit 3 when any path is quarantined
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'DIAGNOSTIC_ONLY.json'
SCHEMA = 'diagnostic-only-marker-v1'
RECORDS = ('result.json', 'manifest.json')
FORBIDDEN_USES = ('ruler or comparison (tools/g1_compare.py)', 'gate', 'ensemble-weight fit', 'ensemble member', 'stacking input',
                  'teacher / pseudo-label source (tools/exp103_distill_labels.py)')


class DiagnosticOnlyInput(SystemExit):
    """Raised by require_decision_eligible: a SystemExit, so a broad `except Exception` cannot swallow the refusal."""


def _false(value):
    return value is False or (isinstance(value, str) and value.strip().lower() == 'false')


def _dirs(path, root):
    p = Path(path).resolve()
    start = p if p.is_dir() else p.parent
    dirs = [start, *start.parents]
    root = Path(root).resolve()
    return dirs[:dirs.index(root)] if root in dirs else dirs


def quarantine_reason(path, root=ROOT):
    """None when nothing marks `path` as diagnostic only, else a one-line reason naming the directory and the record."""
    for d in _dirs(path, root):
        marker = d / MARKER
        if marker.exists():
            try:
                body = json.loads(marker.read_text(encoding='utf-8'))
                why = body.get('reason') if isinstance(body, dict) else None
                refs = body.get('discoveries') if isinstance(body, dict) else None
            except (OSError, ValueError):
                why, refs = 'marker not readable (the marker alone quarantines)', None
            tail = f" ({', '.join(refs)})" if isinstance(refs, list) and refs else ''
            return f'{d} carries {MARKER} (decision_eligible false){tail}: {why or "no reason recorded"}'
        for name in RECORDS:
            record = d / name
            if not record.is_file():
                continue
            try:
                body = json.loads(record.read_text(encoding='utf-8'))
            except (OSError, ValueError) as e:
                return f'{record} cannot be read as JSON ({type(e).__name__}), so decision eligibility is unverifiable (fail closed)'
            if not isinstance(body, dict):
                continue
            evaluation = body.get('evaluation') if isinstance(body.get('evaluation'), dict) else {}
            if _false(body.get('decision_eligible')) or _false(evaluation.get('decision_eligible')):
                leak = body.get('leak') or evaluation.get('label_sets') or ''
                return f'{record} records decision_eligible false' + (f': {leak}' if leak else '')
    return None


def require_decision_eligible(path, consumer, root=ROOT):
    """Raise DiagnosticOnlyInput (exit message) when `path` is quarantined; return the resolved path otherwise."""
    reason = quarantine_reason(path, root)
    if reason:
        raise DiagnosticOnlyInput(
            f'{consumer}: refusing {path}: {reason}. Diagnostic-only predictions (LEAKY by teacher ancestry, D-B-024) are never a '
            'ruler, comparison, gate, ensemble-weight fit, ensemble member, stacking input or teacher / pseudo-label source.')
    return Path(path).resolve()


def marker_body(experiment, reason, discoveries, **extra):
    return dict(schema=SCHEMA, decision_eligible=False, experiment=experiment, reason=reason, discoveries=list(discoveries),
                forbidden_uses=list(FORBIDDEN_USES), written_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra)


def _text(body):
    return json.dumps(body, indent=1, ensure_ascii=False, allow_nan=False) + '\n'


def write_marker(directory, body):
    """Create <directory>/DIAGNOSTIC_ONLY.json (never overwrites; LF newlines). Returns its path."""
    if body.get('decision_eligible') is not False:
        raise ValueError('a diagnostic-only marker must record decision_eligible false')
    path = Path(directory) / MARKER
    with open(path, 'x', encoding='utf-8', newline='\n') as f:
        f.write(_text(body))
    return path


def update_marker(directory, **fields):
    """Add fields to an existing marker (atomic replace; decision_eligible stays false). Returns the new body."""
    path = Path(directory) / MARKER
    body = json.loads(path.read_text(encoding='utf-8'))
    body.update(fields)
    if body.get('decision_eligible') is not False:
        raise ValueError('a diagnostic-only marker must record decision_eligible false')
    tmp = path.with_name(f'{MARKER}.tmp-{os.getpid()}')
    with open(tmp, 'x', encoding='utf-8', newline='\n') as f:
        f.write(_text(body))
    os.replace(tmp, path)
    return body


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] != 'check' or len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    worst = 0
    for p in argv[1:]:
        reason = quarantine_reason(p)
        print(json.dumps(dict(path=p, decision_eligible=reason is None, reason=reason), ensure_ascii=False))
        worst = 3 if reason else worst
    return worst


if __name__ == '__main__':
    sys.exit(main())
