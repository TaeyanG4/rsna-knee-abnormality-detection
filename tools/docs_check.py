"""Read-only checks for active project documents; no model/API/DB calls.

Checks UTF-8, LF-normalized size, headings, simple inline local link paths and
retired periodic handoff requirements. Does not validate prose, web URLs or anchors.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = (
    'AGENTS.md', 'CLAUDE.md', 'HANDOFF.md', 'plan.md', 'discoveries.md', 'docs/STATE.md',
    'docs/runbooks/handoff-maintenance.md', 'docs/runbooks/continuous-research.md',
    'docs/runbooks/tools.md', 'docs/runbooks/strategy-review.md', 'docs/handoffs/TEMPLATE.md',
    'docs/decisions/0002-personal-research.md', 'docs/handoffs/INDEX.md',
)
LIMITS = {'AGENTS.md': (2500, None), 'HANDOFF.md': (2000, 50),
          'docs/STATE.md': (2500, None), 'docs/handoffs/INDEX.md': (2500, None)}
REQUIRED_SECTIONS = {
    'HANDOFF.md': ('현재 작업', '결과·보류', '재개·예약'),
    'plan.md': ('우선순위', '실험 계약', '기록 형식'),
    'discoveries.md': ('요약·색인', 'Closed ideas', '교훈 상세'),
}
LINK = re.compile(r'\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)(?:\s+"[^"]*")?\)')


def check(root: Path, names=DEFAULTS):
    root = root.resolve()
    problems, files = [], []
    for name in dict.fromkeys(names):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            problems.append({'file': str(name), 'issue': 'path outside workspace'})
            continue
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding='utf-8')  # universal newlines => LF
        except (OSError, UnicodeError) as exc:
            problems.append({'file': rel, 'issue': str(exc)})
            continue
        lines = text.splitlines()
        files.append({'file': rel, 'characters': len(text), 'lines': len(lines)})
        def issue(message, line=None):
            problems.append({'file': rel, 'line': line, 'issue': message})
        cap, line_cap = LIMITS.get(rel, (None, None))
        if rel.startswith('docs/handoffs/') and re.fullmatch(r'\d{4}-\d{2}-\d{2}-[^/]+\.md', path.name):
            cap, line_cap = 2000, 50
        if cap and len(text) > cap:
            issue(f'{len(text)} characters exceeds {cap} (LF normalized)')
        if line_cap and len(lines) > line_cap:
            issue(f'{len(lines)} lines exceeds {line_cap}')
        if rel == 'CLAUDE.md':
            if text != '[AGENTS.md](AGENTS.md)를 읽고 따르세요.\n':
                issue('CLAUDE must contain only the AGENTS pointer')
        else:
            headings = []
            body = []
            fence = None
            for index, line in enumerate(lines):
                marker = re.match(r'^\s*(`{3,}|~{3,})', line)
                if marker:
                    token = marker.group(1)
                    if fence is None:
                        fence = token
                    elif token[0] == fence[0] and len(token) >= len(fence):
                        fence = None
                    continue
                if fence:
                    continue
                body.append((index + 1, line))
                if re.match(r'^#{1,6} ', line):
                    headings.append(line)
                    if index and lines[index - 1].strip():
                        issue('blank line required before heading', index + 1)
                    if index + 1 < len(lines) and lines[index + 1].strip():
                        issue('blank line required after heading', index + 1)
                if line.rstrip() != line:
                    issue('trailing whitespace', index + 1)
            if sum(h.startswith('# ') for h in headings) != 1:
                issue('exactly one H1 title required')
            for required in REQUIRED_SECTIONS.get(rel, ()):
                if '## ' + required not in headings:
                    issue(f'missing section: {required}')
            for number, line in body:
                if re.search(r'최대\s*30\s*분|30\s*분마다.*(?:갱신|기록|체크포인트)', line):
                    issue('retired periodic handoff requirement; use state changes', number)
                for match in LINK.finditer(line):
                    link = match.group(1).strip('<>')
                    target = unquote(link.split('#', 1)[0])
                    if not target or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target):
                        continue
                    if not (path.parent / target).exists():
                        issue(f'missing local link: {link}', number)
        if text.startswith('\ufeff') or not text.endswith('\n'):
            issue('use UTF-8 without BOM and a final newline')
    return {'ok': not problems, 'checked': len(files), 'files': files, 'problems': problems,
            'scope': 'file paths and basic Markdown; web/anchor/prose validation excluded'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', help='workspace-relative files; defaults to active documents')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--json', action='store_true', help='structured output for a caller to save')
    args = parser.parse_args()
    result = check(args.root, args.paths or DEFAULTS)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"{'PASS' if result['ok'] else 'FAIL'}: {result['checked']} documents; {len(result['problems'])} issues")
        for item in result['problems']:
            print(f"{item['file']}:{item.get('line') or '-'}: {item['issue']}")
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
