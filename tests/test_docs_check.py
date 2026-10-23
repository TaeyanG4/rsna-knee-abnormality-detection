"""Check errors a maintainer can miss without changing project documents."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('docs_check', Path(__file__).resolve().parents[1] / 'tools/docs_check.py')
docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docs)


def test_windows_newlines_do_not_change_character_budget(tmp_path):
    text = '# Rules\n\n' + '가' * 2489 + '\n'
    assert len(text) == 2499
    (tmp_path / 'AGENTS.md').write_bytes(text.replace('\n', '\r\n').encode('utf-8'))
    report = docs.check(tmp_path, ['AGENTS.md'])
    assert report['ok'] and report['files'][0]['characters'] == 2499
    (tmp_path / 'AGENTS.md').write_text(text + 'too long\n', encoding='utf-8')
    assert not docs.check(tmp_path, ['AGENTS.md'])['ok']


def test_link_audit_skips_code_and_web_but_reports_missing_file(tmp_path):
    (tmp_path / 'some file.md').write_text('# Existing\n', encoding='utf-8')
    path = tmp_path / 'review.md'
    path.write_text('# Review\n\n[ok](<some file.md>) [web](https://example.test/x)\n\n'
                    '```md\n[example](not-a-real-file.md)\n```\n\n[broken](missing.md)\n', encoding='utf-8')
    report = docs.check(tmp_path, ['review.md'])
    assert len(report['problems']) == 1
    assert 'missing local link: missing.md' == report['problems'][0]['issue']


def test_retired_instruction_and_required_structure_are_reported(tmp_path):
    (tmp_path / 'HANDOFF.md').write_text('# Handoff\n\n최대 30분마다 기록한다.\n', encoding='utf-8')
    report = docs.check(tmp_path, ['HANDOFF.md'])
    assert any('retired periodic' in p['issue'] for p in report['problems'])
    assert sum('missing section' in p['issue'] for p in report['problems']) == 3


def test_input_outside_workspace_is_not_read(tmp_path):
    report = docs.check(tmp_path, ['../outside.md'])
    assert report['checked'] == 0 and report['problems'][0]['issue'] == 'path outside workspace'
