"""v1.0.0: source replay of v1.5.0, with its original hash inventory."""
from types import ModuleType

from tools import k1n_guard_v1 as previous

ROOT, util, exact = previous.ROOT, previous.util, previous.exact
PINS = {
    'k1n_config_v1.py': 'b4f731bdda21afb6fa7a861e61fe9e85e91fc61b18c2af74f3c5aad0e8931c88',
    'k1n_stage_v1.py': '330528e7a6a127917d488eb26ab2e2b2f1050add6948bbb2aa383163814b3364',
    'k1n_guard_v1.py': '2f64da2e4e9cc34b1899d00e2fc18fb6833f1524de28b30749eb5ab565a70162',
}


def load():
    directory = ROOT/'tools/k1n_compat/v150'
    for name, pin in PINS.items():
        util.need(util.sha(directory/name) == pin, 'v150 source drift: '+name)
    schema, builder, guard = [ModuleType('v150_'+n) for n in ('schema', 'builder', 'guard')]
    exec(compile((directory/'k1n_config_v1.py').read_bytes(), '<v150 schema>', 'exec'), schema.__dict__)
    guard.schema, guard.builder = schema, builder
    guard.source_path = lambda n: directory/n.split('/')[-1] if n in (
        'tools/'+name for name in PINS) else ROOT/n
    text = (directory/'k1n_guard_v1.py').read_text(encoding='utf-8')
    text = exact(text, '\nfrom tools import k1n_config_v1 as schema\n', '\n# pinned schema\n')
    text = exact(text, '    from tools import k1n_stage_v1 as builder', '    # pinned builder')
    text = exact(text, '\n        builder_sources={n:util.sha(ROOT/n) for n in paths},',
                 '\n        builder_sources={n:util.sha(source_path(n)) for n in paths},')
    text = exact(text, "    paths += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'tools/k1n_compat').rglob('*.py'))",
                 "    paths += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'tools/k1n_compat').rglob('*.py') if p.parent.name in ('v110','v120','v131','v140'))")
    exec(compile(text, '<v150 guard>', 'exec'), guard.__dict__)
    builder.schema, builder.guard = schema, guard
    text = (directory/'k1n_stage_v1.py').read_text(encoding='utf-8')
    text = exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema')
    text = exact(text, 'from tools import k1n_guard_v1 as guard', '# pinned guard')
    exec(compile(text, '<v150 builder>', 'exec'), builder.__dict__)
    return builder
