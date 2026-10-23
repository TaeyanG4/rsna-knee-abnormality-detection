"""v1.0.0: pinned v1.6.0 source replay, preserving all historical artifact bytes."""
from types import ModuleType

from tools import k1n_replay_v150 as previous

ROOT, util, exact = previous.ROOT, previous.util, previous.exact
PINS = {
    'k1n_config_v1.py': '9a35f5b6cfd82b53ecdabdaf03ef41c5c3539c56c36a4708c162afc09aa2051b',
    'k1n_stage_v1.py': 'da01725e3b61fd280f26e6297d98c051887bc619bfaf708314614d3af60b8bfc',
    'k1n_stage_v9.py': 'a20ba005aa5448e96956b0fed66aa534617c2871a3cfd8ce4c468f1594bc2672',
}


def load():
    directory = ROOT/'tools/k1n_frozen_v160'
    for name, pin in PINS.items():
        util.need(util.sha(directory/name) == pin, 'v160 source drift: '+name)
    schema, builder, runtime = [ModuleType('v160_'+n) for n in ('schema', 'builder', 'runtime')]
    exec(compile((directory/'k1n_config_v1.py').read_bytes(), '<v160 schema>', 'exec'), schema.__dict__)
    runtime.schema, runtime.builder = schema, builder
    runtime.source_path = lambda n: directory/n.split('/')[-1] if n in (
        'tools/'+name for name in PINS) else ROOT/n
    text = (directory/'k1n_stage_v9.py').read_text(encoding='utf-8')
    text = exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema')
    assert text.count('    from tools import k1n_stage_v1 as builder') == 2
    text = text.replace('    from tools import k1n_stage_v1 as builder', '    # pinned builder')
    text = exact(text, "(ROOT/'tools/k1n_config_v1.py').read_bytes()",
                 "source_path('tools/k1n_config_v1.py').read_bytes()")
    text = exact(text, 'builder_sources={n:util.sha(ROOT/n) for n in paths}',
                 'builder_sources={n:util.sha(source_path(n)) for n in paths}')
    exec(compile(text, '<v160 runtime builder>', 'exec'), runtime.__dict__)
    builder.schema, builder.runtime9 = schema, runtime
    text = (directory/'k1n_stage_v1.py').read_text(encoding='utf-8')
    text = exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema')
    text = exact(text, 'from tools import k1n_stage_v9 as runtime9', '# pinned runtime builder')
    exec(compile(text, '<v160 builder>', 'exec'), builder.__dict__)
    return builder
