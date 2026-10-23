"""v1.0.0: pinned v1.7.3 replay, including astra-b1 projection changes."""
from types import ModuleType

from tools import k1n_replay_v160 as previous

ROOT, util, exact = previous.ROOT, previous.util, previous.exact
PINS = {
    'k1n_stage_v1.py': 'c410566e2024bffe2710307a91fcf6fe7eba88bb9acf54682f0267d2e7405f37',
    'k1n_config_v1.py': '113822173dabbe11f4a55249a3a155f09cae3bda6382d8f6b4fce7c348ba8ffc',
    'k1n_members_v1.py': '698f7b737725162fd2ddbe97d0fc00f2e3becfc35380147b8a0d1cab83f1a5f6',
}


def load():
    directory = ROOT/'tools/k1n_frozen_v173'
    for name, pin in PINS.items():
        util.need(util.sha(directory/name) == pin, 'v173 source drift: '+name)
    schema, builder, members = [ModuleType('v173_'+n) for n in ('schema', 'builder', 'members')]
    exec(compile((directory/'k1n_config_v1.py').read_bytes(), '<v173 schema>', 'exec'), schema.__dict__)
    members.schema, members.builder = schema, builder
    members.source_path = lambda n: directory/n.split('/')[-1] if n in (
        'tools/'+name for name in PINS) else ROOT/n
    text = (directory/'k1n_members_v1.py').read_text(encoding='utf-8')
    text = exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema')
    text = text.replace('    from tools import k1n_stage_v1 as builder', '    # pinned builder')
    text = exact(text, "(ROOT/'tools/k1n_config_v1.py').read_bytes()",
                 "source_path('tools/k1n_config_v1.py').read_bytes()")
    text = exact(text, 'builder_sources={n:util.sha(ROOT/n) for n in paths}',
                 'builder_sources={n:util.sha(source_path(n)) for n in paths}')
    exec(compile(text, '<v173 members>', 'exec'), members.__dict__)
    builder.schema, builder.members = schema, members
    text = (directory/'k1n_stage_v1.py').read_text(encoding='utf-8')
    text = exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema')
    text = exact(text, 'from tools import k1n_members_v1 as members', '# pinned members')
    exec(compile(text, '<v173 builder>', 'exec'), builder.__dict__)
    from tools import k1n_stage_v1 as current
    builder.stage = current.stage
    return builder
