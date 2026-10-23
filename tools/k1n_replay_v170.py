"""v1.0.0: pinned v1.7.0 member replay; preserve existing artifact bytes."""
from types import ModuleType

from tools import k1n_replay_v160 as previous

ROOT, util, exact = previous.ROOT, previous.util, previous.exact
PINS = {
    'k1n_stage_v1.py': '31eae7f3c251b32afe0058eac4beece4351220bd9c891bb81c0dd52d7b521342',
    'k1n_config_v1.py': '997c4260b3be24d535240268f89866d3ecf9f385ad80bc853730f57c2a5ca8b2',
    'k1n_members_v1.py': '73a1fe99859882de6bcd3b1c32d8dd2647f36632729c252549addc2cae6eef40',
}


def load():
    directory = ROOT/'tools/k1n_frozen_v170'
    for name, pin in PINS.items():
        util.need(util.sha(directory/name) == pin, 'v170 source drift: '+name)
    schema, builder, members = [ModuleType('v170_'+n) for n in ('schema', 'builder', 'members')]
    exec(compile((directory/'k1n_config_v1.py').read_bytes(), '<v170 schema>', 'exec'), schema.__dict__)
    members.schema, members.builder = schema, builder
    members.source_path = lambda n: directory/n.split('/')[-1] if n in (
        'tools/'+name for name in PINS) else ROOT/n
    text = (directory/'k1n_members_v1.py').read_text(encoding='utf-8')
    text = exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema')
    text = exact(text, '    from tools import k1n_stage_v1 as builder', '    # pinned builder')
    text = exact(text, "(ROOT/'tools/k1n_config_v1.py').read_bytes()",
                 "source_path('tools/k1n_config_v1.py').read_bytes()")
    text = exact(text, 'builder_sources={n:util.sha(ROOT/n) for n in paths}',
                 'builder_sources={n:util.sha(source_path(n)) for n in paths}')
    exec(compile(text, '<v170 members>', 'exec'), members.__dict__)
    builder.schema, builder.members = schema, members
    text = (directory/'k1n_stage_v1.py').read_text(encoding='utf-8')
    text = exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema')
    text = exact(text, 'from tools import k1n_members_v1 as members', '# pinned members')
    exec(compile(text, '<v170 builder>', 'exec'), builder.__dict__)
    # Preserve the caller's destination hook used by CPU-only fixture builds.
    from tools import k1n_stage_v1 as current
    builder.stage = current.stage
    return builder
