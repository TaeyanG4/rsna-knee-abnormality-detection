"""v1.0.0: pinned v1.7.1 member replay; preserve existing artifact bytes."""
from types import ModuleType

from tools import k1n_replay_v160 as previous

ROOT, util, exact = previous.ROOT, previous.util, previous.exact
PINS = {
    'k1n_stage_v1.py': '7fdfbf8aaab7ab83ebf95d9db573c6f8798defae02bd34520afcf7ca38a8457e',
    'k1n_config_v1.py': '5ad1fa58290ad012c126c0a767b7d37c0445ccc9be0a85becc054377d14b2051',
    'k1n_members_v1.py': '11aac5ba3618dd5465ca0433e9382e56dc5d3454fa11e024fb3ec8eb755ea622',
}


def load():
    directory = ROOT/'tools/k1n_frozen_v171'
    for name, pin in PINS.items():
        util.need(util.sha(directory/name) == pin, 'v171 source drift: '+name)
    schema, builder, members = [ModuleType('v171_'+n) for n in ('schema', 'builder', 'members')]
    exec(compile((directory/'k1n_config_v1.py').read_bytes(), '<v171 schema>', 'exec'), schema.__dict__)
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
    exec(compile(text, '<v171 members>', 'exec'), members.__dict__)
    builder.schema, builder.members = schema, members
    text = (directory/'k1n_stage_v1.py').read_text(encoding='utf-8')
    text = exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema')
    text = exact(text, 'from tools import k1n_members_v1 as members', '# pinned members')
    exec(compile(text, '<v171 builder>', 'exec'), builder.__dict__)
    # Preserve the caller's destination hook used by CPU-only fixture builds.
    from tools import k1n_stage_v1 as current
    builder.stage = current.stage
    return builder
