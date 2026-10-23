"""v1.0.0: pinned v1.8.1 replay; preserve every v9 output byte."""
from types import ModuleType

from tools import k1n_replay_v173 as previous

ROOT, util, exact = previous.ROOT, previous.util, previous.exact
PINS = {
    'k1n_config_v1.py': 'aeb5ebdc204a285a4660a2c448829c93c79c11e87ea329792318dd5483a6fea4',
    'k1n_flat_p_v1.py': '3776bb60238ba0a36ed66055b0a0073f86492c9c432bd0b16bbc28d841722851',
    'k1n_members_v1.py': '1be50aa37678c74ebfe76dfff9fd171df41a79c9991356df284f4cb33a916618',
    'k1n_stage_v1.py': '60f1597fd6d8007005b36c20dc4679aa4139cb57929b2ae0b62ffe1bbfa62edd',
}


def load():
    directory = ROOT/'tools/k1n_frozen_v181'
    for name, pin in PINS.items():
        util.need(util.sha(directory/name) == pin, 'v181 source drift: '+name)
    modules = {n:ModuleType('v181_'+n) for n in ('schema', 'builder', 'members', 'flat_p')}
    def source_path(name):
        return directory/name.split('/')[-1] if name in ('tools/'+n for n in PINS) else ROOT/name
    for key, filename in [('schema','k1n_config_v1.py'), ('members','k1n_members_v1.py'),
                          ('flat_p','k1n_flat_p_v1.py'), ('builder','k1n_stage_v1.py')]:
        module = modules[key]
        module.__dict__.update(modules, source_path=source_path)
        text = (directory/filename).read_text(encoding='utf-8')
        for name, alias in [('k1n_config_v1','schema'), ('k1n_stage_v1','builder'),
                            ('k1n_members_v1','members'), ('k1n_flat_p_v1','flat_p')]:
            text = text.replace('from tools import '+name+' as '+alias, 'pass  # pinned '+alias)
        text = text.replace("(ROOT/'tools/k1n_config_v1.py').read_bytes()",
                            "source_path('tools/k1n_config_v1.py').read_bytes()")
        text = text.replace('(ROOT/SCHEMA).read_bytes()', 'source_path(SCHEMA).read_bytes()')
        text = text.replace('builder_sources={n:util.sha(ROOT/n) for n in paths}',
                            'builder_sources={n:util.sha(source_path(n)) for n in paths}')
        exec(compile(text, '<v181 '+key+'>', 'exec'), module.__dict__)
    from tools import k1n_stage_v1 as current
    modules['builder'].stage = current.stage
    return modules['builder']
