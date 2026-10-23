"""v1.0.0: pinned source replay of K1N v1.1.0/v1.2.0 builders.

Regenerate artifacts; never return stored stage outputs as regeneration evidence.
Only local source reads are redirected. Serving code is not modified.
"""
from pathlib import Path
from types import ModuleType

from tools import c85r7_r6w958k1x3sr_v1164 as before

ROOT, util = before.ROOT, before.util
PINS = {
    'v110': {
        'k1n_stage_v1.py': '356b7b10fd10f4c539b5d86e75e3296f36f6e5a3fc0b67674c062b2ad494a35c',
        'k1n_config_v1.py': '1607b9543e033e6d1d91b6787b2f50d63b9c104c42924e247250b6ecf7147967',
        'k1sr_runtime_v7.py': 'c34432033c511fad4077c70ce4a8743b2f54f21b6b1403bcab69bdc854f0435d',
    },
    'v120': {
        'k1n_stage_v1.py': '11065726daf52e5827e141214b2ed43ab112b347c6792792205dbf19939b334e',
        'k1n_config_v1.py': 'e534ada13820073d5cfea1eddced74fe2a5abbca3f80367fd874d40f69f138f1',
        'k1sr_runtime_v7.py': '392d3014d453b5751fe955f0a9c2001f41f21150c26593faea0c411844ec5970',
    },
}


def load(profile):
    directory = ROOT/'tools/k1n_compat'/profile
    for name, pin in PINS[profile].items():
        util.need(util.sha(directory/name) == pin, 'compat source drift: '+profile+'/'+name)
    schema = ModuleType('k1n_frozen_schema_'+profile)
    exec(compile((directory/'k1n_config_v1.py').read_bytes(), '<pinned schema>', 'exec'), schema.__dict__)
    module = ModuleType('k1n_frozen_builder_'+profile)
    module.schema = schema
    def source_path(name):
        return directory/Path(name).name if Path(name).name in PINS[profile] else ROOT/name
    module.source_path = source_path
    text = (directory/'k1n_stage_v1.py').read_text(encoding='utf-8')
    text = before.exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema supplied above')
    for name in ('k1n_config_v1.py', 'k1sr_runtime_v7.py'):
        text = before.exact(text, "(ROOT/'tools/"+name+"')", "source_path('tools/"+name+"')")
    text = before.exact(text, 'util.sha(ROOT/n)', 'util.sha(source_path(n))')
    exec(compile(text, '<pinned builder '+profile+'>', 'exec'), module.__dict__)
    return module
