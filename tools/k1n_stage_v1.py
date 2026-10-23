"""K1 N-model stage generator v1.8.2, local CPU only.

Changelog v1.8.2: runtime v10 capture/call deadlines; pinned v1.8.1 replay.

Changelog v1.8.1: compose flat P with all recipes; N32 qualification limit 9 h.

Changelog v1.8.0: opt-in flat P twin; all public artifacts replay pinned v1.7.3.
Changelog v1.7.3: P25S3 and opt-in measured N32 qualification; pinned v1.7.2 replay.
Changelog v1.7.2: exact P25 member recipe and calibrated projection; preserve v1.7.1 bytes.
Changelog v1.7.1: flat S1 fold recipes, guard8.8, eight models; pinned v1.7.0 replay.
Changelog v1.7.0: member weights, fold templates/projection, pinned v1.6.0 replay.
Changelog v1.6.0: opt-in v9, pinned v1.5.0 regeneration.
Changelog v1.5.0: registered guard-only successors; pinned v1.4.0 regeneration.
Changelog v1.4.0: opt-in runtime v8; policy-absent artifacts replay pinned v1.3.1.
Changelog v1.3.1: bind historical replay to exact registered identities;
unrelated configs with the same recipe keep their v1.2.0 runtime.
Changelog v1.3.0: registered R6 stride 1/2, frozen-source compatibility replay,
exact parent-relative stride-only diffs; no K1/runtime/scientific recipe drift.
Absent/explicit-1 R6 stride preserves all legacy generated artifact bytes.
"""
from __future__ import annotations

import argparse
import base64
import copy
import difflib
import json
from pathlib import Path

from tools import c85r7_r6w958k1x3sr_v1164 as before
from tools import k1n_config_v1 as schema
from tools import k1n_parent_v1 as parent_edit
from tools import k1n_compat_v1 as compat
from tools import k1n_stage_v8 as fail_loud
from tools import k1n_guard_v1 as guard
from tools import k1n_replay_v150 as replay150
from tools import k1n_replay_v160 as replay160
from tools import k1n_members_v1 as members
from tools import k1n_replay_v173 as replay173
from tools import k1n_flat_p_v1 as flat_p
from tools import k1n_replay_v181 as replay181
from tools import k1n_stage_v10 as runtime10

runtime9 = replay160.load().runtime9

VERSION = '1.8.2'
ROOT, util = before.ROOT, before.util
exact, digest = before.exact, before.digest
ADDED = {'tools/k1sr_runtime_v7.py', 'tools/k1n_config_v1.py', 'tools/k1_shared_render_v2.py'}
# Exact historical source profiles, not an inference-time recipe selector.
PARENTS = (
    ('configs/k1n/x3-stride2-drop-o.json', 'v110'),
    ('configs/k1n/x3-stride2-alpha-vector-v2.json', 'v120'),
    ('configs/k1n/x3-stride2-drop-o-alpha-vector-v2.json', 'v120'),
)
# Explicit successors requested by H-B-228; never infer lineage from recipe alone.
SUCCESSORS = {
    ('1.1.70-r6s2k1n3s2p25', 'taeyangg4/knee-lb-r6s2k1n3s2p25-v1'):
        'configs/k1n/x3-stride2-drop-o.json',
    ('1.1.70-r6s2k1n3s2av2', 'taeyangg4/knee-lb-r6s2k1n3s2av2-v1'):
        'configs/k1n/x3-stride2-alpha-vector-v2.json',
}
STRIDE_ANCHOR = '"--eval-stride", str(manifest.get("eval_stride", 3))'
ADAPTER_ANCHOR = "dataset.eval_stride) == (384, 'v4', 3, 3.5, 1)"


def legacy(value):
    c = schema.validate(value)
    c.pop('r6_eval_stride', None)
    return c


def recipe(value):
    return {k:v for k,v in legacy(value).items() if k not in ('release_version', 'kernel_id')}


def baseline(value):
    """Use the exact frozen parent for known recipes; v1.2.0 for other configs.

    Without stride2, only the historical release identity selects v1.1.0.
    For the two explicitly registered stride2 successors, require the entire
    parent recipe to match. Other identities retain the v1.2.0 generator.
    """
    c = legacy(value)
    for filename, profile in PARENTS:
        registered = legacy(json.loads((ROOT/filename).read_bytes()))
        successor = (value.get('r6_eval_stride', 1) == 2 and
                     SUCCESSORS.get((c['release_version'], c['kernel_id'])) == filename)
        if successor:
            util.need(recipe(c) == recipe(registered), 'registered successor recipe drift')
        if c == registered or successor:
            return compat.load(profile), registered
    return compat.load('v120'), c


def canonical(value):
    """One recipe identity for generation, qualification and admission."""
    c = schema.validate(value)
    if c.get('k1_runtime_version') == 10:
        return runtime10.canonical(c)
    if c.get('p_weight_mode') == 'flat':
        return flat_p.canonical(c)
    return members.canonical(c) if members.selected(c) else c


def config(value, diagnostic=False):
    c = schema.validate(value)
    return (runtime10 if c.get('k1_runtime_version') == 10 else replay181.load()).config(c, diagnostic)


def stage(value, diagnostic=False):
    c = schema.validate(value)
    return before.parent.INF3.parent / ('v'+c['release_version']+('-n32' if diagnostic else ''))


def adapter(data):
    """Same two-line, fail-closed adapter change as H-B-228."""
    text = exact(data.decode(), ADAPTER_ANCHOR,
                 "dataset.eval_stride) == (384, 'v4', 3, 3.5, 2)")
    return exact(text, "VERSION = '2.0.0'", "VERSION = '2.0.1-hb228-r6s2'").encode()


def frozen_parent(engine, registered, diagnostic):
    """Regenerate, then compare every artifact against the immutable parent."""
    outputs, reg = engine.artifacts(registered, diagnostic)
    dest = engine.stage(registered, diagnostic)
    if dest.exists():
        util.need({p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()} == set(outputs),
                  'parent artifact inventory drift')
        for name, data in outputs.items():
            util.need((dest/name).read_bytes() == data, 'parent artifact drift: '+name)
    return outputs, reg


def package(value, diagnostic=False):
    c = schema.validate(value)
    return (runtime10 if c.get('k1_runtime_version') == 10 else replay181.load()).package(c, diagnostic)


def artifacts(value, diagnostic=False):
    c = schema.validate(value)
    return (runtime10 if c.get('k1_runtime_version') == 10 else replay181.load()).artifacts(c, diagnostic)


def run(command, value):
    """Validate all bytes and both stride-only proofs before creating either stage."""
    if isinstance(value, dict) and value.get('k1_runtime_version') != 10:
        return replay181.load().run(command, value)
    c = canonical(value)
    pairs = [(d, stage(c,d), *artifacts(c,d)) for d in (False,True)]
    if command == 'prepare':
        util.need(all(not dest.exists() for _,dest,_,_ in pairs), 'immutable stage exists; use validate')
        for _, dest, outputs, _ in pairs:
            dest.mkdir()
            for name, data in outputs.items(): util.write_once(dest/name, data)
    else:
        util.need(command == 'validate', 'unknown command')
    result = []
    for diagnostic, dest, outputs, reg in pairs:
        util.need({p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()} == set(outputs),
                  'immutable artifact inventory drift')
        for name, data in outputs.items():
            util.need((dest/name).read_bytes() == data, 'immutable artifact drift: '+name)
        result.append(dict(stage=reg['stage'], kernel_id=reg['kernel_id'],
            kernel_sha256=reg['kernel_sha256'], never_submit=diagnostic))
    result = dict(status='PREPARED_LOCAL_ONLY' if command == 'prepare' else 'PASS_CPU_VALIDATE',
        stages=result, qualified_for_submission=False)
    if members.selected(c):
        result['runtime_projection'] = members.projection(c)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','validate'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.command, json.loads(args.config.read_bytes())), indent=2))


if __name__ == '__main__':
    main()
