"""PL production release integration 1.0.0; metadata only, never launches jobs.

Changelog 1.0.0: check the frozen release roster before training; reuse A7 for
checkpoint/receipt binding. A five-member run cannot satisfy a ten-member seal.
No frozen registration, checkpoint, or receipt is rewritten by this adapter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp169_release_c85r7_pl15_v140 as release
from tools import vnext_r7_pl_v1 as pl
from tools import pl_production_a7_v1 as production_a7

VERSION = '1.0.0'
OUT = release.OUT/'production-v1'
MIL = [m for m in pl.member_roster(seeds=(42,)) if m['kind'] == 'MIL']


def register(stage=OUT, parent=release.OUT):
    """A new child seal changes only the MIL roster to five seed42 members."""
    stage, parent = Path(stage), Path(parent)
    pl.need(stage.resolve() != parent.resolve(), 'frozen parent stage')
    release.validate(parent)
    students = {}
    for arm in ('global', 'fill'):
        original = pl.registration(parent/'student'/arm)
        projected = dict(original, members=MIL, seeds=[42],
                         projection='C85-R7-PL15 production-v1 seed42 MIL-only',
                         parent_registration_sha256=pl.digest(original))
        pl.dump(stage/'student'/arm/'registration.json', projected, immutable=True)
        pl.metadata.write_once(stage/'student'/arm/'registration.sha256',
                               (pl.digest(projected)+'\n').encode())
        template = pl.read(parent/'student'/arm/'bindings-template.json')
        template.update(registration_sha256=pl.digest(projected),
                        members=[dict(m,path=None,sha256=None,bytes=None,training_receipt=None) for m in MIL])
        pl.dump(stage/'student'/arm/'bindings-template.json', template, immutable=True)
        students[arm] = pl.digest(projected)
    reg = dict(version=VERSION, parent=str(parent.resolve()),
               parent_registration_sha256=pl.sha(parent/'registration.json'),
               student_registrations=students, members=MIL, variants=release.VARIANTS,
               outer_weights=[.85,.15], exposure='EXPOSED_EXPLORATORY', clean_oof=False,
               source_sha256=pl.sha(__file__), a7_compatibility_sha256=pl.sha(production_a7.__file__),
               qualification='a visible+N32; b visible; no auto qualification')
    pl.dump(stage/'registration.json',reg,immutable=True)
    pl.metadata.write_once(stage/'registration.sha256',(pl.digest(reg)+'\n').encode())
    return dict(status='REGISTERED_CHILD',stage=str(stage),members=5)


def registration(stage=OUT):
    stage = Path(stage)
    reg = pl.read(stage/'registration.json')
    pl.need(pl.digest(reg)==(stage/'registration.sha256').read_text().strip(), 'child registration seal')
    pl.need(reg['version']==VERSION and reg['members']==MIL and reg['variants']==release.VARIANTS
            and reg['source_sha256']==pl.sha(__file__)
            and reg['a7_compatibility_sha256']==pl.sha(production_a7.__file__), 'child source/recipe drift')
    parent = Path(reg['parent'])
    release.validate(parent)
    pl.need(pl.sha(parent/'registration.json')==reg['parent_registration_sha256'],'parent release changed')
    for arm in ('global','fill'):
        child=pl.registration(stage/'student'/arm)
        original=pl.registration(parent/'student'/arm)
        pl.need(child==dict(original,members=MIL,seeds=[42],
                    projection='C85-R7-PL15 production-v1 seed42 MIL-only',
                    parent_registration_sha256=pl.digest(original)), 'child projection drift')
        pl.need(pl.digest(child)==reg['student_registrations'][arm],'child student seal')
    return reg


def validate(stage=OUT):
    registration(stage)
    return dict(status='PASS_PREPARED_CHILD',members=5,qualified_for_submission=False)


def preflight(stage=release.OUT, *, arm, seeds=(42,)):
    """Authenticate metadata, then compare exactly what B intends to train."""
    pl.need(arm in ('global', 'fill'), 'winning arm must be global or fill')
    child = (Path(stage)/'registration.json').exists() and pl.read(Path(stage)/'registration.json').get('version')==VERSION
    (validate if child else release.validate)(stage)
    reg = pl.registration(Path(stage)/'student'/arm)
    requested = [m for m in pl.member_roster(seeds=seeds) if m['kind'] == 'MIL']
    sealed = reg['members']
    missing = [m['id'] for m in sealed if m not in requested]
    extra = [m['id'] for m in requested if m not in sealed]
    return dict(version=VERSION, status='HOLD' if missing or extra else 'READY_FOR_EXPORT',
                arm=arm, registration_sha256=pl.digest(reg),
                requested_members=[m['id'] for m in requested],
                sealed_members=[m['id'] for m in sealed],
                missing_members=missing, extra_members=extra,
                reason='frozen release roster differs; new versioned registration required'
                if missing or extra else None,
                exposure='EXPOSED_EXPLORATORY', clean_oof=False)


def bind_completed(stage, inputs, output):
    """Bind A6 exports through existing A7; do not copy or alter model tensors.

    `inputs` is the existing v140 bindings template populated with model_best.pt
    and training-receipt.json paths plus authenticated GO and teacher assets.
    The existing immutable binder checks exact roster, hashes and every receipt.
    """
    bound = pl.read(inputs)
    arm = bound.get('arm')
    pl.need(arm in ('global', 'fill'), 'winning arm absent')
    child = (Path(stage)/'registration.json').exists() and pl.read(Path(stage)/'registration.json').get('version')==VERSION
    (validate if child else release.validate)(stage)
    reg = pl.registration(Path(stage)/'student'/arm)
    entries = bound.get('members', [])
    pl.need([m.get('id') for m in entries] == [m['id'] for m in reg['members']],
            'frozen release checkpoint roster; cannot bind five seed42 models to ten-member v1.4.0')
    # Fail before invoking the binder if the destination is a frozen input file.
    output = Path(output).resolve()
    pl.need(output != Path(inputs).resolve(), 'binding output must differ from inputs')
    protected = {Path(stage).resolve()/'registration.json',
                 Path(stage).resolve()/'registration.sha256',
                 Path(stage).resolve()/'variant-a.json',
                 Path(stage).resolve()/'variant-b.json'}
    pl.need(output not in protected, 'frozen stage output prohibited')
    return production_a7.adapter.bind(Path(stage)/'student'/arm, inputs, output) if child else release.bind(stage, inputs, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register','validate','preflight', 'bind'])
    parser.add_argument('--stage', type=Path, default=OUT)
    parser.add_argument('--arm', choices=['global', 'fill'])
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command in ('register','validate'):
        result=globals()[args.command](args.stage)
    elif args.command == 'preflight':
        parser.error('--arm is required') if args.arm is None else None
        result = preflight(args.stage, arm=args.arm)
    else:
        if args.inputs is None or args.output is None:
            parser.error('bind requires --inputs and --output')
        result = bind_completed(args.stage, args.inputs, args.output)
    print(json.dumps(result, indent=2))
    return 2 if result['status'] == 'HOLD' else 0


if __name__ == '__main__':
    raise SystemExit(main())
