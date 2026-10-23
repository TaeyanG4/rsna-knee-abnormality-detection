"""CPU collection/readout/preparation, v1.0.0 (2026-10-12).

Changelog: immutable, source-bound builder HOLD evidence from executed tests;
exact-result reuse. This module has no training or reviewer-PASS operation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from tools import job_runtime as rt
from training.arms import lifecycle as life
from training.arms import collected_screen as cs

VERSION = '1.0.0'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def path(root, name):
    """Restrict all inputs/outputs to the explicit project root before reading."""
    root = Path(root).resolve()
    value = Path(name)
    value = value if value.is_absolute() else root / value
    value = value.absolute()
    try:
        member = value.relative_to(root).as_posix()
    except ValueError:
        raise ValueError('path outside project root') from None
    parts = member.lower().split('/')
    need(not set(parts) & {'api-keys', '.kaggle', '.codex', '.ssh', '.relabel'} and
         '.local/relabel' not in member.lower(), 'restricted path')
    return life._path(root, member)


def pinned(root, record):
    need(set(record) == {'path', 'sha256'}, 'pinned input required')
    target = path(root, record['path'])
    need(life.file_sha256(target) == record['sha256'], 'pinned input changed')
    return target


def put(file, value):
    life.create_new(file, life.canonical_bytes(value))


def inventory(directory):
    return {name: life.file_sha256(life._path(directory, name))
            for name in sorted(life._files(directory))}


def checked_test_receipt(root, receipt_path, spec):
    receipt = rt.read_json(receipt_path)
    need(receipt.get('schema') == 'executed-cpu-tests/v1', 'executed test receipt required')
    need(type(receipt.get('exit_code')) is int and receipt['exit_code'] == 0, 'failed tests')
    counts = receipt.get('counts', {})
    need(set(counts) == {'tests', 'failures', 'errors', 'skipped'} and
         all(type(v) is int and v >= 0 for v in counts.values()) and
         counts['tests'] > 0 and not any(counts[k] for k in ('failures','errors','skipped')), 'tests not fully green')
    command = receipt.get('command')
    need(isinstance(command, list) and command and all(isinstance(v, str) and v for v in command), 'test command missing')
    before, after = receipt.get('source_before'), receipt.get('source_after')
    need(isinstance(before, dict) and before and before == after, 'test source drift')
    for name, digest in before.items():
        need(life.file_sha256(path(root, name)) == digest, 'tested source changed: ' + name)
    for name, digest in spec.document['inputs']['sources'].items():
        need(before.get(name) == digest, 'arm source not covered by test snapshot: ' + name)
    artifacts = receipt.get('artifacts', {})
    need(set(artifacts) == {'log','junit'}, 'test log and junit required')
    files = {name: pinned(root, record) for name, record in artifacts.items()}
    suites = list(ET.parse(files['junit']).getroot().iter('testsuite'))
    actual = {k: sum(int(row.attrib[k]) for row in suites) for k in counts}
    need(actual == counts, 'JUnit counts differ')
    need(not any(list(row.iter('failure')) or list(row.iter('error')) or list(row.iter('skipped')) for row in suites),
         'JUnit has unsuccessful cases')
    return receipt, files


def attest(root, spec, receipt_path, destination):
    """An engineering attestation is always HOLD, never independent approval."""
    receipt_path = path(root, receipt_path)
    receipt_hash = life.file_sha256(receipt_path)
    life.validate_inputs(spec, root)
    receipt, files = checked_test_receipt(root, receipt_path, spec)
    destination = Path(destination)
    work = life._working_directory(destination)
    evidence = {}
    for name, source in {**files, 'test_receipt': receipt_path}.items():
        member = 'artifacts/' + name + source.suffix
        target = work / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        expected = receipt_hash if name == 'test_receipt' else receipt['artifacts'][name]['sha256']
        need(life.file_sha256(target) == expected == life.file_sha256(source), 'evidence copy changed')
        evidence[name] = dict(path=member, sha256=life.file_sha256(target))
    command = dict(schema=life.EVIDENCE_SCHEMA, command=receipt['command'], exit_code=0,
                   artifacts=evidence, smoke=None)
    digest = life.content_sha256(command)
    put(work/'evidence'/f'{digest}.json', command)
    note = dict(schema=life.VERIFICATION_SCHEMA, scope=spec.scope,
                reviewer=dict(name='builder-test-receipt', host='Codex'), verdict='HOLD',
                blockers=['Independent review and actual-target qualification remain required'],
                notes=['Executed CPU engineering tests only; no scientific or GPU launch qualification'],
                bindings=spec.bindings,
                checks=[dict(name='executed-cpu-tests', command=receipt['command'], exit_code=0, log_sha256=digest)],
                smokes=dict(cpu=None, target=None))
    life.verify_evidence(spec, note, work)
    checked_test_receipt(root, receipt_path, spec)
    need(life.file_sha256(receipt_path) == receipt_hash, 'test receipt changed during attestation')
    put(work/'verification.json', note)
    life._publish(work, destination)
    return note


def run(root, plan, destination):
    """Collect -> holistic screen -> optional prepared stage + HOLD admission.

    Collection entries: name/spec{path,sha256}/archive{path,sha256}/receipt_sha256.
    screen_request is a normal collected-screen request; a collection path of
    @name is resolved to this run's authenticated collection.json after collect.
    next is null or {spec:{path,sha256}, tests:{path,sha256}}. No commands in plans.
    """
    root = Path(root).resolve()
    destination = path(root, destination)
    need(set(plan) == {'schema','collections','screen_request','next'} and
         plan['schema'] == 'cpu-pipeline/v1', 'pipeline schema differs')
    need(isinstance(plan['collections'], list) and plan['collections'], 'collections required')
    bindings = {}
    runtime_sources = {Path(module.__file__).resolve().as_posix(): life.file_sha256(module.__file__)
                       for module in (rt, life, cs)}
    runtime_sources[Path(__file__).resolve().as_posix()] = life.file_sha256(__file__)
    entries = []
    for row in plan['collections']:
        need(set(row) == {'name','spec','archive','receipt_sha256'}, 'collection entry differs')
        name = rt.tag_name(row['name'])
        need(name not in {x[0] for x in entries}, 'duplicate collection name')
        spec_file, archive = pinned(root, row['spec']), pinned(root, row['archive'])
        spec = life.ArmSpec(spec_file.read_bytes())
        entries.append((name, spec, archive, row['receipt_sha256']))
        for record in (row['spec'], row['archive']):
            bindings[str(pinned(root, record).relative_to(root)).replace('\\','/')] = record['sha256']
    next_spec = None
    if plan['next'] is not None:
        need(set(plan['next']) == {'spec','tests'}, 'next-stage entry differs')
        next_file = pinned(root, plan['next']['spec'])
        next_spec = life.ArmSpec(next_file.read_bytes())
        life.validate_inputs(next_spec, root)
        tests = pinned(root, plan['next']['tests'])
        checked_test_receipt(root, tests, next_spec)
        bindings.update(next_spec.inventory)
        for record in plan['next'].values():
            bindings[str(pinned(root, record).relative_to(root)).replace('\\','/')] = record['sha256']
    request = copy.deepcopy(plan['screen_request'])
    # Paths in registered screen evidence refer to the final, stable directory.
    for arms in request['collections'].values():
        for arm in arms['arm']:
            ref = arm['collection']
            if str(ref['path']).startswith('@'):
                name = ref['path'][1:]
                need(name in {x[0] for x in entries}, 'unknown collection reference')
                _, spec, archive, rh = next(row for row in entries if row[0] == name)
                manifest = dict(schema='arm-collection/v1', status='COLLECTED_ARTIFACTS_VERIFIED',
                    scientific_status='UNASSESSED', spec_sha256=spec.sha256,
                    receipt_sha256=rh, archive_sha256=life.file_sha256(archive))
                arm['collection'] = dict(path=(destination/'collections'/name/'collection.json').relative_to(root).as_posix(),
                                         sha256=life.content_sha256(manifest))
    # Hold one per-output mutex through validation and publication. Partial runs
    # remain inspectable; never silently complete/overwrite one after a failure.
    with rt.file_lock(destination.with_name(destination.name+'.pipeline-lock')):
        if destination.exists():
            final = rt.read_json(destination/'pipeline.json')
            need(final['plan_sha256'] == life.content_sha256(plan), 'pipeline input identity differs; use new version')
            need(final['bindings'] == bindings, 'pipeline bindings differ')
            need(final['runtime_sources'] == runtime_sources, 'pipeline implementation changed; use new version')
            actual = inventory(destination); actual.pop('pipeline.json')
            need(actual == final['outputs'], 'pipeline output changed; use new version')
            cs.validate(root, rt.read_json(destination/'screen-spec.json'))
            for name, spec, archive, rh in entries:
                collected = destination/'collections'/name
                life.verify_artifact_receipt(spec, collected/'outputs', rt.read_json(collected/'receipt.json'), expected_receipt_sha256=rh)
            if next_spec:
                life.validate_stage(next_spec, destination/'stage')
                life.verify_evidence(next_spec, rt.read_json(destination/'attestation/verification.json'), destination/'attestation')
            return final
        destination.mkdir(parents=True)
        for name, spec, archive, rh in entries:
            life.collect_artifacts(spec, archive, destination/'collections'/name, expected_receipt_sha256=rh)
        screen_spec = cs.register(root, request, destination/'screen-spec.json')
        cs.screen(root, screen_spec, destination/'screen.json')
        if next_spec:
            life.prepare(next_spec, root, destination/'stage')
            note = attest(root, next_spec, tests, destination/'attestation')
            admission = life.admission_dry_run(next_spec, destination/'stage', note, destination/'attestation')
            put(destination/'admission.json', admission)
        cs.recheck(root, bindings)
        need(all(life.file_sha256(name) == digest for name, digest in runtime_sources.items()),
             'pipeline implementation changed during execution')
        final = dict(schema='cpu-pipeline-result/v1', plan_sha256=life.content_sha256(plan), bindings=bindings,
                     runtime_sources=runtime_sources,
                     outputs=inventory(destination), status='CPU_COMPLETE', holistic_decision='B_DECISION_PENDING',
                     launch_authorized=False)
        put(destination/'pipeline.json', final)
        return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=rt.CODE_ROOT)
    commands = parser.add_subparsers(dest='action', required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--plan', required=True)
    run_parser.add_argument('--out', required=True)
    attest_parser = commands.add_parser('attest')
    attest_parser.add_argument('--spec', required=True)
    attest_parser.add_argument('--tests', required=True)
    attest_parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        if args.action == 'run':
            result = run(args.root, rt.read_json(path(args.root, args.plan)), args.out)
            print(json.dumps({k:result[k] for k in ('status','holistic_decision','launch_authorized')}))
        else:
            spec = life.ArmSpec(path(args.root, args.spec).read_bytes())
            note = attest(args.root, spec, args.tests, path(args.root, args.out))
            print(json.dumps(dict(verdict=note['verdict'], launch_authorized=False)))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ET.ParseError) as exc:
        print('CPU_PIPELINE_REJECTED: '+str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
