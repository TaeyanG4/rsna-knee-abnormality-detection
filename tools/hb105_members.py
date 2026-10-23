"""H-B-105 production member operations v1.0.0 (2026-10-12).

Changelog: immutable control-free fold specs, final-only output audit,
receipt-pinned export/collect, offline package. No launcher or GPU probe.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import tarfile
from training.arms import lifecycle as life, mil_small_members as mil

ROOT = Path(__file__).resolve().parents[1]
EXP = 'artifacts/experiments/HB105_CONVNEXT_S_MIL_MEMBERS'
READOUTS = ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv')
FITS = tuple((s, f) for s in (42, 43) for f in (1, 2, 3, 4))
read, sha, save, need = mil.read, mil.sha, mil.write_new, mil.need


def config_path(seed, fold):
    need((seed, fold) in FITS, 'unsupported member')
    return f'training/configs/hb105-member-s{seed}-f{fold}-v1.json'


def spec_for(root, seed, fold):
    return life.ArmSpec((Path(root) / config_path(seed, fold)).read_bytes())


def configure(root):
    root = Path(root)
    results = []
    for seed, fold in FITS:
        recipe = f'training/configs/hb105-small-s{seed}-f0-v1.json'
        original = life.ArmSpec((root / recipe).read_bytes())
        need(original.sha256 == mil.RECIPE_HASHES[seed], 'H-B-105 recipe changed')
        # The production Tiny members define the exact EXP099 order/weights.
        doc = life.ArmSpec((root / f'training/configs/mil-member-s43-f{fold}-v1.json').read_bytes()).document
        doc.update(arm=f'HB105_MEMBER_S{seed}_F{fold}', seed=seed, attempt='v1')
        doc['config']['model'] = original.document['config']['model']
        doc['inputs']['data'] = original.document['inputs']['data']
        doc['inputs']['sources'] = {p: sha(root / p) for p in mil.SOURCE_FILES}
        target = 'training/configs/mil-target-3090-v1.json'
        # A fold-specific recipe document is declarative provenance only. Its
        # f0 ancestor is additionally bound in the executing source inventory.
        doc['inputs']['evidence'] = {target: sha(root / target),
            f'training/configs/mil-member-s43-f{fold}-v1.json': sha(root / f'training/configs/mil-member-s43-f{fold}-v1.json')}
        derived = f'training/configs/hb105-recipe-s{seed}-f{fold}-v1.json'
        life.register(doc, root / derived)
        doc['inputs']['evidence'] = {derived: sha(root / derived), target: sha(root / target)}
        spec = life.ArmSpec.from_document(doc)
        mil.validate_data(spec, root, root / doc['config']['roots']['corpus'])
        life.register(doc, root / config_path(seed, fold))
        results.append(dict(seed=seed, fold=fold, spec_sha256=spec.sha256))
    return dict(status='CONFIGURED_NO_EXECUTION', specs=results)


def combine(spec, cpu, target, dest, reviewer_name, reviewer_host):
    """Explicit independent-review attestation over two actual HOLD smokes.

    B invokes after inspecting both smokes. The CPU builder never issues PASS.
    Hardlinks avoid duplicate evidence checkpoint storage on the same volume.
    """
    original = {}
    for kind, root in (("cpu", cpu), ("target", target)):
        note = read(root / "verification.json")
        life.verify_evidence(spec, note, root)
        mil.need(note["verdict"] == "HOLD" and note["smokes"][kind] is not None
                 and note["smokes"]["target" if kind == "cpu" else "cpu"] is None, "two original smokes required")
        original[kind] = note
    mil.need(bool(reviewer_name.strip()) and bool(reviewer_host.strip()), "reviewer identity required")
    dest.mkdir(parents=True, exist_ok=False)
    checks, smokes = [], {}
    for kind, root in (("cpu", cpu), ("target", target)):
        note = original[kind]
        for record, role in ((note["checks"][0], "check"), (note["smokes"][kind], "smoke")):
            log = read(root / "evidence" / (record["log_sha256"] + ".json"))
            for artifact in log["artifacts"].values():
                member = kind + "/" + artifact["path"]
                destination = dest / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    try:
                        os.link(root / artifact["path"], destination)
                    except OSError:
                        shutil.copyfile(root / artifact["path"], destination)
                mil.need(sha(destination) == artifact["sha256"], "evidence copy differs")
                artifact["path"] = member
            digest = life.content_sha256(log)
            save(dest / "evidence" / (digest + ".json"), log)
            updated = dict(record, log_sha256=digest)
            if role == "check":
                updated["name"] = kind + "-executed-family-smoke"
                checks.append(updated)
            else:
                smokes[kind] = updated
    merged = {"schema": life.VERIFICATION_SCHEMA, "scope": spec.scope,
        "reviewer": {"name": reviewer_name, "host": reviewer_host}, "verdict": "PASS", "blockers": [],
        "notes": ["Independent reviewer attests inspection of actual CPU and target evidence."],
        "bindings": spec.bindings, "checks": checks, "smokes": smokes}
    life.verify_evidence(spec, merged, dest)
    save(dest / "verification-pass.json", merged)
    return {"status": "PASS_EVIDENCE_MERGED", "verification_sha256": life.content_sha256(merged)}


def check_outputs(spec, output, *, smoke=False):
    output = Path(output)
    summary, prov, history = [read(output / n) for n in ('summary.json', 'provenance.json', 'history.json')]
    c, d = spec.document['config'], spec.document
    need(summary['spec_sha256'] == prov['spec_sha256'] == spec.sha256, 'output spec differs')
    need(summary['status'] == ('ENGINEERING_SMOKE' if smoke else 'FIXED_FINAL_COMPLETE'), 'output mode differs')
    need(summary['run_kind'] == prov['run_kind'] == 'member', 'member run kind required')
    epochs = 1 if smoke else 20
    need(summary['epochs'] == epochs and len(history) == epochs
         and [h['epoch'] for h in history] == list(range(epochs)), 'epoch history differs')
    need(summary['attempted_updates'] == (1 if smoke else c['updates']['steps'])
         and summary['skipped_updates'] <= (0 if smoke else c['updates']['max_skips']), 'updates differ')
    need(prov['checkpoint_selection'] == 'fixed-final' and prov['metric_access'] == 'disabled', 'selection differs')
    if not smoke:
        need(summary['cpu_cuda_initialized'] is None and prov['verification_sha256'] is not None, 'CUDA/PASS missing')
        need((summary['train'], summary['val']) == (c['roots']['n_train'], c['roots']['n_val']), 'roster sizes differ')
        need(prov['train_ids_ordered_sha256'] == c['roots']['train_sha256']
             and prov['eval_ids_ordered_sha256'] == c['roots']['val_sha256'], 'OOF provenance differs')
    need(not (output.parent / 'epochs').exists(), 'resume checkpoints forbidden')
    need({p.name for p in output.iterdir()} == set(mil.OUTPUTS), 'unexpected output files')
    inventory = read(output / 'output-inventory.json')
    need(set(inventory) == set(mil.OUTPUTS) - {'output-inventory.json'}, 'inventory differs')
    for p, r in inventory.items():
        need(sha(output / p) == r['sha256'] and (output / p).stat().st_size == r['bytes'], 'output integrity differs')
    checkpoint = mil.torch.load(output / 'model_best.pt', map_location='cpu', weights_only=False)
    need(checkpoint['fold'] == d['fold'] and checkpoint['tag'] == d['arm']
         and checkpoint['epoch'] == epochs - 1 and checkpoint['arch'] == mil.ARCH
         and checkpoint['provenance'] == prov, 'checkpoint metadata differs')
    return summary


def finish(spec, output, note_path, archive, receipt_path, *, smoke=False):
    note_path = Path(note_path)
    note = read(note_path)
    life.verify_evidence(spec, note, note_path.parent)
    if not smoke:
        life.require_verification_pass(note, **spec.verification_args)
    summary = check_outputs(spec, output, smoke=smoke)
    if not smoke:
        need(read(Path(output) / 'provenance.json')['verification_sha256'] == life.content_sha256(note), 'run evidence differs')
    receipt = life.artifact_receipt(spec, output, verification_sha256=life.content_sha256(note), exit_code=0,
        elapsed_seconds=summary['elapsed_seconds'], metrics=dict(attempted_updates=summary['attempted_updates'],
        val_rows=summary['val'], engineering_smoke=int(smoke)))
    save(receipt_path, receipt)
    digest = life.content_sha256(receipt)
    archive_digest = life.export_artifacts(spec, output, receipt, archive, expected_receipt_sha256=digest)
    return dict(status='EXPORTED_ENGINEERING_SMOKE' if smoke else 'EXPORTED_ARTIFACTS',
                receipt_sha256=digest, archive_sha256=archive_digest)


def collect(spec, archive, export_log, dest):
    record = read(export_log)
    need(sha(archive) == record['archive_sha256'], 'transferred archive differs')
    return life.collect_artifacts(spec, archive, dest, expected_receipt_sha256=record['receipt_sha256'])


def package(root, out, archive, cpu_root):
    root, out, cpu_root = Path(root), Path(out), Path(cpu_root)
    files = set()
    for seed, fold in FITS:
        spec = spec_for(root, seed, fold)
        life.validate_inputs(spec, root)
        files.update(spec.inventory)
        files.add(config_path(seed, fold))
    files.update([EXP + '/LAUNCH.md', EXP + '/queue.sh', EXP + '/cpu_card.py',
                  'tools/hb105_members_readout.py', 'tests/test_hb105_members.py'])
    out.mkdir(parents=True, exist_ok=False)
    for name in sorted(files):
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
    for seed, fold in FITS:
        source = cpu_root / f's{seed}-f{fold}' / 'cpu'
        note = read(source / 'verification.json')
        life.verify_evidence(spec_for(root, seed, fold), note, source)
        need(note['verdict'] == 'HOLD' and note['smokes']['cpu'] is not None, 'CPU smoke required')
        for p in source.rglob('*'):
            if p.is_file():
                name = f'cpu/s{seed}-f{fold}/' + p.relative_to(source).as_posix()
                dest = out / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
                files.add(name)
    save(out / 'package.json', dict(schema='hb105-members-package/v1', fits=FITS,
         files={p: sha(out / p) for p in sorted(files)}))
    with tarfile.open(archive, 'x:gz') as tar:
        for p in sorted(files | {'package.json'}):
            tar.add(out / p, arcname=p, recursive=False)
    return dict(**verify_package(out), archive_sha256=sha(archive), bytes=Path(archive).stat().st_size)


def verify_package(root):
    root = Path(root)
    manifest = read(root / 'package.json')
    need(manifest['schema'] == 'hb105-members-package/v1'
         and manifest['fits'] == [list(x) for x in FITS], 'package schema differs')
    life._verify_files(root, manifest['files'])
    for seed, fold in FITS:
        life.validate_inputs(spec_for(root, seed, fold), root)
    return dict(status='PACKAGE_VERIFIED', files=len(manifest['files']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('configure'); sub.add_parser('verify-package')
    p = sub.add_parser('package')
    for n in ('out', 'archive', 'cpu-root'):
        p.add_argument('--' + n, type=Path, required=True)
    for command in ('combine', 'finish', 'collect'):
        p = sub.add_parser(command)
        p.add_argument('--seed', type=int, required=True); p.add_argument('--fold', type=int, required=True)
        names = dict(combine=('cpu','target','out'), finish=('outputs','note','archive','receipt'),
                     collect=('archive','export-log','out'))[command]
        for n in names:
            p.add_argument('--' + n, type=Path, required=True)
        if command == 'combine':
            p.add_argument('--reviewer-name', required=True); p.add_argument('--reviewer-host', required=True)
        if command == 'finish':
            p.add_argument('--engineering-smoke', action='store_true')
    a = ap.parse_args()
    if a.command == 'configure': result = configure(a.root)
    elif a.command == 'verify-package': result = verify_package(a.root)
    elif a.command == 'package': result = package(a.root,a.out,a.archive,a.cpu_root)
    else:
        spec = spec_for(a.root,a.seed,a.fold)
        if a.command == 'combine': result = combine(spec,a.cpu,a.target,a.out,a.reviewer_name,a.reviewer_host)
        elif a.command == 'finish': result = finish(spec,a.outputs,a.note,a.archive,a.receipt,smoke=a.engineering_smoke)
        else: result = collect(spec,a.archive,a.export_log,a.out)
    print(life.canonical_bytes(result).decode())


if __name__ == '__main__':
    main()
