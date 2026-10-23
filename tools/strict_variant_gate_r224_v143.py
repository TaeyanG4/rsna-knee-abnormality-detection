"""Strict variant contract v1.4.3: exact H-B-167 R224 + FLIP_ALL, CPU only.

Changelog 1.4.3: separate immutable contract for exactly knee-lb-r224flipq-v1;
authenticate admitted R224 parent, three exact changes, 28 identical members,
FAST cell and byte-identical historical FLIP_ALL hunk. No inference execution.
Backup: .local/backups/strict-variant-gate-v1.4.3/ (v142 and old consumers).
PIN is a proposal; independent dispatcher-authenticated review is mandatory.
"""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import os
from pathlib import Path
import re
import sys
import time
import zipfile

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import strict_variant_gate as g
from tools import strict_variant_gate_standalone as s

from tools import strict_variant_gate_r224_v142 as previous

VERSION = '1.4.3'
SLUG = 'taeyangg4/knee-lb-r224flipq-v1'
PARENT = '.local/scratch/astra-b5/r224-gate/gate-copy-v142'
STAGE = 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq'
CANONICAL_STAGE = 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq'
PARENT_PINS = {'kernel-metadata.json': '2180101dcf9b5172ef80072afdd995e7c0d3039ab25e57c428af56891910f853', 'release.ipynb': 'ae9b10561a5638f57558868bfdd8a16ce3276c7608c4aa82239fc8fe50a49018'}
CHILD_PINS = {'kernel-metadata.json': '361bc5f487244d178669c0faec6be935207da6db14b0ed72888dea9b7568c078', 'release.ipynb': 'c96f5d0d9ed53c5a57870b97eca4e03fd4226c131fa79489d6bea1d02956c95a'}
PARENT_RECEIPT = '.local/state/gates/knee-lb-r224q-v1.json'
PARENT_RECEIPT_SHA = '25e9c03e00293e4cbb0ac5088afca7875dfaa1d56d9d16b6ff7248d98e69afe3'
FLIP_REFERENCE = 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.22-flipallq'
FLIP_REFERENCE_PINS = {'kernel-metadata.json': '25225232fd209c78763772c366da5f5d13e1ee4e23c49a123bb71c88e27c3015', 'release.ipynb': '2b9dc81818a8adb8a06a77a7c6e83c7957eb8e13af110b3f24f17b49dd8c8c0b'}
FLIP_MEMBER = 'exp169_r7_port.py'
EXACT_CHANGES = {'exp169_r7_port.py': {'before_sha256': 'dd6eae30d742a4f02eb6602ca882e4f9df8b54824e83154b0fac5de4e119f4da', 'after_sha256': '072a754b12a2014fd2ce060da90170e72d916683de1a00dc94dcbd9bf818140d', 'unified_diff': '--- parent/exp169_r7_port.py\n+++ stage/exp169_r7_port.py\n@@ -161,8 +161,7 @@\n         args += ["--compile"]\n     if manifest.get("fluid_stacker"):\n         args += ["--fluid-stacker", str(bundle / manifest["fluid_stacker"])]\n-    if manifest.get("flip_tta"):\n-        args += ["--flip-tta"]\n+    args += ["--flip-tta"]\n     if manifest.get("state_stacker"):\n         args += ["--state-stacker", str(bundle / manifest["state_stacker"])]\n     if manifest.get("views") in ("v3k", "v4"):\n'}, 'exp169_release_c85r7.py': {'before_sha256': '38cd9e84349b6fdb93cada8b657844ba78c63749ba5e23f5027be9d2d5ce7091', 'after_sha256': '033ba0d613ffd524404b4f1a7adc57c8aea1916e4acaeabf61a0da05469caa5c', 'unified_diff': '--- parent/exp169_release_c85r7.py\n+++ stage/exp169_release_c85r7.py\n@@ -43,7 +43,7 @@\n C85_SHA = "291f63d4bc9b79ec7bd06957842d7c2360e352cfd5aad257c64daf003348e27c"\n C85_STRESS_SHA = "4d10508371139e9a9125f96c183db08e079b6502c41ad4ebb4b568b5f5cb42d1"\n C85_REPLAY_CSV_SHA = "a59be5c6066f87b5309e242ecced132fab1894b8dd4b69169e8450a171ddee8a"\n-R7_PORT_SHA = "dd6eae30d742a4f02eb6602ca882e4f9df8b54824e83154b0fac5de4e119f4da"\n+R7_PORT_SHA = "072a754b12a2014fd2ce060da90170e72d916683de1a00dc94dcbd9bf818140d"\n EXP155_ARCHIVE_SHA = "40b8d02847bf36938d82a79e88cb1dad4296db0cc95691b8cd0d2a2ce1a7e5fa"\n DATASETS = [c85.DATASETS[0], c85.DATASETS[1], p.DATASETS[0], r7.DATASET_SLUG, v5q.CODE_SLUG]\n MEMBERS = ["R", "S_R7", "EXP143v2", "EXP155", "Q_A5q"]\n'}, 'exp169_release_c85r7codec.py': {'before_sha256': '4a69d1be66e58a30c68264a506a6e6aee74adfa14cf6bea757b4285cd144d77f', 'after_sha256': '7c06767f497a302978f1668095afb71101d96ef3f608d8c498976e3fcbfa568d', 'unified_diff': '--- parent/exp169_release_c85r7codec.py\n+++ stage/exp169_release_c85r7codec.py\n@@ -49,12 +49,12 @@\n need, sha, read_json, write_once, json_bytes = base.need, base.sha, base.read_json, base.write_once, base.json_bytes\n telemetry, optional_json, competition_root = base.telemetry, base.optional_json, base.competition_root\n ROOT = base.ROOT\n-VERSION = "1.1.23-r224q"\n-OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.23-r224q"\n-KERNEL_ID = "taeyangg4/knee-lb-r224q-v1"\n-RELEASE = "C85R7+codecs R224"\n+VERSION = "1.1.24-r224flipq"\n+OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq"\n+KERNEL_ID = "taeyangg4/knee-lb-r224flipq-v1"\n+RELEASE = "C85R7+codecs R224 FLIP_ALL"\n FORMULA = r224.FORMULA\n-C85R7_SHA = "38cd9e84349b6fdb93cada8b657844ba78c63749ba5e23f5027be9d2d5ce7091"\n+C85R7_SHA = "033ba0d613ffd524404b4f1a7adc57c8aea1916e4acaeabf61a0da05469caa5c"\n CODECS_SHA = "bd463c800196081f2396c8706fa735cdcd5e173eebad8be269169e1bda1dc919"\n C85R7_STAGE = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.0.0"\n DATASETS = [*base.DATASETS, codecs.DATASET_SLUG]\n'}}
IDENTICAL = {'a172-cell.py.txt': '0d0f7a7aaaa94f3182c9cb15500ebe02e641656465261b717278bd60e1559d5d', 'exp152_own0949_release.py': 'ebb8e06734b72f40f090389b23d8767137c8eb6c92dee547367c1f18188f94e2', 'exp169_c85_stress.py': '4d10508371139e9a9125f96c183db08e079b6502c41ad4ebb4b568b5f5cb42d1', 'exp169_partner_bundle_release.py': 'c48aac04128f524116097077a7f1ffd1365f421e3af123c91bb228da09d51665', 'exp169_r224.py': 'bb2e7e2c9bb6dc3e69722eaa3cd59a32f6e281acbf06563300e1701cc93aa95d', 'exp169_r7_src/compact_infer.py': '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d', 'exp169_r7_src/compact_train.py': '04df7e2e154b2c2521696908f272940c63ce17e9b6337b85847fee64504eeaba', 'exp169_r7_src/knee_anchor_bank_v2.py': 'e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309', 'exp169_r7_src/probe_dual.py': 'a6155eb08701f9bb746f838bc8ef8e2129c85f76a5c58e379692ffe2e76bbef0', 'exp169_r7_src/probe_infer.py': 'a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980', 'exp169_r7_src/px_cache.py': 'b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632', 'exp169_r7_src/side_common.py': 'bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566', 'exp169_release_c85.py': '291f63d4bc9b79ec7bd06957842d7c2360e352cfd5aad257c64daf003348e27c', 'exp169_release_p956.py': '7d04817dee31c36a51babf63e1ae2752301875586935d363fd241c9f60e8d901', 'exp169_release_v2.py': 'f526eda7a7e5192c51ecdcaa35b1eff9b4c9f598d4e7bf2d2a5a31c58918d3d4', 'exp169_release_v5q.py': '7e0f005dbd33b06fc30bec2e087a2bb5f4b1ca79f956dd90857e385320add55e', 'exp189_s6_port.py': '6d08301e691e79c0076636ce109739247ec784b300ef7d3081c6e6033da0e6dd', 'exp189_s6_src/compact_infer.py': '34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a', 'exp189_s6_src/compact_train.py': '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80', 'exp189_s6_src/knee_anchor_bank_v2.py': 'e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309', 'exp189_s6_src/probe_dual.py': '9171522ada7ad94145529bb0a47ec48b6f0d8d32d651a60f3f207558649e36b8', 'exp189_s6_src/probe_infer.py': 'a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980', 'exp189_s6_src/px_cache.py': 'b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632', 'exp189_s6_src/side_common.py': 'bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566', 'hb126_codecs.py': 'bd463c800196081f2396c8706fa735cdcd5e173eebad8be269169e1bda1dc919', 'public-serving.py': '25ef6ab9d2e942ac291433a498246d0507481c2aefa5630cdd601d0a36852cf2', 'q-a5q-manifest.json': '39ff0cb9f4e0b4fb0967a00c8c0897a7de90a697242d3042fe3e51b5f2cf36bc', 'raptor224_run.py': 'bc29b23a7aac8ace281619b71c8b49d8c4ac83e645d4e29e457e7e83f477fe2c'}
SIDECAR_SHA = 'f3ae74a1ad08fcc2fc0b08bf37a79a2ff9cf0050e012c0ee5b9c31b82f26bfda'
SOURCE_DIFF = 'full-diff-vs-v1.1.23-r224q.txt'
SOURCE_DIFF_SHA = '5dd5837dc55090513056b8d41b40cdb06e3122ef8157149e9fd04bfb1bc1834a'
SCOPE = 'R224 parent unchanged except exact historical FLIP_ALL R7 flag hunk, chained hashes and version metadata; 3 changed and 28 identical embedded members; FAST identical; R=.65*rank(R384)+.35*rank(R224), failures, survivor behavior and existing launches/pool/thread/timing unchanged'


def member_diff(name, before, after):
    return dict(before_sha256=g.sha(before), after_sha256=g.sha(after),
        unified_diff=''.join(difflib.unified_diff(before.decode().splitlines(True),
            after.decode().splitlines(True), fromfile='parent/'+name, tofile='stage/'+name)))


def admitted_parent(inputs):
    raw = inputs.read(inputs.root / PARENT_RECEIPT)
    g.need(g.sha(raw) == PARENT_RECEIPT_SHA, 'parent admission receipt pin differs')
    record = g.parse(raw)
    g.need(record.get('gate_version') == '1.4.2' and record.get('status') == 'PASS'
        and record.get('errors') == [] and record.get('fixture_only') is False
        and record.get('kernel_id') == previous.SLUG
        and g.local_path(record['stage'], inputs.root) == inputs.root / PARENT,
        'parent admission identity differs')
    g.need(bool(record.get('input_sha256')), 'parent admission has no input hashes')
    for name, digest in record['input_sha256'].items():
        g.need(g.sha(inputs.read(g.local_path(name, inputs.root))) == digest,
               'parent admission input drift: ' + name)
    return record


def fast_inventory(old, new, inputs):
    # Both candidates already have FAST. Validate both against the frozen
    # two-cell base, then compare raw three-cell inventories without stripping.
    base = g.kernel(inputs.root / previous.PARENT, inputs)
    g.need(base[2]['kernel_files'] == previous.PARENT_PINS, 'FAST base pins differ')
    g.fast_inventory(base, old, g.FAST_REFERENCE, inputs)
    g.fast_inventory(base, new, g.FAST_REFERENCE, inputs)
    return new[1]


def flip_equivalence(old, new, inputs):
    reference = g.kernel(inputs.root / FLIP_REFERENCE, inputs)
    g.need(reference[2]['kernel_files'] == FLIP_REFERENCE_PINS, 'flip reference pins differ')
    base = g.kernel(inputs.root / previous.PARENT, inputs)
    g.need(base[2]['kernel_files'] == previous.PARENT_PINS, 'flip base pins differ')
    key = 'embedded/' + FLIP_MEMBER
    current = member_diff(FLIP_MEMBER, old[1][key], new[1][key])
    historical = member_diff(FLIP_MEMBER, base[1][key], reference[1][key])
    g.need(current == historical == EXACT_CHANGES[FLIP_MEMBER], 'historical flip hunk differs byte-for-byte')
    return dict(stage=FLIP_REFERENCE, kernel_files=FLIP_REFERENCE_PINS,
                member=FLIP_MEMBER, exact_diff=historical)


def material(stage, receipt, inputs):
    lane, report, _ = g.dispatch(receipt, inputs)
    g.need(stage == inputs.root / STAGE, 'source stage must be exact registered path')
    admitted_parent(inputs)
    old = g.kernel(inputs.root / PARENT, inputs)
    g.need(old[2]['kernel_files'] == PARENT_PINS, 'R224 parent pins differ')
    new = g.kernel(stage, inputs)
    g.need(new[0]['id'] == SLUG and new[2]['kernel_files'] == CHILD_PINS, 'child kernel pins/slug differ')
    digest = g.sha(g.encoded(new[2]['kernel_files']))
    g.need(re.findall(r'^Kernel-SHA256: (' + g.HEX + r')\s*$', report.decode('utf-8-sig'), re.M) == [digest],
           'completed builder report does not bind this kernel')
    meta = copy.deepcopy(old[0]); meta.update(id=SLUG, title=SLUG.split('/')[1])
    g.need(g.encoded(new[0]) == g.encoded(meta), 'metadata exceeds slug/title change')
    normalized = fast_inventory(old, new, inputs)
    before = {n[9:]:v for n,v in old[1].items() if n.startswith('embedded/')}
    after = {n[9:]:v for n,v in new[1].items() if n.startswith('embedded/')}
    g.need(before.keys() == after.keys() == EXACT_CHANGES.keys() | IDENTICAL.keys(), 'member set differs')
    changed = {n:member_diff(n,before[n],after[n]) for n in before if before[n]!=after[n]}
    same = {n:g.sha(before[n]) for n in before if before[n]==after[n]}
    g.need(changed == EXACT_CHANGES and same == IDENTICAL and len(same) == 28,
           'exact three-member diff or 28 identical members differ')
    flip_equivalence(old, new, inputs)
    raw = inputs.read(stage / 'new-members.json')
    g.need(g.sha(raw) == SIDECAR_SHA, 'new-members pin differs')
    side = g.parse(raw)
    g.need(side['added'] == side['removed'] == [] and side['exact_flip_reference'] is True,
           'new-members added/removed/reference differs')
    g.need(side['changed'] == [dict(name=n, before_sha256=EXACT_CHANGES[n]['before_sha256'],
        after_sha256=EXACT_CHANGES[n]['after_sha256']) for n in sorted(EXACT_CHANGES)]
        and side['unchanged'] == [dict(name=n,sha256=h) for n,h in sorted(IDENTICAL.items())],
        'new-members inventory differs')
    source_diff = inputs.read(stage / SOURCE_DIFF)
    g.need(g.sha(source_diff) == SOURCE_DIFF_SHA, 'builder full-diff pin differs')
    for name, delta in EXACT_CHANGES.items():
        hunk = delta['unified_diff'].splitlines(keepends=True)[2:]
        g.need(''.join(hunk).encode() in source_diff, 'builder full-diff lacks exact hunk: '+name)
    rules, diff, a, b = s.comparison(old[1], (new[0], normalized, *new[2:]), old[3], g)
    return old, new, lane, rules, diff, a, b


def pin_payload(stage, receipt, inputs):
    old, new, lane, rules, diff, _, _ = material(stage, receipt, inputs)
    return dict(version=VERSION, status='REVIEW_REQUIRED', fixture_only=STAGE != CANONICAL_STAGE, kernel_id=SLUG,
        parent=PARENT, parent_kernel_files=PARENT_PINS, source_stage=STAGE,
        parent_receipt=PARENT_RECEIPT, parent_receipt_sha256=PARENT_RECEIPT_SHA,
        builder_receipt=receipt.relative_to(inputs.root).as_posix(), builder_lane=lane,
        builder_receipt_sha256=g.sha(inputs.read(receipt)), kernel_files=new[2]['kernel_files'],
        new_members_sha256=SIDECAR_SHA, source_diff_sha256=SOURCE_DIFF_SHA,
        exact_member_diffs=EXACT_CHANGES, identical_members=IDENTICAL,
        flip_reference=flip_equivalence(old, new, inputs),
        files=rules, diff_sha256=g.sha(diff.encode()), scope=SCOPE, fast_path=dict(g.FAST_REFERENCE),
        tools={Path(tool).name:g.sha(inputs.read(Path(tool))) for tool in (g.__file__,s.__file__,previous.__file__,__file__)}), (old, new, lane, rules, diff)


def write_pin(path, payload):
    """Create-once proposal, including explicitly marked test proposals."""
    g.need(payload.get('status') == 'REVIEW_REQUIRED', 'PIN cannot publish approval')
    data = g.encoded(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for attempt in range(40):
            try:
                if path.exists():
                    g.need(path.read_bytes() == data, 'immutable PIN differs; use a new proposal path')
                else:
                    os.link(temp, path)
                break
            except FileExistsError:
                g.need(path.read_bytes() == data, 'PIN race with different inputs')
                break
            except PermissionError:
                if attempt == 39: raise
                time.sleep(.05)
    finally:
        for attempt in range(40):
            try:
                temp.unlink()
                break
            except PermissionError:
                if attempt == 39: raise
                time.sleep(.05)


def pin(stage, builder_receipt, output, root=g.ROOT):
    inputs = g.Inputs(root)
    stage, receipt, output = [g.local_path(p, root) for p in (stage, builder_receipt, output)]
    payload, _ = pin_payload(stage, receipt, inputs)
    inputs.read(Path(__file__)); inputs.recheck()
    g.need(not output.is_relative_to(root / '.local/state/gates'), 'PIN is not an admission receipt')
    write_pin(output, payload)
    return dict(status='REVIEW_REQUIRED', pin=str(output), pin_sha256=g.sha(g.encoded(payload)))


def authenticated_pin(path, inputs):
    saved = inputs.json(path)
    g.need(saved.get('version') == VERSION and saved.get('fixture_only') is (STAGE != CANONICAL_STAGE) and
           saved.get('status') == 'REVIEW_REQUIRED', 'placeholder/invalid R224 PIN')
    expected, materialized = pin_payload(g.local_path(saved['source_stage'], inputs.root),
        g.local_path(saved['builder_receipt'], inputs.root), inputs)
    g.need(g.encoded(saved) == g.encoded(expected), 'R224 PIN differs from reconstructed completed stage')
    return saved, materialized


def parent_files(old):
    files = {'kernel/' + n:b for n,b in old[3].items()}
    files['registration.json'] = g.encoded(dict(version=VERSION, kernel_id=old[0]['id'], canonical_parent=PARENT))
    files['SHA256SUMS'] = s.sums(files, g)
    return files


def project_review_line(saved):
    return ('R224-Flip-Content: ' + FLIP_MEMBER + ' sha256=' +
        saved['exact_member_diffs'][FLIP_MEMBER]['after_sha256'] +
        '; exact byte-for-byte v1.1.22-flipallq hunk; 3 changed, 28 identical; FAST unchanged')


def bind(pin_path, output, review, root=g.ROOT):
    inputs = g.Inputs(root)
    pin_path, output, review = [g.local_path(p, root) for p in (pin_path, output, review)]
    g.need(not output.exists(), 'bind output exists; use a fresh versioned gate copy')
    saved, (old, new, lane, rules, diff) = authenticated_pin(pin_path, inputs)
    g.need(not output.is_relative_to(inputs.root / STAGE) and not output.is_relative_to(inputs.root / PARENT), 'bind output inside source/parent')
    pfiles = parent_files(old)
    binding = dict(source_stage=STAGE, review=review.relative_to(inputs.root).as_posix(),
        builder_receipt=saved['builder_receipt'], builder_receipt_sha256=saved['builder_receipt_sha256'])
    reg = dict(version='1.1.24-r224flipq', binding_version=VERSION, kernel_id=SLUG, parent=str(output/'parent'),
        parent_sha256sums=g.sha(pfiles['SHA256SUMS']), source_stage=STAGE, kernel_files=new[2]['kernel_files'],
        builder_lane=lane, qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    files = {'kernel/' + n:b for n,b in new[3].items()}
    files.update({'registration.json':g.encoded(reg), 'full-diff.txt':diff.encode(), 'R224-PIN.json':inputs.read(pin_path)})
    files['SHA256SUMS'] = s.sums(files, g)
    allow = dict(version=VERSION, review_binding=binding, parent_sha256sums=reg['parent_sha256sums'],
        stage_sha256sums=g.sha(files['SHA256SUMS']), files=rules, pin_sha256=g.sha(files['R224-PIN.json']), scope=SCOPE)
    files['allowlist.json'] = g.encoded(allow)
    files['review-request.md'] = (f'Independent scope review required; no automatic mathematical approval.\n'
        f'{project_review_line(saved)}\n'
        'Exact three-member before/after SHA256 and unified hunks are in R224-PIN.json.\n'
        'Verify all 28 other parent members identical, historical flip hunk identical, and FAST cell unchanged.\n'
        f'Reviewer-Lane: <independent lane>\nSHA256SUMS: {allow["stage_sha256sums"]}\n'
        f'Allowlist-SHA256: {g.sha(files["allowlist.json"])}\nR224-PIN-SHA256: {allow["pin_sha256"]}\n'
        f'R224-Scope: {SCOPE}\nCheck all 31 parent members, admitted parent receipt and historical flip provenance; formula, failure masks, '
        'survivor denominators, target/UID order and untouched existing launches.\nVERDICT: PENDING\n').encode()
    files.update({'parent/' + n:b for n,b in pfiles.items()})
    inputs.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        target=output/name; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(data)
    return dict(status='REVIEW_REQUIRED', stage=str(output), review=str(review))


def evaluate(stage, review, root=g.ROOT):
    inputs = g.Inputs(root)
    stage, review = [g.local_path(p, root) for p in (stage, review)]
    parent = stage/'parent'
    errors=[]; old=new=None; diff=''; reviewer=None; a=b={}
    try:
        saved, (old,new,lane,rules,diff) = authenticated_pin(stage/'R224-PIN.json', inputs)
        copy_info = g.kernel(stage, inputs)
        g.need(copy_info[3] == new[3], 'gate copy differs from completed source')
        pfiles = parent_files(old)
        for name, data in pfiles.items():
            g.need(inputs.read(parent/name) == data, 'parent copy differs: '+name)
        g.need(g.kernel(parent, inputs)[3] == old[3], 'parent copy kernel set differs')
        pm=g.manifest(parent,inputs,[parent/n for n in pfiles if n!='SHA256SUMS'])[1]
        sm=g.manifest(stage,inputs,copy_info[4]+[stage/n for n in ('registration.json','full-diff.txt','R224-PIN.json')])[1]
        allow=inputs.json(stage/'allowlist.json')
        expected=dict(version=VERSION, review_binding=dict(source_stage=STAGE,
            review=review.relative_to(inputs.root).as_posix(), builder_receipt=saved['builder_receipt'],
            builder_receipt_sha256=saved['builder_receipt_sha256']), parent_sha256sums=pm,
            stage_sha256sums=sm, files=rules, pin_sha256=g.sha(inputs.read(stage/'R224-PIN.json')), scope=SCOPE)
        g.need(g.encoded(allow)==g.encoded(expected), 'R224 allowlist differs')
        reg=inputs.json(stage/'registration.json')
        expected_reg=dict(version='1.1.24-r224flipq', binding_version=VERSION, kernel_id=SLUG, parent=str(parent),
            parent_sha256sums=pm, source_stage=STAGE, kernel_files=new[2]['kernel_files'], builder_lane=lane,
            qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
        g.need(g.encoded(reg)==g.encoded(expected_reg), 'R224 registration differs')
        g.need(inputs.read(stage/'full-diff.txt')==diff.encode(), 'R224 full diff differs')
        reviewer=g.review_check(review,allow,sm,inputs,new,stage/'allowlist.json')
        text=inputs.read(review).decode('utf-8-sig')
        g.need(re.findall(r'^R224-PIN-SHA256: ('+g.HEX+r')\s*$',text,re.M)==[allow['pin_sha256']], 'review does not bind R224 PIN')
        g.need(re.findall(r'^R224-Scope: (.+)$',text,re.M)==[SCOPE], 'independent R224 scope review missing')
        g.need([line for line in text.splitlines() if line.startswith('R224-Flip-Content:')] ==
               [project_review_line(saved)], 'independent flip content review missing or stale')
        normalized=fast_inventory(old,new,inputs)
        _,_,a,b=s.comparison(old[1],(new[0],normalized,*new[2:]),old[3],g)
    except (ValueError,OSError,KeyError,TypeError,SyntaxError,IndexError,zipfile.BadZipFile) as exc:
        errors.append(str(exc))
    for tool in (g.__file__,s.__file__,previous.__file__,__file__): inputs.read(Path(tool))
    try: inputs.recheck()
    except (ValueError,OSError) as exc: errors.append(str(exc))
    return dict(gate_version=VERSION,fixture_only=STAGE != CANONICAL_STAGE,status='FAIL' if errors else 'PASS',errors=sorted(set(errors)),
        stage=str(stage),parent=str(parent),kernel_id=new[0]['id'] if new else None,reviewer_lane=reviewer,
        input_sha256=inputs.hashes(),parent_inventory=old[2] if old else None,stage_inventory=new[2] if new else None,
        comparison_sha256=dict(parent={n:g.sha(v) for n,v in a.items()},stage={n:g.sha(v) for n,v in b.items()}),
        full_diff=diff,diff_sha256=g.sha(diff.encode()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    cmd=commands.add_parser('pin'); cmd.add_argument('--stage',default=STAGE)
    cmd.add_argument('--builder-receipt',required=True); cmd.add_argument('--output',required=True)
    cmd=commands.add_parser('bind'); cmd.add_argument('--pin',required=True)
    cmd.add_argument('--output',required=True); cmd.add_argument('--review',required=True)
    cmd=commands.add_parser('evaluate'); cmd.add_argument('--stage',required=True)
    cmd.add_argument('--review',required=True); cmd.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='pin': result=pin(args.stage,args.builder_receipt,args.output)
        elif args.command=='bind': result=bind(args.pin,args.output,args.review)
        else:
            result=evaluate(args.stage,args.review)
            if result['status']=='PASS' and not args.check_only:
                g.need(result['fixture_only'] is False, 'fixture-only gate cannot publish admission')
                g.write_receipt(g.ROOT/'.local/state/gates'/ (SLUG.split('/')[1]+'.json'),result)
        print(g.encoded({k:v for k,v in result.items() if k!='full_diff'}).decode())
        return 2 if result.get('status')=='FAIL' else 0
    except (ValueError,OSError,KeyError,TypeError,SyntaxError,IndexError,zipfile.BadZipFile) as exc:
        print('REFUSED: '+str(exc)); return 2


if __name__=='__main__':
    raise SystemExit(main())
