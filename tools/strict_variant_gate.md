# strict_variant_gate 1.3.0

Changelog 1.3.0: add exact public-R/R-KD and HB151/ConvNeXt-S L2 parent
contracts in `strict_variant_gate_standalone.py`. Consumer: submit_once v1.3.4.
Backups: `.local/backups/strict-variant-gate-v1.2.0/`. Existing family protocol
remains 1.2.0; their tests and pins are unchanged. Historical receipt files are
not rewritten: source-hash freshness still requires revalidation after updates.

## Standalone families added in 1.3.0

Only these new child IDs select the additive contract:

- `taeyangg4/knee-hb157-rkd-standalone-v1`: public R notebook
  `storage/external/nartaa__rsna-knee-0949-anatomical-mirror/rsna-knee-0949-anatomical-mirror.ipynb`,
  SHA256 `8f1f96530b08543bd6caeed48489eea7c01657d6994f22e21a8400047fd4e100`.
  Allow only registered checkpoint/mount replacements and the exact reviewed
  own25q3 FAST cell plus guards. Parent metadata is a fixed **synthetic provenance
  literal**, not historical upload metadata. Its SHA256 is
  `b42f0cb371d4a970074eee5cd58ec75992c264c3e1ad25a5994b9e9e97757ef1`.
- `taeyangg4/knee-hb156-cnxs-l2-v1`: pinned HB151 v1.0.2 notebook, metadata,
  build receipt, dataset metadata and four-member archive. Allow exactly the
  ConvNeXt-S ARCH and constructor substitutions plus checkpoint, identifiers and
  manifest-chain pins. The independent ConvNeXt schema pin is
  `2c18c2b48bc57546931f144ccd59c17140e5b0762f5f56b61426122c19a8f2da`.

The gate reconstructs every notebook field/cell, runtime and manifest exactly;
canonical JSON comparison preserves boolean/integer/float distinctions.
regex widening cannot authorize extra bytes. Full diff covers every inventory
entry (identity and decoded diff), all archive members (binary checkpoint SHA),
dataset metadata and raw notebook/metadata. Runtime/import/process/pool/thread/
timing changes are refused. Dataset file sets are exact and snapshotted. No
notebook, model or pickle is executed. ZIP pickle opcodes are inspected only for
fixture markers. Placeholder flags, hashes and fixture markers are refused even
for check-only. Training lineage/tensor compatibility remain candidate-review duties.

`bind --fast` accepts only the canonical parent and creates a fresh gate copy
with nested parent, full-diff.txt, registration, SHA256SUMS, exact allowlist and
review-request.md. The dispatched builder report binds `Kernel-SHA256`. A different
dispatched reviewer checks complete diff/provenance/CPU evidence and ends with
Reviewer-Lane, SHA256SUMS, Allowlist-SHA256 and VERDICT: PASS (one line each).
No approval is synthesized. Re-evaluation rechecks every input; consumer requires
the main gate and both helper hashes and exact child/parent pair. New pins do not
widen legacy receipts. CPU commands and gate-copy names:
`.local/scratch/strict-gate-v130/strict-variant-gate/LAUNCH.md`.
The test runner redirects only the old tests' large scratch variable; old test
files remain byte-identical. Fixed basetemp: `.local/pytest-tmp/strict-gate-v130`.

Changelog 1.2.0: exact EXP143v2/HB151 external-archive contract, nested registered
parent copy, source/version/parent bindings, and full reviewed diff. Fixture-only
validation cannot publish admission. Consumer compatibility: submit_once v1.3.3.
Changed originals are backed up in `.local/backups/strict-variant-gate-v1.1.0/`.

Changelog 1.1.0: exact reviewed FAST cell/guard exception; dispatcher-derived
builder/reviewer identities; create-only `bind` helper; offline `--check-only`.
1.0.0 introduced exhaustive embedded-file/cell/metadata comparison and receipts.
Original files backed up under `.local/backups/strict-variant-gate-v1.0.0/`.
No scored stage or existing gate receipt is modified by this update.

## B's workflow

Run from the project root. `bind` creates a NEW directory containing a byte-for-byte
kernel copy, registration.json, SHA256SUMS, allowlist.json and review-request.md.
The output must not exist. This is a provenance binding copy, not a runtime rebuild;
registration preserves the source runtime version and records binding_version=1.2.0.
It never manufactures a PASS review. Never edit a scored source stage to add bindings.

```powershell
python tools/strict_variant_gate.py bind --stage <source-stage> --parent artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0 --output <fresh-versioned-copy> --builder-receipt <build-out.md.receipt.json> --review <independent-review.md> --fast
python tools/strict_variant_gate.py --stage <fresh-versioned-copy> --parent artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0 --allow <fresh-versioned-copy>/allowlist.json --review <independent-review.md>
```

Omit `--fast` for a normal two-cell candidate. The helper drafts exact changed-file
hashes and escaped full-line patterns. For a NEW stage it returns REVIEW_REQUIRED.
A separately dispatched reviewer must inspect the complete decoded/raw diff and
allowlist before producing these declarations in the exact selected review file:

```text
Reviewer-Lane: <actual independent dispatcher lane>
SHA256SUMS: <hash from review-request.md>
Allowlist-SHA256: <hash from review-request.md>
VERDICT: PASS
```

The verdict must be the unique final nonblank verdict line. The reviewer dispatcher
must publish `<review.md>.receipt.json`. The builder's dispatched report must include
`Kernel-SHA256: <digest>`: SHA256 of the UTF-8, sorted-key, indent-2 JSON mapping of
ALL kernel-relative file names to their raw SHA256 values, with a final newline
(`sha(encoded(kernel(stage, Inputs(ROOT))[2]['kernel_files']))` in this module).
Generate this declaration before dispatch completes; do not alter archived reports.
Registration/manifest generation does not certify scientific qualification:
qualified_for_submission stays false, exposure is EXPOSED, clean_oof is false.

Builder and reviewer lanes come from matching successful dispatch-terminal/v1
receipts in `.local/astra-improve/lanes/runs/<lane>/<run_id>/receipt.json` with
matching archived report and prompt hashes. Caller builder_lane/builder_scratch
fields are rejected. Review must be outside the derived builder scratch directory;
generic review dispatch must be from a different lane. Dispatcher archives are the
local trust boundary, not signatures against an attacker who can rewrite that archive.

## Exact historical migration

B explicitly requested reuse of the real own25q3 and p65q2 reviews. Their old reports
lack manifest declarations and reviewer dispatch receipts. Only those TWO exact
source paths + complete kernel byte hashes + builder dispatch receipt hashes +
original review paths/bytes are migration anchors in the gate. Their response is
BOUND. The binding copy supplies the missing registration/manifest; the original
reviews remain unchanged. `reviewer_lane=slot-b` denotes this historical pairing,
not newly verified reviewer authorship. A copied/edited review or changed kernel
cannot inherit this exception. Other stages require the independent declarations above.

Use source `artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.18-own25q3` with
`.local/scratch/slot-b/presub/own25q3-build-out.md.receipt.json` and
`.local/scratch/slot-b/presub/own25q3-review-out.md`;
for `v1.1.19-p65q2`, use corresponding `p65q2-build-out.md.receipt.json` and
`p65q2-review-out.md`. The smoke script contains complete executable commands.

## Allowlist and comparison contract

The v1.1.0 allowlist requires version, parent_sha256sums, stage_sha256sums, files,
and review_binding. The latter contains source_stage, review, builder_receipt,
builder_receipt_sha256. Every changed file rule requires before_sha256,
after_sha256, removed and added (full-line regex arrays). SHA256SUMS means the hash
of the manifest bytes. All kernel files and registration must be covered by it.
Ancillary historical manifest entries are not read or certified. Both kernels are
fully decoded, every embedded PINS digest verified, all cells/fields/top metadata
compared, and raw notebook diff retained. Added/deleted/renamed files remain forbidden.
Title must equal the id slug and have no slash; registration kernel_id must equal id.
Docker image, dataset_sources, machine_shape and enable_internet must match parent.

Optional fast_path must exactly equal the module's FAST_REFERENCE, including:
- source stage v1.1.18-own25q3 and original own25q3 review path/hash;
- cell0_sha256: 14e0f93fb67c435c49966b891955e7b2bf721b012dd72c64c3482d068a212ef2;
- cell0_object_sha256: 8c206ac888ba4a9be5f8244c473b03226c17652d04e733c4caa933467df53ddd.

The candidate must have precisely this cell0, unchanged parent markdown, and the
parent run cell prefixed by exactly `if not FAST:\n` with four spaces on EVERY
remaining line. After removing that wrapper, notebook fields/top and executable
source must match parent, apart from verified BLOB/PINS literals and archive hash.
No extra executable line, arbitrary new cell, metadata alteration or FAST reference
is admitted. Embedded changes still need explicit file/hash/line rules. Complete
raw hashes and raw diff preserve cell0 and wrapper evidence in the gate receipt.

## Receipt and scope

Normal success publishes `.local/state/gates/<slug>.json` with gate_version=1.2.0,
status=PASS, errors=[], stage, parent, kernel_id, reviewer_lane, input_sha256,
parent/stage inventories, comparison hashes and full diff/hash. All read inputs
(including dispatcher archives, historical references, manifests, review, allowlist
and gate source) are hashed and rechecked before success. Publication is create-once,
atomic, with bounded Windows lock retries. Identical reuse is allowed; different
inputs at an occupied slug fail. Use `--check-only` for a full offline evaluation
without publication. Every nonzero exit blocks pushing. B must rerun immediately
before EVERY push; an old receipt's existence alone is never approval.

The gate never calls Kaggle or executes notebook code. It does not automate push.
The v1.2.0 producer requires submit_once v1.3.3, which accepts 1.0/1.1/1.2 receipts
and rejects fixture-only admission. One-attempt, ledger, lock, daily-cap and
prediction fallback checks remain unchanged.

## Exact HB151 external-archive exception

Only `taeyangg4/knee-hb151-pl-coat-v1` v1.0.2 from the pinned EXP143v2 standalone
`taeyangg4/knee-exp152-own0949-v2` (submission ref 57005140) is supported. Run the
existing HB151 `bind --source seed43` after the real export arrives. Then run gate
`bind` with parent `.local/exp152-own0949-v1/stage`; it creates a new binding stage
and a nested `parent/` registration/manifest. Both kernel copies preserve bytes.
The parent's non-executable DESCRIPTION.md is separately hash-pinned and excluded
from the two-file kernel copy. No existing source, stage or receipt is overwritten.

The version-1.2.0 `external_archive` allowlist entry must match the derived policy:
parent kernel path + SHA, ref 57005140, exact dataset slug
`taeyangg4/knee-hb151-pl-coat-ckpt-v1`, checkpoint SHA from the bind receipt,
archive SHA, manifest SHA, bind/export receipt paths + SHAs, and fixture status.
Both archives must contain exactly the same four named members. Every member is
hashed; public-serving.py is byte-identical, the runtime admits only three exact
checkpoint-provenance substitutions, and manifest changes must equal the derived
checkpoint/protocol/provenance delta. No arbitrary dataset or executable change
can be authorized by broadening line rules. Large archives are streamed and
rehashed before success. This validates local dataset content; B still publishes
and verifies the actual private dataset and visible notebook version.

FAST must be the same reviewed own25q3 cell object and source hashes above. The
parent's four markdown cells are unchanged; its run cell changes only the dataset
slug, chained pins and exact `if not FAST:` indentation guard. Exactly 5 to 6 cells
is required. Other metadata is unchanged except child id/title. The generated
`full-diff.txt` contains every normalized textual and raw notebook change plus
checkpoint SHA change; it is manifest-covered and must match the recomputed diff.
No executable checkpoint data is deserialized by this gate.

The new binding's builder evidence must authenticate the actual bound source's
Kernel-SHA256, followed by a separate dispatched review bound to the generated
manifest and allowlist. Old pre-export builder evidence cannot certify a later
checkpoint. `bind` returns REVIEW_REQUIRED; it never writes a PASS review.
Fixture-checkpoint check-only evaluation may PASS, but fixture gate publication
and consumer admission are refused. Actual qualification still requires the
real export, independent review and actual COMPLETE visible receipt.

Exact commands and prerequisites: `.local/scratch/astra-b5/strict-variant-gate-v120/LAUNCH.md`.

## CPU LAUNCH / validation

```powershell
python -m pytest tests/test_strict_variant_gate.py tests/test_strict_variant_gate_hb151.py tests/test_kaggle_submit_once.py tests/test_c85r7_fluidr_v119.py tests/test_c85r7_own25_v1110.py tests/test_c85r7_own25q_v1111.py tests/test_c85r7_own25q3_v1118.py tests/test_c85r7_p65q2_v1119.py --basetemp=.local/pytest-tmp/astra-b5 -q
python tools/strict_variant_gate_smoke.py
```

The smoke unpacks the two real release code artifacts (29 files each), uses their
original reviews, and checks PASS plus cell0/title/runtime-rename failures per stage.
It runs no inference and reads no labels or radiology. Fixed scratch:
`.local/scratch/astra-b5/strict-variant-gate-v120/c85-cli/`. The CLI bind/check-only
path and receipt publication are also exercised by tests, under the fixed pytest
base. No GPU/register/prepare/admit launch commands apply to this static gate.
