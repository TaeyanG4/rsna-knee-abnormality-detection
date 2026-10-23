# H-B-88 OAI header audit / routing preview v1.0.1

Implementation contract for B's test run and the separate astra review. Sources:
H-B-88, D-B-104 and `.local/astra-improve/20261010-1347-par/D.md`.
This is an investigation wrapper in the existing `tools/extcache` package.
`sources.dicom_series`, `sources.fluid_rule`, `common.render_study` and EXP155
remain unchanged because EXP155 binds their code hashes. No cache admission,
training, download, label access or submission is implemented here.

## Changelog

- v1.0.1 (2026-10-10): Open tar archives in seekable `r:*` mode so pydicom can
  rewind while parsing headers on Python 3.12; streaming `r|*` members raised
  `AttributeError` (`_Stream.seekable`) and incorrectly marked valid archives
  `READ_ERROR`. Header-only reads, bounds, quarantine and routing rules are unchanged.

## Inputs and fixed sample

- Only the OAI GENERAL `storage/cache/external/OAI/gen/index.csv` and its exact
  `source_archive_member` paths under `storage/external_data/oai/` are read.
- `oai_hb88_sample.json` pins the index SHA256 and 18 explicit series keys:
  six AX_MPR and six COR_MPR indexed Sagittal, plus three SAG_IW_TSE and three
  COR_IW_TSE with regular **indexed** planes. Regular raw geometry is to be tested.
  Both commands require that pinned index hash. Selection is lexicographic
  within stratum, independent of raw availability.
  All selected discordant MPRs are left knees; this is a diagnostic sample,
  not a representative prevalence estimate.
- An absent raw path is `MISSING_RAW`. The tool never searches for a same-name
  replacement, reconstructs geometry from GENERAL, downloads, or reselects cases.
  A recorded Windows path on Linux is explicitly unavailable; run on the original
  Windows workspace unless a separately reviewed relocation mechanism is added.

## Header and grouping rules

Classic and Enhanced MR are read with pydicom `stop_before_pixels=True` and a
tag whitelist. No `pixel_array`, modality LUT or image rendering is called.
Compressed tar traversal still decompresses the stream to reach later members;
this is not pixel decoding. No member is extracted to disk. Member paths, counts,
size and per-archive frame counts are bounded; symlinks/unsafe names are rejected.
Non-DICOM files or malformed headers quarantine the archive rather than letting
partial contents silently contribute. Errors expose member names and exception
types only, never whole datasets or arbitrary header strings from exceptions.

Every audit frame records member + one-based frame number, SOP/Series/Study UIDs,
IOP, IPP, echo number/time/effective time, laterality, NumberOfFrames, matrix,
spacing, source group and flags. Enhanced per-frame attributes override shared
functional groups and top-level defaults; declared frame counts must agree.
The tool does not try to decode vendor-private/mosaic geometry.

Groups split on Study/Series/FrameOfReference UID, full IOP (rounded to 1e-4),
echo, resolved laterality, matrix, pixel spacing (1e-6), acquisition/time/stack,
image type and DICOM series description. SOP plus frame number is frame identity.
Normals require unit, orthogonal direction vectors within 0.01 and a dominant
axis within 20 degrees; no sequence-name plane fallback. Sort each homogeneous
group along its own normal. Duplicate SOP/frame, positions within 0.001 mm,
single-frame groups, localizers, missing geometry, in-plane IPP drift over 1 mm,
UID absence and side/source conflicts quarantine the group. Gaps are reported
as median/max spacing; they do not by themselves prove mixed series.

Index subject/side/visit must reproduce its study key. Header laterality and
description side must agree where present. Missing DICOM side stays explicitly
`index_only_unconfirmed_by_DICOM`; it is not invented. Visit/subject validation
is an index join check, not independent DICOM patient/date authentication.
Multiple StudyInstanceUIDs quarantine an archive. The same SeriesInstanceUID
in multiple archives of one index study is quarantined in preview to prevent
duplicate counting. No claim of patient identity or OAI/OAIZIB deduplication.

## Proposed routing (preview only)

| Source sequence | Required geometric plane | Proposed fluid | Contrast family |
| --- | --- | --- | --- |
| SAG_IW_TSE | Sagittal | 0 | IW_TSE |
| COR_IW_TSE | Coronal | 0 | IW_TSE |
| COR_MPR | Coronal | 1 | DESS_MPR |
| AX_MPR | Axial | -1 | DESS_MPR |

SAG_IW_TSE fluid0 is the explicit requested proposal, overriding legacy fluid1.
COR_MPR fluid1 is a source-contract proxy, not proof DESS equals fat-suppressed
PD/T2. AX_MPR keeps unknown/non-binary fluid=-1, since the axial contract accepts
any contrast and there is no need to invent an equivalence. Wrong-plane groups
are quarantined, never relabelled to force agreement. An archive may yield more
than one eligible homogeneous group. Rank them by physical span, then frame
count, then stable key; select at most one per slot without any fallback.
This tie-break is frozen before reading results, not a performance selection.

## BEFORE / AFTER interpretation

`preview` visits **every** row of the index. BEFORE calls the original
`pick_series`/`common.pick_strict` with the original CSV order and one used set per
study, exactly matching both existing renderers' metadata-only selection.
The slot order is Sag1, Sag0, Cor1, Cor0, Ax-1; 64 depths 18/14/12/8/12 and
96 depths 26/22/18/12/18 are imported from the existing contracts.

- `BEFORE`: selected-study count per slot from the whole old index.
- `AFTER_HEADER_ELIGIBLE`: strict selection using actual homogeneous raw-header
  groups only, over the same whole-index denominator. Unreadable/missing rows
  contribute zero confirmed groups and remain unresolved. This is a lower
  observed availability count, not a measured loss from correcting preprocessing.
- `AFTER_DECLARED_ONLY`: hypothetical strict policy applied to index sequence
  names over the whole index. It deliberately ignores old plane flags and says
  nothing about the actual geometry or usability of absent raws.

No 18-archive sample is extrapolated into verified whole-index geometry. Counts
are study selections and planned slice positions, not unique native frames,
rendered pixel-valid masks, QA admission or expected AUC. Correct rendering,
laterality, clinical joins, new GENERAL/PRODUCTION versions and H-B-89 provenance
remain later work after this investigation and visual review.

## B execution and review order

The builder supplies code/tests; B runs CPU tests, then a **separate read-only
astra** verifies this exact code/spec/test receipt. Only the PASS hashes run.
Commands below are PowerShell from the workspace root. Use the existing `.venv`
Python (pydicom/numpy/pytest). If that interpreter is unavailable, use B's approved
working **Windows** CPU Python for the same module commands; do not silently
remap raw Windows paths to another host. The module itself forces six numerical-library
thread env vars to 1, uses one sequential worker, and never calls a GPU API.

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_extcache_oai_audit.py
.venv/Scripts/python.exe -m tools.extcache.oai_audit fingerprint
```

The separate reviewer writes a JSON verification note, e.g.
`.local/oai-hb88-review/verification.json`, containing the **actual** review/test
receipt and full fingerprint map (never fabricate PASS or a test result):

```json
{
  "verdict": "PASS",
  "tool_version": "oai-header-audit-v1.0.1",
  "reviewer": "astra",
  "tests_passed": true,
  "test_command": "actual B command and result/log location",
  "code_sha256": {"every fingerprint path": "its exact SHA256"}
}
```

Then B runs the fixed audit, examines missing paths/group flags, and runs the
full preview. New output directories are mandatory; existing ones are refused.

```powershell
.venv/Scripts/python.exe -m tools.extcache.oai_audit audit --verification-note .local/oai-hb88-review/verification.json --out .local/oai-hb88-audit-v1
.venv/Scripts/python.exe -m tools.extcache.oai_audit preview --verification-note .local/oai-hb88-review/verification.json --out .local/oai-hb88-preview-v1
```

Both commands emit `archives.jsonl`, `frames.jsonl` and `summary.json` only in the
requested new artifact directory. Fixed audit emits all 18 archives' frames;
preview stores sorted frame references per group in archives.jsonl and leaves
frames.jsonl empty to avoid duplicating the full header inventory. Every record
includes tool version and verification-note path; the summary additionally binds
code, note, input index and output hashes. Headers bind their canonical parsed
hash and raw size/mtime. Raw archive bytes are not cryptographically authenticated.
An index/code change during the run leaves no completion receipt. Output is local
OAI-derived metadata; retain locally under the registry's data terms.

Exit 0 means all referenced headers were read, **not** every group is admissible.
Exit 2 plus `INCOMPLETE_RAW_EVIDENCE` means missing/unreadable raws or bad archive
headers; per-slot counts still print and missing paths are in summary.json.
Unexpected failures leave no summary completion receipt. Do not treat process
exit alone as validation. Fixes require new B tests and another astra review.

## Deliberate limits / next decision

No cache rewrite/refit is implemented or authorized by this tool. Geometry-only
eligibility cannot prove image quality, correct contrast, pixel-data integrity,
laterality or training value. Thresholds are conservative investigation choices;
inspect rejected groups before relaxing them. If raws were removed by the rolling
cache process, B must obtain the exact archives/headers through the existing
approved data workflow; the wrapper must not guess from old GENERAL metadata.
Only after header evidence and image review should B decide on a new cache
version and whether an EXP155-recipe refit is worth its GPU budget.
