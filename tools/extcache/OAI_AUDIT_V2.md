# H-B-88 OAI full audit v2.0.4

Spec/registration: `artifacts/experiments/HB88_OAI_HEADER_AUDIT_V2/PROTOCOL.md`.
This extends frozen `oai_audit.py` through `oai_audit_v2.py`; v1 and its original
incorrect Sag0 preview remain unchanged historical artifacts. Use **v2 only** for
the corrected preview. `fingerprint` reads code/input hashes, not raw headers.

PowerShell, workspace root; the existing virtualenv launcher references a missing
Python312 installation. This existing uv-managed interpreter can use its packages
without changing the virtualenv or installing anything:

```powershell
$cpuPython = 'C:\Users\Taeyang\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) '.venv/Lib/site-packages')
$env:PYTHONDONTWRITEBYTECODE = '1'
& $cpuPython -B -m pytest -q -p no:cacheprovider tests/test_extcache_oai_audit.py tests/test_extcache_oai_audit_v2.py
& $cpuPython -B -m tools.extcache.oai_audit_v2 fingerprint
```

After tests, send protocol/code/tests/test receipt to a **separate read-only reviewer**
(allow-list `REVIEWERS`: `astra`, or `claude-opus` in a session other than the
implementer's) with `.local/astra-improve/VERIFY-RULES.md`. It must inspect requirements, not trust
this summary. B confirms each claimed defect and spot-checks an OK item on PASS.
Only then create the review note (actual evidence, never a fabricated PASS):

```json
{
  "verdict": "PASS",
  "reviewer": "astra or claude-opus (exactly one string)",
  "tool_version": "oai-header-audit-v2.0.4",
  "tests_passed": true,
  "test_command": "actual command + result/log",
  "review_report": "path of independent report",
  "code_sha256": {"every fingerprint path": "exact hash"},
  "input_sha256": {"every input path": "exact hash"}
}
```

Actual full command (new output directory required):

```powershell
& $cpuPython -B -m tools.extcache.oai_audit_v2 audit-all --verification-note .local/oai-hb88-v2-review/verification.json --out artifacts/experiments/HB88_OAI_HEADER_AUDIT_V2/run-v1
```

One sequential CPU worker, numerical library threads 1, GPU 0 h; <=3 wall-hour cap.
Do not run a second audit concurrently. Check CPU/RAM/I/O and the shared resource
claim first. Gzip traversal decompresses data to reach headers; no DICOM pixels are
interpreted, decoded, rendered or cached. Partial outputs may be large.

Outputs: `archive-progress.jsonl` (pre-cross-archive decisions; not final),
`archives.jsonl` (final per-archive receipts and ordered frame references),
`routing.jsonl` (all retained groups and eligibility/quarantine reasons),
`slot-selections.jsonl` (both legacy/corrected choices), `summary.json` (completion,
BEFORE/AFTER counts, source/code/review/output hashes, resource measurements).
Every record has version and verification note. Original and relocated paths are
bound; the unindexed archive is only inventoried. No raw-byte hash claim: archive
size/mtime and parsed-header hashes are recorded. All data stay local.

Read `status`, `raw_statuses`, `join_issues`, `routing_issues`, exclusions and counts.
`COMPLETE_HEADERS` means every indexed archive's headers were read; quarantined
groups can still exist. It is **not** a cache/label admission. Exit 2 means raw
evidence is incomplete. An exception, timeout or changed input leaves no summary.
MOAKS clinical scores are not inspected: duplicate ID/SIDE/READPRJ/VERSION identities
are quarantined conservatively, and score disagreement certification stays open.

## Changelog

### v2.0.4 — 2026-10-10 (Claude opus for B)

- Codex/astra is unavailable. `verification()` now accepts any reviewer in the
  allow-list `REVIEWERS = ("astra", "claude-opus")`. Every other gate is unchanged:
  verdict PASS, tool_version == VERSION, tests_passed true, a test command, a review
  report, and code/input hashes bound exactly. The reviewer is stamped on every
  output record and in summary.json. The code cannot check that a claude-opus
  reviewer is independent of the implementer session; that stays procedural.
- Test: both allow-listed reviewers are accepted. These are refused: an unknown,
  case-changed, empty, non-string or missing reviewer; a FAIL verdict; an old
  tool_version; stale code or input hashes; a missing test receipt.
- PROTOCOL.md amendment A2 records the reviewer allow-list. Notes for v2.0.3 do
  not authorize v2.0.4.

### v2.0.3 — 2026-10-10 (Claude opus for B)

- D4: `irregular_ipp_spacing`, `ordered_positions_mm`, archive/instance IPP order
  and the IOP/pixel-spacing homogeneity checks now use the canonical reference
  frame (minimum `frame_sort_key`, the frame v1 records `iop`, position range and
  drift from) instead of the first spatial frame. A within-rounding IOP tilt on
  the first spatial slice can no longer flip quarantine in either direction.
  A mismatch between that frame and the recorded iop/pixel spacing fails closed
  (`geometry_reference_frame_mismatch`).
- Serialized regressions: missed quarantine (0.05002 mm) and inverse false
  rejection, tilted frame first/middle/last, all frame orders. D1-D3 unchanged.
- v2.0.2 verification notes do not authorize v2.0.3; a new independent PASS is
  required before the full audit.

### v2.0.2 — 2026-10-10

- D3: Canonically sort parsed frames by IOP, identity and header content before
  calling the frozen v1 geometry helper. Its reference normal, IPP order and
  in-plane drift now depend on header content rather than input enumeration.
  The parsed-header hash uses this order; the archive IPP diagnostic uses
  member-name/frame order, which is stable under tar enumeration changes.
- A serialized DICOM regression permutes the same near-threshold frame set and
  requires identical group receipts and routing eligibility for every order.
  D1 and D2 remain covered by their serialized regressions; frozen v1 is unchanged.
- v2.0.1 verification notes do not authorize v2.0.2. Obtain a new independent
  PASS bound to the current files before the full audit.

### v2.0.1 — 2026-10-10

Both defects in `.local/astra-improve/builds/hb88-audit2-verify-out.md` were
reproduced on v2.0.0 before editing production code.

- D1: Enhanced `FrameContentSequence.FrameAcquisitionNumber` now follows
  per-frame > shared > top-level precedence and partitions groups independently
  of classic `AcquisitionNumber`. Group receipts expose `frame_acquisition_number`.
  Missing values remain distinct from explicit numbers, including zero. The
  whitelist includes the top-level fallback; dimension pointer `(0020,9156)` is
  recognized as nonspatial. Serialized Enhanced MR `[1,1,2,2]` now forms two
  two-frame groups, with stable group IDs and ordered references under reversal.
- D2: Routing retains the header description's complete side-token set and
  rejects contradictory LEFT+RIGHT even if DICOM laterality is absent or agrees
  with the index. Receipts expose `header_description_sides`; absent and repeated
  single-side descriptions keep their prior valid behavior.
- Regression tests cover real DICOM serialization through the production tag
  whitelist, acquisition precedence/identity/dimensions, both conflicting side
  orders, and the existing header-only/no-cache-write path. Frozen v1 is unchanged.
- v2.0.0 verification notes cannot authorize v2.0.1. Obtain a new independent
  PASS bound to the current code, tests, runbook and amended protocol before a
  full real-header audit. This patch itself does not rebuild caches or certify
  clinical labels, defect prevalence, or model quality.
