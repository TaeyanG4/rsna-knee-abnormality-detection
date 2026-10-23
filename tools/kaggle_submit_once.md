# kaggle_submit_once v1.3.4

Changelog v1.3.4: accept strict gate 1.3.0 only for two explicit standalone
child/parent pairs (R-KD/public R; ConvNeXt-S L2/HB151 v1.0.2). Require main gate
and standalone/HB151 helper hashes. A's parent metadata is synthetic provenance,
not historical upload metadata. New pins do not widen old-version receipts.
Backups: `.local/backups/strict-variant-gate-v1.2.0/`. Submission execution,
one-attempt/ledger/lock, fallback and visible checks are unchanged.

Changelog v1.3.3: accept gate v1.2.0 and refuse fixture-only gate admission.
Originals backed up in `.local/backups/strict-variant-gate-v1.1.0/`.
Submission execution, ledger/lock/daily cap, parent pins, prediction fallback
and one-attempt safeguards are unchanged.

Changelog v1.3.2: accept strict gate 1.1.0 (and 1.0.0), bind its parent to
registration.parent and registration.parent_sha256sums, and pin both supported
hidden-proven parent kernel identities. Source/docs/tests/smoke originals backed
up at `.local/backups/kaggle-submit-once-v1.3.1/`. No scored stage, production gate,
receipt, ledger, fallback validation or submission execution safeguard changed.

Changelog v1.3.1: prediction-only fallback admission; anchor-method counts ignored;
HB127 missing-member/R7 counters supported. Backups of all four changed files:
`.local/backups/kaggle-submit-once-v1.3.0/`. FAST gate interface unchanged.

Changelog v1.3.0: mandatory visible receipt and PASS review admission, explicit remote
slug/version binding, FAST strict-gate binding, offline `--check-only`, and bounded
Windows lock cleanup. Original v1.2.0 source/tests are backed up under
`.local/backups/kaggle-submit-once-v1.2.0/`. Existing lock, durable IN_FLIGHT ledger,
one-attempt/retry-approval, reconciliation, daily-cap and confirmation rules remain.

```text
python tools/kaggle_submit_once.py --id <unique-id> --kernel <owner/actual-slug> --version <Kaggle-version> --message "<description>" --receipt <visible.json> --review <pre-submission.md>
```

Admission failure exits 2 before any list/submit call. `--check-only` runs admission
without endpoints, locking or ledger writes; `--message` is unnecessary in this mode.
Submission qualifies again after the list read, immediately before IN_FLIGHT.
Every admission input SHA256 is recorded in the attempt ledger. Reconciliation
remains unchanged and does not require new evidence for an existing attempt.

## Visible receipt transport

Receipt `status` must start with `COMPLETE`. Prediction fallback counters must be
numeric zero: `fallbacks.{a172_prediction,r_prediction,s_r7_prediction}` (or generic
`fallbacks.prediction`) and nested `prediction_fallbacks`, `a172_prediction_fallbacks`,
`r_prediction_fallbacks`, `s_r7_prediction_fallbacks`, `r_fallback_count` equivalents.
Anchor-method counters such as `s_r7_anatomy_anchors.fallback` are ignored, including
copies nested under `pr7` or `fallbacks`. Missing/unrecognized prediction evidence
fails. HB127 instead requires zero `missing` member counts (including R2-s43,
R2-s44, S_R7) and both prediction counters under `r7_telemetry`.
The producer boolean `pr7.telemetry.r_fallback_count` (or HB127's
`r7_telemetry.telemetry.r_fallback_count`) is an equality assertion: true passes,
false fails. Other prediction-counter booleans fail.
The existing FAST-only `fallbacks.placeholder_errors` schema remains supported;
its acceptance still requires the full strict gate and review below.
Codecs, when present, require `ACTIVE` (`codecs.status`, plus `s_r7_gate` if present).

Native receipts contain these transport fields alongside runtime fields:

```json
{
  "status": "COMPLETE_C85R7",
  "version": "1.1.9-fluidR",
  "kernel": "taeyangg4/taeyangg4-knee-lb-fluidrd-v1",
  "kernel_version": 1,
  "registration": "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.9-fluidR/registration.json",
  "registration_sha256": "<64 lowercase hex digits>",
  "fallbacks": {"prediction": 0},
  "codecs": {"status": "ACTIVE", "s_r7_gate": "ACTIVE"}
}
```

For immutable old receipts, create a sibling `submission-binding.json` containing
`schema_version: "1.0.0"`, `receipt_sha256`, and the four transport fields `kernel`,
`kernel_version`, `registration`, `registration_sha256`. This sidecar must record
the confirmed remote version; a directory name alone never supplies the version.
Do not alter scored receipts. Semantic runtime `version` must equal registration
`version`; a positive integer receipt `version` must equal CLI `--version`.
Conflicting native kernel/version values fail even when a sidecar exists.

Registration `kernel_id` must equal the actual CLI kernel, except the explicit
historical mapping `owner/RegisteredSlug` → `owner/owner-registeredslug` (lowercase
slug). That alias, and any owner-prefixed title slug, additionally require a
receipt directory named `<actual-slug>-visible`, `<owner>-<actual-slug>-visible`,
or either form with `-v<Kaggle-version>-visible`, or a matching gate binding.
No arbitrary alias or case normalization is accepted.

Review text must name the exact full actual slug, contain one standalone verdict
line, and end with `VERDICT: PASS` (trailing whitespace allowed). This is local
evidence validation, not cryptographic authentication of its author or a remote
version lookup. B supplies truthful download/version provenance and review.

## FAST interface with strict_variant_gate v1.0.0 / v1.1.0 / v1.2.0

FAST is identified by `FAST` in status/mode, `fast_path: true`, or a sibling
`FAST_PATH_VISIBLE` marker. It requires `gate_sha256` in the transport binding.
The gate file is exactly `.local/state/gates/<actual-slug>.json` and must contain:

```text
gate_version: "1.2.0" # 1.0.0 and 1.1.0 also supported
status: "PASS"
errors: []
kernel_id: <exact owner/actual-slug>
stage: <project stage path>
parent: <registered parent stage path>
input_sha256: {<project input path>: <sha256>, ...}
```

The stage registration must contain `parent` (project-relative or absolute stage
path) and `parent_sha256sums` (SHA256 of that parent's SHA256SUMS[.txt] bytes).
The receipt's parent must resolve to exactly this path; stage cannot equal parent.
Parent input coverage requires registration, manifest and ALL kernel files, whose
bytes must match the tool's immutable pins for C85-R7 v1.1.0 or EXP143v2 standalone
(kernel taeyangg4/knee-exp152-own0949-v2, ledger ref 57005140). A byte-identical
provenance copy may change directory layout. Arbitrary parents, spoofed identity,
missing hashes and altered parent bytes fail. These checks also apply to v1.0.0.
Old registrations without these fields need a NEW versioned binding stage and
fresh manifest/allowlist/independent review/gate; never edit a scored stage. Gate
v1.2.0 `bind` adds these fields automatically.

The consumer rehashes every gate input. Coverage must include stage registration,
all stage kernel files including metadata, stage SHA256SUMS[.txt], and
`tools/strict_variant_gate.py`. Added/renamed kernel files fail. Gate stage
registration must be the visible transport registration; registration and metadata
identities must match. Missing/failed/stale/unbound gates fail closed. An optional
gate binding on a non-FAST receipt receives the same checks. FAST also requires
COMPLETE, version, zero prediction fallback counters, and the pre-submission review; gate
approval does not claim that placeholder output exercised hidden inference.

## EXP143v2 / HB151 gate v1.2.0

The exact external archive and 5-to-6 FAST path are now supported by gate v1.2.0.
Gate `bind` consumes the canonical EXP143v2 parent and produces a new nested
registered parent copy plus the HB151 registration/manifest/allowlist/full diff.
It preserves source kernel bytes and requires new independently dispatched
review evidence. See `tools/strict_variant_gate.md` for the hash-bound contract.
Fixture-only gate output is rejected, even if its validation status is PASS.

Use the command sequence in
`.local/scratch/astra-b5/strict-variant-gate-v120/LAUNCH.md` after the real seed43
export is bound and independently reviewed. The visible receipt's sidecar must
bind the confirmed remote version, new registration SHA and published gate SHA.
No actual Kaggle calls or real submission admission occurred during this build.

## CPU LAUNCH / validation

No register/prepare/training commands apply. Run from the project root:

```text
python -m pytest tests/test_kaggle_submit_once.py tests/test_strict_variant_gate.py --basetemp=.local/pytest-tmp/astra-b5 -q
python tools/kaggle_submit_once_smoke.py
python tools/kaggle_submit_once.py --check-only --id synthetic-smoke --kernel taeyangg4/taeyangg4-knee-lb-fluidrd-v1 --version 1 --receipt .local/scratch/astra-b5/kaggle-submit-once-v132/native/taeyangg4-knee-lb-fluidrd-v1-visible/visible.json --review .local/scratch/astra-b5/kaggle-submit-once-v132/native/taeyangg4-knee-lb-fluidrd-v1-visible/review.md
```

The smoke uses immutable real receipts and parent/candidate kernel bytes, with
explicitly synthetic transport/review/gate fixtures in one fixed scratch directory.
It proves consumer admission, not gate-producer approval. Both parent paths admit;
stale/wrong parent evidence refuses. Prediction-only fallback checks and mocked
one-attempt submission still pass. subprocess/network are blocked. Original hashes
are rechecked; production stages, receipts, gate directory and ledger stay untouched.
`results.json` records outcomes; `parent-probe.json` records actual gate blockers.
