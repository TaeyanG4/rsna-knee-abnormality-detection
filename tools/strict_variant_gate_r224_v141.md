# R224 addition contract 1.4.1

Changelog: add one named, hash-bound project integration member requiring
independent content review. The separate entry point preserves the v1.2.0,
v1.3.0 and v1.4.0 gates, receipts and consumers unchanged. Baseline backups:
`.local/backups/strict-variant-gate-v1.4.1/`.

The parent, slug, stage, FAST cell/guard, completed-builder authentication and
all existing wrapper restrictions remain those of
`strict_variant_gate_r224.md`. The only accepted added set is
`{raptor224_run.py, exp169_r224.py}`. The former still reconstructs the pinned
nartaa checkpoint cell and scored source with exactly two registered edits.
The latter requires a matching full SHA256 from `new-members.json` and
`origin.kind = "project-authored integration and arithmetic"`. No other added
project file is allowed. Changed-member rows still require `reason`;
`registered_reason` is not an alias.

The generated PIN records `project_content_review_required` with the exact
integration SHA. Bind writes `requires independent content review` into
`review-request.md`. An independent dispatcher-authenticated PASS must include
the exact `R224-Project-Content:` line, PIN hash, scope and manifest/allowlist
bindings. The reviewer must examine the entire integration member and confirm
only `r_R = 0.65 r(R384) + 0.35 r(R224)`, heliosli failure semantics and output
authentication, including frozen C85 survivor denominators, UID/12-target
ordering and finite/missing-output behavior. Syntax and supplied hashes alone
do not establish those semantics or candidate qualification.

## CPU launch card

Run from the project root with system Python. These are static operations;
they do not run public notebook code, inference or Kaggle commands.

```powershell
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$pin = '.local/scratch/astra-b5/r224-gate/R224-PIN-v141.json'
$gate = '.local/scratch/astra-b5/r224-gate/gate-copy-v141'
$review = '.local/scratch/slot-b/hb164/r224-v141-review.md'
& $py tools/strict_variant_gate_r224_v141.py pin --builder-receipt .local/scratch/slot-b/hb164/build2-out.md.receipt.json --output $pin
# Only after PIN success; a new immutable gate copy is created.
& $py tools/strict_variant_gate_r224_v141.py bind --pin $pin --output $gate --review $review
# Only after independently dispatched candidate review completion.
& $py tools/strict_variant_gate_r224_v141.py evaluate --stage $gate --review $review --check-only
```

These are the register/prepare/validate equivalents; there are no separate
register/prepare/admit subcommands. Tests exercise all three commands through
isolated fixture-only proposals. Fixture admission is refused.

The real build2 PIN attempt is REFUSED: its R224 invocation differs from the
preserved exact R384-style splice. The source itself matches exactly. No real
PIN or gate copy is created, so real bind/check-only commands refuse missing
PIN input. See `.local/scratch/astra-b5/r224-gate/v141-diagnosis.md` and
`v141-build-report.md` for exact evidence and further preserved blockers.

Submitter and preflight v1.4.1 admission support is deliberately not enabled
while the candidate fails PIN. Existing consumer behavior remains unchanged.
This card does not authorize publication of an admission receipt, GPU work,
submission, or any edits to the candidate, scored releases or receipts.
