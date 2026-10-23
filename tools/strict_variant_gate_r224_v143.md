# H-B-167 exact R224 + FLIP_ALL gate v1.4.3

Changelog 1.4.3 (2026-10-15): new immutable gate contract for exactly
`taeyangg4/knee-lb-r224flipq-v1`; new consumers `kaggle_submit_once_v137.py`
v1.3.7 and `.local/run-scripts/b-preflight-v105.py` v1.0.5. Previous versions
and the R224 parent admission receipt remain unchanged. Backups are under
`.local/backups/strict-variant-gate-v1.4.3/`.

The parent is the admitted `.local/scratch/astra-b5/r224-gate/gate-copy-v142`.
The source is `artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq`.
The gate pins all kernel bytes, three changed embedded members with exact
before/after hashes and unified hunks, 28 identical members, and the unchanged
FAST cell. The full before/after `exp169_r7_port.py` bytes and unified hunk must
match the original v1.1.0 -> v1.1.22-flipallq change. The parent admission and
every input it records must still hash correctly. This is static release
integrity, not proof of numerical quality or submission qualification.

## CPU proposal commands

Run from the project root with system Python. These commands do not import or
execute the embedded candidate code. Pin is idempotent only for identical
inputs; bind creates a new directory and refuses to overwrite one.

```powershell
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' tools/strict_variant_gate_r224_v143.py pin --stage artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq --builder-receipt .local/scratch/slot-b/hb167/build-out.md.receipt.json --output .local/scratch/astra-b5/hb167-gate/R224-PIN-v143.json
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' tools/strict_variant_gate_r224_v143.py bind --pin .local/scratch/astra-b5/hb167-gate/R224-PIN-v143.json --output .local/scratch/astra-b5/hb167-gate/gate-copy-v143 --review .local/scratch/slot-b/hb167/gate-review-out.md
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' tools/strict_variant_gate_r224_v143.py evaluate --stage .local/scratch/astra-b5/hb167-gate/gate-copy-v143 --review .local/scratch/slot-b/hb167/gate-review-out.md --check-only
```

The last command must refuse admission while the future independent review
is missing. There are no separate register/prepare/admit-dry-run commands:
pin, bind and evaluate --check-only implement those CPU protocol steps.

## Independent candidate review and later admission

Give a different dispatcher lane `gate-copy-v143/review-request.md`, the PIN,
the complete diff and both inventories. The authenticated report at
`.local/scratch/slot-b/hb167/gate-review-out.md` must retain the exact
`SHA256SUMS`, `Allowlist-SHA256`, `R224-PIN-SHA256`, `R224-Scope` and
`R224-Flip-Content` lines; name its actual `Reviewer-Lane`; and conclude
`VERDICT: PASS`. Review semantic scope independently; copying request text is
not a review. The dispatcher receipt must authenticate that report, and its
lane must differ from the builder. Tooling review is a separate requirement.

Only after that independent review, B can run the following CPU command:

```powershell
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' tools/strict_variant_gate_r224_v143.py evaluate --stage .local/scratch/astra-b5/hb167-gate/gate-copy-v143 --review .local/scratch/slot-b/hb167/gate-review-out.md
```

On PASS this writes the create-once admission
`.local/state/gates/knee-lb-r224flipq-v1.json`. It does not authorize a Kaggle
submission by itself. B must explicitly use the new versioned consumer paths;
the old launchers and consumers are not redirected. The new consumers preserve
their prior branches and accept gate version 1.4.3 only for this exact slug,
source, child pins and admitted parent. GPU/Kaggle actions and quota decisions
remain outside this tooling build.

## Validation evidence

Builder report, full regression log and completed-artifact CPU smoke live in
`.local/scratch/astra-b5/hb167-gate/`. Tests use only the fixed
`--basetemp=.local/pytest-tmp/astra-b5`; positive reviewer fixtures cannot publish
a production receipt. Negative cases cover resealed child changes, member
sets and content, FAST/guard/launch changes, parent/reference drift, stale
bound files, review hashes/scope/content, same-lane review, and both consumers.
