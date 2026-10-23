# R224 exact-diff contract v1.4.2

Changelog: use the independently reviewed v1.1.23-r224q wrapper bytes as the
contract. Preserve v1.2.0/v1.3.0 and separate v1.4.0/v1.4.1 tools. Consumers:
submit-once v1.3.6, preflight v1.0.4. Backups (including the previous snapshot
test) are under `.local/backups/strict-variant-gate-v1.4.2/`.

Only `taeyangg4/knee-lb-r224q-v1` and its pinned C85-R7 v1.1.0 parent qualify.
The contract fixes both added member hashes, reconstructs the original nartaa
checkpoint cell + one LF + scored cell with exactly two edits, fixes wrapper
before/after hashes, and stores its seven exact unified-diff hunks in the PIN.
Any wrapper byte change fails, even with a rebuilt sidecar and builder receipt.
Removing only the fixed R224 insertion must recover the complete parent
`run_public` function byte-for-byte. The other 28 members must be identical;
FAST/guard and metadata retain their existing pinned checks. `reason` and
`registered_reason` are accepted; supplied aliases must be nonempty strings
with identical values. Extra, missing and renamed members remain forbidden.

The reviewed source is `.local/scratch/hb164-r224/hb164/stage-result-review.md`.
The added project integration still **requires independent content review**:
only r_R = 0.65 r(R384) + 0.35 r(R224), heliosli failure semantics and output
authentication. Review must bind the PIN, allowlist, manifest, exact scope and
project hash, with an authenticated dispatcher receipt from a different lane.
PIN/bind success means REVIEW_REQUIRED, not admission or mathematical approval.
Reviewers inspect the entire wrapper diff, all 28 identities, source reconstruction,
failure/survivor arithmetic, target/UID order, and the unchanged existing launches.

## CPU launch card

Run from the project root in PowerShell. These commands execute no public
notebook code, CUDA inference or Kaggle write. PIN is create-once/idempotent;
bind creates one new immutable copy and refuses an existing output.

```powershell
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' tools/strict_variant_gate_r224_v142.py pin --stage artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.23-r224q --builder-receipt .local/scratch/slot-b/hb164/build2-out.md.receipt.json --output .local/scratch/astra-b5/r224-gate/R224-PIN-v142.json
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' tools/strict_variant_gate_r224_v142.py bind --pin .local/scratch/astra-b5/r224-gate/R224-PIN-v142.json --output .local/scratch/astra-b5/r224-gate/gate-copy-v142 --review .local/scratch/slot-b/hb164/r224-gate-review-out.md
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' tools/strict_variant_gate_r224_v142.py evaluate --stage .local/scratch/astra-b5/r224-gate/gate-copy-v142 --review .local/scratch/slot-b/hb164/r224-gate-review-out.md --check-only
```

The last command must refuse until the exact independent bound review exists.
Fixture PIN/bind/evaluate and both consumers are tested without writing any
production admission. There are no separate register/prepare commands for this
tool; PIN authenticates source completion and bind prepares the review package.

```powershell
& 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe' -m pytest tests/test_strict_variant_gate.py tests/test_strict_variant_gate_hb151.py tests/test_strict_variant_gate_standalone.py tests/test_kaggle_submit_once.py tests/test_b_preflight_v1.py tests/test_strict_variant_gate_r224.py tests/test_strict_variant_gate_r224_v141.py tests/test_strict_variant_gate_r224_v142.py -q --basetemp=.local/pytest-tmp/astra-b5
```

The authorized consumer update replaces v141's live consumer byte-equality
assertion with equality of the v142 pre-change backup and v141 backup. Frozen
contract source hashes remain tested directly, and all older behavior tests run.
