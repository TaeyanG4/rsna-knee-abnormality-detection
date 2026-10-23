# R224 addition contract 1.4.0

Separate, static CPU tooling for H-B-164. The legacy gate and standalone helper
are unchanged. This contract is restricted to C85-R7 v1.1.0, candidate
v1.1.23-r224q and `taeyangg4/knee-lb-r224q-v1`.

## Trust boundary

`pin` authenticates a SUCCESS dispatcher receipt before inspecting the candidate.
The archived builder report must contain `Kernel-SHA256: <digest>`, where digest
is SHA256 of the gate's canonical JSON encoding of the exact kernel file hashes.
The dispatcher archive is the existing local trust boundary, not a signature.
Never fabricate or edit its reports/receipts to satisfy this contract.

The parent kernel pair and 29 embedded members are pinned. The public original
notebook is pinned independently. The added member must be one directly
executable script consisting of its checkpoint-verification cell and scored
source, with only the two exact `[blend]` edits. Supported separators between the
cells are LF, two LF, LF plus `# %% [code]` and LF, or two LF plus that marker.
No added member is accepted merely because its supplied hash/provenance agrees.

`new-members.json` requires `added` and `changed` arrays. Added rows have `name`,
`sha256` and `origin`; origin includes the original notebook/source SHA and the
two filenames `raptor224_sub.csv` and `raptor224_raw.csv`. Changed rows have
`name`, `before_sha256`, `after_sha256`, and a nonempty `reason`. All inventories
must match exactly; placeholders fail.

Only `exp169_release_c85r7codec.py` can change among existing embedded members.
Allowed AST spans are collect, candidate_receipt, r224_ numerical helpers, and
explicit version/formula metadata. Existing orchestration functions are locked;
each newly inserted helper may include zero to two LF separators after its body.
run_public must equal its parent with exactly the direct R224 child splice
returned by `invocation(name)`, immediately before `def infer_s`. All other
members, notebook fields and existing launch lines remain byte-identical after
the pinned FAST cell/guard and archive bindings are accounted for.

The numerical spans are review material, NOT a mathematical proof or Python
sandbox. An independent dispatched reviewer must verify the R-slot formula,
independent failure masks, survivor renormalization, unchanged own branch,
UID/target order, NaN handling and absence of orchestration changes. Review
must bind SHA256SUMS, allowlist and PIN and contain the exact R224-Scope line in
the generated review request. A PASS without these bindings is rejected.

## CPU launch card

Use the system Python. These commands never launch inference or call Kaggle.
Set `receipt` to the completed builder's authentic dispatcher receipt; `review`
is the future independently dispatched candidate review report path, outside
the builder's scratch. The candidate build and review are separate from the
tooling implementation review.

```powershell
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$pin = '.local/scratch/astra-b5/r224-gate/R224-PIN.json'
$gate = '.local/scratch/astra-b5/r224-gate/gate-copy'
# Supply $receipt and $review from the actual dispatcher jobs.
& $py tools/strict_variant_gate_r224.py pin --builder-receipt $receipt --output $pin
& $py tools/strict_variant_gate_r224.py bind --pin $pin --output $gate --review $review
# After the independent candidate review has completed:
& $py tools/strict_variant_gate_r224.py evaluate --stage $gate --review $review --check-only
# B publishes the create-once gate receipt only after the check succeeds:
& $py tools/strict_variant_gate_r224.py evaluate --stage $gate --review $review
```

PIN is immutable REVIEW_REQUIRED evidence, never admission. Bind creates a new
copy without changing source, parent or scored receipts. A differing existing
PIN or existing output directory is refused; investigate rather than overwrite.
There are no register/prepare/admit subcommands: pin/bind/evaluate are their
static tooling equivalents. The gate does not qualify GPU/Linux behavior,
runtime, output predictions or the final submission minimum review.

Tests isolate synthetic candidates under the lane's fixed pytest basetemp and
mark their PINs and evaluations fixture_only. CLI admission and consumers reject
fixture-only evidence. Test source reconstruction uses the real immutable
parent/public-source bytes; no running builder output or prediction data is read.

## Consumer changes

`kaggle_submit_once.py` 1.3.5 and `b-preflight-v1.py` 1.0.3 accept the new contract
only for this slug and pinned parent, with the new helper/PIN/allowlist/diff
hashes. Backups: `.local/backups/strict-variant-gate-v1.4.0/`. Previous gate
sources and production receipts are not rewritten.
