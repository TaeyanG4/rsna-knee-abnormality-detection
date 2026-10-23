# K1 offline admission and submit-once flow

Version 1.0.0 (2026-10-20). This replaces the candidate-specific `admit_*_v1.py`
scripts. Run from the project root in PowerShell. B owns downloads and submission.
The admission tool itself only reads local evidence and runs the canonical
offline `--check-only`; it never imports Kaggle, downloads, or submits.

## Four-command flow

Choose the exact registered stage, owner/slug and Kaggle version. The stage release
version (for example `1.1.63-r6w958n224k1x3sr20`) and Kaggle version (`1`) are
different fields. Set `$stage`, `$kernel`, `$version`, and `$visible` to those
values and a new project-local output folder. Use the existing app Python
environment for the downloader; do not install or rebuild an environment.

1. Download four explicit outputs at the exact version. The downloader also
   writes the required `*.download.json` provenance sidecars. It preserves
   existing files and refuses overwrites.

   ```powershell
   foreach ($name in @('k1sr-receipt.json', 'c85r7-receipt.json', 'submission.csv', 'k1sr-parent.csv')) {
       python tools/kaggle_output_file.py --kernel $kernel --version $version --file $name --out (Join-Path $visible $name)
       if ($LASTEXITCODE -ne 0) { throw "Download refused: $name" }
   }
   ```

2. Prepare the admission. Add `--n32-dir <evidence-folder>` when qualifying
   runtime from N32. An exit code of 2 is REFUSE: fix the evidence or candidate;
   never sed-edit a receipt or generated PASS.

   ```powershell
   python tools/k1_admit_v1.py prepare --stage $stage --kernel $kernel --version $version --visible-dir $visible
   ```

3. Inspect the generated checklist and obtain/retain the separate independent
   review for this exact candidate. Refresh live settings, daily allowance and
   duplicates using B's established procedure. Do not query scores.

   ```powershell
   Get-Content -LiteralPath (Join-Path $visible 'k1-admission-v1/READY-REVIEW.md')
   ```

4. Execute the **exact one-line command printed by step 2**, once, only when the
   independent review and live checks permit submission. Its form is:

   ```powershell
   & '<python>' '<project>/tools/kaggle_submit_once.py' '--id' 'K1-<printed-id>' '--kernel' '<owner/slug>' '--version' '<n>' '--message' '<verbatim-receipt-formula>' '--receipt' '<visible>/k1-admission-v1/c85r7-receipt.json' '--review' '<visible>/k1-admission-v1/READY-REVIEW.md'
   ```

   Copy the emitted line; do not manually reconstruct it or evaluate log text
   with `Invoke-Expression`. Never automatically retry an uncertain attempt.
   The canonical wrapper owns its durable submission ledger and lock.

If any evidence changes between preparation and submission, rerun step 2.
Generated outputs are immutable: use a fresh visible evidence copy if an old
admission differs. A previous READY file is historical evidence, not permission
to ignore a later refusal. The tool does not revoke old artifacts.

## Inputs and outputs

Required visible files are the four downloaded outputs above, plus download
sidecars for `k1sr-receipt.json`, `c85r7-receipt.json`, and `submission.csv`.
`k1sr-parent.csv` is necessary to verify the parent hash against actual bytes.
For a historical replay, an existing byte-identical parent CSV can supply those
bytes, provided its source is explicitly recorded; never invent a download receipt.
The tool also reads the stage registration/notebook/metadata, the local sample
submission template and the fixed production submission ledger.

Output directory: `<visible>/k1-admission-v1/`.

- `ordinary-visible-registration.json`: exact formula, stage pins, all evidence
  hashes, and the optional corrected projection.
- `submission-binding.json`: exact remote kernel/version and receipt hashes.
- `c85r7-receipt.json`: byte-identical parent receipt copy for the canonical
  wrapper's sibling-binding convention.
- `machine-check.pending.md`: passing machine checks used only for the canonical
  offline check. This is not an independent review or a final READY file.
- `READY-REVIEW.md`: written only after all checks and canonical
  `ADMITTED_OFFLINE`, ending in `VERDICT: PASS`.

PASS means these automated local checks passed. It does not establish scientific
gain, independent review, live quota, or guaranteed hidden runtime. Omitting N32
is supported and recorded as **no runtime qualification claimed**; retain the
candidate's separately established runtime evidence before submission.

## N32 corrected projection evidence

Use the same model hash, formula, contract and R224 weight as the visible
candidate, with COMPLETE_SHARED_K1, `k1_executed: true`, `k1_studies: 32`, and
32 unique successful `per_study` rows. Phase totals must match those rows.
Existing timing from a different recipe is not accepted merely because it is
convenient. No timing correction is inferred from undocumented constants.

Alongside `k1sr-receipt.json`, supply `projection-inputs.json`:

```json
{
  "schema": "k1-admit-projection/v1",
  "receipt_sha256": "<sha256 of N32 k1sr-receipt.json>",
  "compatibility_reason": "<why this measured run represents the visible candidate>",
  "startup": {"path": "startup.json", "sha256": "<sha256>"},
  "selection": {"path": "selection.json", "sha256": "<sha256>"},
  "train": {"path": "train.json", "sha256": "<sha256>"}
}
```

Each evidence file must be project-local and hash-bound:

- `startup.json`: `receipt_sha256`, measured `parent_startup_seconds`, and
  `method` describing the measurement.
- `selection.json`: `n: 32`, and `counts: { "<study UID>": <DICOM count>, ... }`
  for exactly the N32 receipt's roster (existing N32 selector shape).
- `train.json`: aggregate `studies`, `dicom_files`, and `method` identifying
  count provenance. No labels or report text.

The printed rule and all numeric inputs are retained in the registration:

```text
parent_per_study = (parent_seconds - startup_seconds) / 32
                   * (train_dicom_files / train_studies)
                   / (sum(roster_dicom_counts) / 32)
k1_per_study = (classifier_seconds + extra_render_seconds) / 32
hours = (startup_seconds + load_seconds
         + 1300 * (parent_per_study + k1_per_study)) / 3600
PASS requires hours <= 8.0
```

Parent wall time retains overlapping in-pass K1 work, conservatively counting
that overlap twice. Only parent steady-state time is corrected by roster size;
startup and model load are counted once. This is a documented estimate, not a
runtime guarantee. Missing evidence or a projection above 8 hours refuses
admission rather than falling back to an older asserted number.

## Verification and known historical packages

Run the synthetic and canonical regression suites:

```powershell
python -m pytest tests/test_k1_admit_v1.py tests/test_kaggle_submit_once.py tests/test_kaggle_submit_once_metadata.py tests/test_kaggle_submit_once_v137.py tests/test_kaggle_submit_once_v138.py tests/test_kaggle_submit_once_v139.py --basetemp=.local/pytest-tmp/astra-b1 -q
```

The builder's local replay harness and evidence are under
`.local/scratch/astra-b1/k1-admit/`. It explicitly substitutes an empty historical
ledger inside the test process for pre-submission replays; the production CLI
has no ledger override. The real `knee-lb-r6w958k1x3sr20-v2` version 1 is already
SUBMITTED, so production admission must REFUSE even though its historical input
replay passes. The degraded `n224k1x2sr20` visible output always refuses.
Existing N32 packages without measured correction inputs cannot establish PASS.
