# H224 all-data refit v1.0.1 — B launch card

Changelog v1.0.1: exact requested dataset slug `knee-k1-v3gall-v1-assets`.
The authoritative device schedule and transfer allowlist are in
`tools/l1_rco_k1_device_plan_v1.md` and FOLDS.md v4.0.0; earlier package commands
below describe the preserved v3 archive.

Builder: astra-b2. Independent review PENDING. No GPU launch, transfer or upload
was performed by this builder. These new tools do not change any fold contract.

## Fixed scientific contract

Train all 4,167 labelled studies from the sealed train3190 + holdout977 union,
with the same targets and v3-G seed0 recipe: encoder LR 0.0001, head LR 0.0003,
four optimizer groups, batch2, spatial352, 64 training windows, EMA 0.999,
unchanged augmentations and warm fraction 0.1. Ten fixed epochs = 20,830
optimizer steps, with 2,083 warmup steps. The original drop_last=True remains:
every study is eligible each epoch; one shuffled final study is dropped per
epoch to preserve batch2. Internal fold index0 preserves seed0 RNG and selects
one model, but the training split is explicitly the full labelled pool.

Export FINAL EMA only. The loss-best checkpoint is retained solely as a
diagnostic/recovery artifact. No early stopping or holdout selection. Training
loss collapse/nonfinite guards still abort failures; an aborted fit is never
exported as a completed shorter fit. At completion, the final EMA must yield
finite probabilities and a range >1e-6 in EVERY target over a deterministic
32-study training panel. Constant predictions fail closed before EXPORT.json.
The guard is only a collapse diagnostic, not a validation metric.

There is NO own-holdout evaluation: all labelled data is used for training.
Registration, completion, result, export and packaged metadata explicitly state
`clean_oof=false`, final EMA selection and no own-holdout evaluation.

## CPU commands (project root)

Windows uses the supplied Python312 executable; Linux uses `python`.
All CPU commands hide CUDA internally. Reuse the existing completed full cache;
do not rerender it for this refit. The Linux default cache is
`.local/scratch/astra-b2/h224-folds-v1/cache`.

```bash
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python -B tools/l1_rco_k1_all_v1.py register --fold all
python -B tools/l1_rco_k1_all_v1.py prepare --fold all
python -B tools/l1_rco_k1_all_v1.py validate --fold all
python -B tools/l1_rco_k1_all_v1.py cpu-smoke --fold all
python -B tools/l1_rco_k1_all_v1.py admit --fold all --dry-run
python -B tools/l1_rco_k1_allseq_v1.py launch-after-f2 --dry-run
```

New work/output root: `.local/scratch/astra-b2/h224-all-v1/all`.
No evaluation command exists. `admit` is BLOCKED until an independent PASS is
recorded at this root's REVIEW.json. Use the same six-field binding schema as
FOLDS.md, importing `tools.l1_rco_k1_all_v1` instead of the fold module:
`bindings=k.review_binding(reg)` plus distinct builder/reviewer, actual report
path/SHA and verdict PASS. Linux preparation and smoke produce host-local
bindings; do not copy the Windows preparation or review receipt.

Linux admission retains completed-cache and exact 32-study H224-PARITY.json
requirements. Existing host qualification/runtime checks in FOLDS.md still apply.

## After f2 — B only

Run from the SAME absolute project root where f2 was launched, retaining its
gpu0-owner directory and bound SUCCESS receipt. The separate launcher checks
f2's command, owner and terminal receipt, requires the owner idle, then launches
only all-data on GPU0. It does not queue f3 or wait for f2 to finish. If an old
f2→f3 sequence is active, B must first resolve that scheduling conflict; do not
run both sequences concurrently.

```bash
# Only after CPU_ADMISSIBLE, independent review and bound f2 SUCCESS:
python -B tools/l1_rco_k1_allseq_v1.py launch-after-f2
```

This uses canonical gpu_job tag `H224_K1_V3G_ALL`, the same GPU0 owner as f2,
and durable `.local/scratch/astra-b2/h224-all-v1/sequential-all-v1.json`.
Existing job directory, sequence ledger or admission blocks a second attempt.
No retry, automatic cancellation, timeout or recipe shrink is added.

## Export and local packaging

Production run automatically writes the fold-compatible seven-file export:
`fit/models/E_prior_s0_fall.pt`, `fit/train-guard.csv`, `fit/result.json`,
`fit/fit-complete.json`, `fit/runtime.json`, `registration.json`, `prepare.json`.
EXPORT.json and SHA256SUMS bind those files. The training guard CSV replaces
fold holdout.csv. Preserve `fit/ckpt/epochs.json` too for producer revalidation.

```bash
python -B tools/l1_rco_k1_all_v1.py export
python -B -m tools.k1_fold_dataset_v1 verify-export \
  --export .local/scratch/astra-b2/h224-all-v1/all/EXPORT.json
python -B -m tools.k1_fold_dataset_v1 package \
  --export .local/scratch/astra-b2/h224-all-v1/all/EXPORT.json \
  --output .local/scratch/astra-b2/h224-all-v1/dataset-all \
  --dataset-id taeyangg4/knee-k1-v3gall-v1-assets
python -B -m tools.k1_fold_dataset_v1 verify-dataset \
  --dataset .local/scratch/astra-b2/h224-all-v1/dataset-all
```

These file-only commands do not upload anything. Packaging rejects incomplete
epochs/steps, CPU-test exports, wrong source/contract hashes and failed guards.
Model ID is `k1_v3g_all`. Result review and serving qualification remain pending
until actual training has run; no benefit or leaderboard gain is established.

## Portable bundle and existing Linux installation

New tar and SHA are under `.local/scratch/astra-b2/h224-all-v1`:
`h224-folds-all-v3.tar` and `h224-folds-all-v3.tar.sha256`. It includes unchanged
fold sources, additive all tools, serving packager/source and revised FOLDS.md.
It contains private derived targets and initialization/anatomy weights, so
transfer privately. No raw DICOM, full cache, trained output, credentials or
report text. No downloaded registration or running-job directory is included.

```bash
sha256sum -c h224-folds-all-v3.tar.sha256
tar -xf h224-folds-all-v3.tar
cd h224-folds-all-v3
python -B tools/l1_rco_k1_foldpack_v3.py verify
```

For an EXISTING f2 installation, extract to a staging directory first and verify.
B copies only the new all modules/shim/launcher, packager dependencies and docs
to the existing root, comparing shared files by SHA. Do not overwrite fold
registrations, caches, admissions or job output. Do not launch from the staging
root: f2's absolute command/owner binding would differ. New standalone
installations follow the full FOLDS.md cache/qualification workflow.

Local bundle build and verification:

```bash
python -B tools/l1_rco_k1_foldpack_v3.py build
python -B tools/l1_rco_k1_foldpack_v3.py verify \
  --root .local/scratch/astra-b2/h224-all-v1/package-v3
python -B tools/l1_rco_k1_foldpack_v3.py verify-archive
```

## 4090 planning estimate

Same-host f2 training time × 20830/15500 = **1.3439×**, plus final 32-study
guard inference and checkpoint overhead. For example, a measured 8-hour f2
training phase implies about 10.75 hours of all-data training. This is a scaling
estimate, not a measured 4090 speed. The inherited provisional 3.1 seconds/update
gives 17.94 hours training; budget approximately 18 hours plus overhead until
same-host f2 timing is available. No running fold1 output was inspected.
