# H216 staged-transfer core v1.0.0

2026-10-19. Implemented numerical core, not an admitted training run. No disease
checkpoint is read while H210 trains. B direct tests are not independent review.

Both arms start from the same completed K1v3 seed0 final-EMA checkpoint SHA
6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5.
Keep original3190/977/976 cohorts and L2 unchanged. Control adaptation uses
RSNA E_prior targets; candidate uses the existing30masked native categorical OAI
tasks. Each has400adaptation and800RSNA recovery optimizer updates, batch2,
64windows,384input/random352crop, original cover/3.5mm/.25slice-jitter recipe.
No extra augmentation or image smoothing is introduced.

Only `enc.stages.3.*` trains in the encoder in either phase; frozen stem/stages0–2
are in eval mode. Reject a BatchNorm backbone. In adaptation both arms share the
same temporary-head architecture: spatial mean, copied per-channel normalization,
linear384/LN, slot/view/geometry embeddings, dropout.1, masked study attention,
linear output. Output is12 for control or30x4 for OAI. The head is initialized
with the same fixed seed; no disease-head tensor is reassigned a native label.
All temporary-head weights train, then are discarded. Only adapted last-stage
encoder weights survive; restore the complete original disease head exactly.
Recovery trains stage3 plus the original head, including relocated norm.

Use unchanged K1v3 AdamW grouping/no-decay policy: pretrained LR1e-4, head3e-4,
wd.02, clip5. Each phase has its own10%warmup and cosine-to-zero schedule. EMA.999
with original warm-EMA formula is applied at both phase ends, then EMA/optimizer/
scheduler are reset at recovery. This short-transfer schedule is a prospective
choice fixed before outcomes, not a tuned result or an extension of H210.

Each native task uses four-class CE over known grades; missing -1 is masked,
per-task observed means then averaged. RSNA uses exactly the E_prior all-one
weighted BCE. There is no invented pos_weight, empty-cell fill or target weighting.
Recovery seed21602 resets data order, per-study window seed, crop and dropout RNG
independently of different adaptation consumption. A deterministic permutation
schedule with drop-last pair batches is persisted. Identical schedules do not
mean identical RSNA exposure across the full two-phase packages.

Callbacks receive completed-update telemetry and100-step snapshots containing
model/optimizer/scheduler/EMA/RNG/schedule state. This module does not launch,
automatically resume or infer. Before any GPU fit: bind full qualified input
rosters/cache hashes and actual source checkpoint, authenticate exact parameter
manifests, finish outer ownership/watchdog/storage/resume/inference integration,
and measure real memory/throughput at the exclusive slot. Missing inputs or
outer integration remain HOLD. A random-weight CPU smoke is only a mechanics
check. No model efficacy, full input readiness or0.960 forecast follows from it.
