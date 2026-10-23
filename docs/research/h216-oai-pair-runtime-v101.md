# H216 paired outer runner v1.0.1

2026-10-19. Additive explicit restart support; numerical recipe, input800,
original seed0 checkpoint,400+800updates, masks and inference are unchanged.
Original runtimev1/core/producer/pilot registrations and evidence remain intact.
This is CPU implementation work, not actual ROI, training admission or efficacy.

New attempts live under private `h216-oai-pair-runtime-v101/a001`, then `a002`,
etc. A restart is explicitly prepared with `prepare --attempt a002 --resume-from
a001`. It accepts only the immediately preceding attempt's bound gpu_job
terminal exit3 plus matching PAUSED_AFTER_COMMIT receipts. A failed/crashed
exit1, missing/unknown/live/orphaned owner, completed pair, mismatched committed
step/identity or changed checkpoint is rejected. Terminal owner checks precede
checkpoint inspection. Wrapper, outer child and saved worker identities must
all be dead. No current H210 checkpoint is opened or altered by this tool.

Preparation binds original registration, exact checkpoint and sidecar hashes,
terminal/pause receipts, phase/step and completed-arm references. Each attempt
gets its own registration, B-direct precheck, exact command/tag and new actual
GPU admission. No consumed admission is replayed. Source, recipe, input and
cache must match the original attempt. The original source checkpoint and
results remain unchanged; new progress/checkpoints/results go to the new
attempt. The numerical core restores optimizer/scheduler/EMA/RNG and resumes
after the committed update. No automatic retry/resume or new execution queue.

If the control arm was already complete before pausing the native arm, its
exact result/model/holdout hashes are retained as references; it is not fit or
inferred again. Both initial arms still start from the same immutable seed0
finalEMA. The original model is reloaded before applying an authenticated
resume state, preserving the original disease-head semantics.

Checks focus on active-owner refusal before checkpoint reads, wrong terminal
exit, altered sidecars/payloads, cross-attempt command/PID identity, completed
arm preservation and actual disk serialization into the miniature CPU core.
Original cache/inference/CSV functions are AST-identical to the checked v1.
The existing actual-RSNA/random-weight inference smoke is reused for those
unchanged functions. New tests do not read disease checkpoints or initialize
CUDA. B direct checks are not an independent PASS.

Actual execution still requires queued actual ROI8, qualified fixed800 cache,
real-input validation, source-bound B precheck and fresh resource/time checks.
Full worker/process/CUDA restart remains unexecuted. A future genuine pause
requires explicit B preparation and a new launch, not an automatic recovery.
H210 and the existing ROI successor are unchanged. No current job is paused
or newly queued by creating this version.
