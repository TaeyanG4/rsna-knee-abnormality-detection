# H216 explicit resume core v1.0.1

2026-10-19. Preserve the original v1 numerical implementation and receipts.
The new module keeps the fixed400adapt/800recovery recipe and adds explicit
committed-update resume. No checkpoint/model read or GPU work occurs on import.

Each snapshot binds arm, phase, immutable base SHA, production-input contract,
recipe, parameter manifest and exact schedule. It persists model, AdamW,
scheduler, EMA, loss history, loader epoch/seed, crop/CPU/CUDA/NumPy/Python RNG.
Resume indexes the next uncommitted batch directly without replaying skipped
inputs or updates. A phase-end checkpoint stores pre-EMA-application weights;
resume applies that EMA once before the original-head restoration/recovery.
Recovery snapshots contain the completed adaptation receipt. The outer caller
must freshly authenticate the original base and checkpoint file identity.

An explicit pause request is checked after an update commits. The synchronous
checkpoint callback must finish before PausedAtCommit is raised. There is no
automatic resume/retry, no change to the live H210 job and no implicit GPU launch.

CPU tests use a miniature synthetic encoder/data and a smaller crop solely for
resume mechanics. They run the full400+800 update counts, compare v1/v1.0.1 and
interrupted/uninterrupted results for both arms, including full optimizer/EMA/
RNG state, adaptation100/400 and recovery100/800 boundaries and a nonperiodic
pause. Synthetic state is serialized only in memory. This is not CUDA
determinism, production model throughput, new training evidence or efficacy.
Real ConvNeXt/cache CPU mechanics remain the previously completed v1 evidence.
