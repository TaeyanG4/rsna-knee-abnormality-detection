# B_RN3F_SERVING_V1 · CPU stage and next measurement plan

The root decision selects the existing six B8/RNclip20 F0–2 checkpoints. The output is standalone B+RN: `mean_f(0.9 rank_pct(B_f) + 0.1 rank_pct(RN_f))`. Each target/member is ranked over the complete supplied cohort, average ties/N, float64, no final rerank. This differs from rank(mean across folds) and from the generic ensemble's uniform probability mean. No anchor or fresh EXP084 model is included. Public 0.943 improvement and hidden performance remain unproven.

## What this implementation establishes

- Import/build is passive: no training, model construction, GPU execution, decoder installation, network operation or submission. CPU tests use synthetic tensors only.
- Source/checkpoint/wheel hashes and input/target/member order are bound. The legacy D01 DICOM reader is copied unchanged, despite a different current-workspace reader. The existing engineering100 selection and frozen local cache manifest are separately pinned.
- The explicit future `run` command requires the recorded manifest SHA and a fresh output directory. It uses one visible GPU, FP16 strict, batch1, resident exact-six, no retries or fallback; Linux T4 dependencies install offline into a fresh directory. Effective CPU allocation caps requested12 threads; OpenCV1/inter-op1/no loader pool prevent nested oversubscription. Small staging copies remain serial.
- The worker saves each raw float32 member array and the complete-cohort rank blend. Publication is an atomic directory rename after UID/target/schema/range/input/member-hash/completion checks. Launch errors, timeout and missing/partial/changed member receipts fail; no partial final CSV. Partial pending files remain diagnostic evidence.
- Future runs record per-study uint8 volume/mask hashes, selected K24 centers, empty-window count, normalized input hashes, DICOM acquisition receipts, hardware/backend versions and VRAM. Those are observations, not assertions of equality.
- Only existing engineering100 or the public3 cohort is accepted. There is no hidden/full-cohort release path. `COMPLETE_ENGINEERING_OUTPUT_ONLY`, `submission_ready=false` and `serving_parity_gate=NOT_EVALUATED` distinguish successful file production from serving validation.

## Required next decisions and execution

Root must review CPU evidence and the independent review before assigning GPU/remote work. Recheck local jobs and actual Kaggle quota once before launch. Do not launch alongside local training. Existing private Kaggle authorization permits staging a private engineering notebook when root dispatches it; publication/submission/paid use remain separate.

1. **New B-specific input gate, CPU 10–20 minutes.** Implement streaming input comparison for within-slot-valid K24 and 336→224/ImageNet with frozen legacy decode. Reuse existing analytical bilinear code only after reviewing its adapter assumptions; do not reuse a D01 PASS. Require exact UID/target/shape/dtype/centers/uint8 volume/mask identities and successful acquired-slice decode. Include all-zero mask (`-1` centers and normalized blank, not post-normalization zero), slot-boundary triplets, sparse/disconnected masks and high-contrast uint8 synthetic inputs. The producer already saves identities but does not implement this numerical gate.
2. **Exact-six local reference, GPU 10–20 minutes estimated, hard cap30 minutes.** Run the frozen engineering100 cache through the staged helper, using the executable argv in `execution-plan.json`. This is not a benchmark-derived ETA. Record CPU process-tree peak RSS as well as VRAM; V1 currently explicitly marks RSS/cgroup unmeasured. Do not interpret existing cache manifest identity as a full independent checksum of every shard; per-study hashes must be compared to the next decoded run.
3. **One T4 engineering session, wall cap30 minutes, after actual quota confirmation.** Mount the new private immutable stage, competition and the already identified OpenCV wheel dataset; resolve actual mount paths in that session and verify hashes before invocation. No new download/push was performed here. Execute the same100, archive raw/receipt, compare every member/target before ranks. A 30-minute session may debit0.5–1 GPU-hours depending on actual allocation/billing; this is a budget, not a quota claim. If unavailable, stop this branch and retain the completed CPU package.

The planned runner wall caps include model loading and worker-side wheel installation. An outer Kaggle session budget additionally includes container boot/setup, which V1 does not time. Public3 is a separate small engineering check; it never replaces100 or full hidden-runtime assessment. No full-runtime or9h joint anchor estimate is justified before new measured throughput; this candidate currently contains no anchor.

## Proposed new parity gate (must be frozen before new outputs are read)

These are proposed starting tolerances for root review, not an implemented or passed gate. Preserve observed violations and create a separately justified version if a contract truly changes; never relax thresholds after seeing outputs.

- Input: exact integer volume/mask/center equality for each of100 and required synthetics. Per-platform float32 bilinear windows versus independent float64 reference absolute error≤5e-7; normalized absolute error≤3e-6, rtol0. Derived cross-platform upper envelopes≤1e-6/6e-6. These prior D01 arithmetic bounds need a new K24/empty-window validation; they are not inherited approvals. Preserve each platform's tensor hashes even when unequal, plus actual elementwise errors.
- Outputs: all six and all12 targets, exact UID order, finite float32 probabilities [0,1], no missing member/fallback/fill. Proposed raw per-member/per-target max absolute delta≤0.002 and mean delta≤0.0005; record full distributions and pair-order inversions/ties. These numerical limits do not prove competitive quality. Rank-blend deltas are diagnostics, not a reason to retune0.9/0.1 or rerank.
- Decoder: actual compressed fixture decode and observed supported transfer syntaxes; an import/version check or kernel COMPLETE is insufficient. Preserve acquisition errors and separate absent metadata from failed acquisitions.
- Runtime/release: successful bounded worker completion is only engineering output. A separately reviewed implementation must measure process-tree RSS/cgroup peak, whole-session cold/warm timing and representative throughput; then add a full-cohort path and final competition schema gating. Current V1 cannot be used for hidden submission.

## Why this investment remains bounded

The completed readiness audit is reused, not rerun. Source/data/model fidelity and weak-label quality are different questions. This package removes an identified aggregation/serving gap but does not supply an independent validation set or erase researcher/teacher exposure. EXP084 remains a separate fresh grouped comparison. Keep small RN effects and fracture regressions visible; do not infer that more folds, larger models, a higher local AUC or five daily submission slots justify spending without a specific hypothesis. Daily submissions need a validated concrete candidate and individual user approval.

No GPU process was started, so this subtask creates no heartbeat, waiting loop or idle GPU reservation. Root owns central HANDOFF/STATE/app updates and the next dispatched measurement.
