# External storage cleanup inventory — 2026-10-22

Version 1.0.0. Observed 2026-10-22 18:42 KST. **No files deleted or moved.**
Historical inventory status above. Following the user's later deletion instruction,
all 150 confirmed targets were removed at 18:55 KST; see the
[execution result](external-cleanup-result-20261022.md) for preserved files and exclusions.
User request: identify unnecessary project data outside the current working folder,
especially old C: offloads and G: overflow, while preserving ongoing work.

## Scope and size

Current working folder excluded from deletion planning:
`H:\kaggle\competitions\rsna-knee-abnormality-detection`.

Confirmed project-specific external data totals **1,451.65 GiB (1.418 TiB)**.
Sizes are sums of logical file lengths. No sparse, compressed or hard-linked files
were observed in these four project roots; actual reclaimed space is still not
promised until deletion completes. Nested rows must not be added to parent totals.

| Root | GiB | Files | Meaning |
|---|---:|---:|---|
| `C:\kaggle-offload` | 25.98 | 561 | Old R+ fold-1 work only |
| `G:\temp_data\rsna-knee` | 871.76 | 14,954 | K1 caches and moved historical artifacts |
| `G:\temp_data\rsna-h216-oai-v4-v1` | 212.39 | 186,866 | OAI native inputs, rendered shards and assembled arrays |
| `F:\rsna-knee-cache` | 341.53 | 7 | Verified assembled K1 cache copied from G: |

## Priority candidates

These are concrete cleanup candidates, not a claim that a folder can be removed
while an unknown reader is using it. Recheck active paths immediately before any
later destructive step. Preserve research records and the stated retained copy.

| Exact path | GiB | Why it is a candidate | Retain / condition |
|---|---:|---|---|
| `C:\kaggle-offload\sol-1\rplus-f1-v121` | 25.98 | October 12 R+ parent training, smoke, stage and intermediate checkpoint copies; no current R+ process observed | All 7 collected parent output files match the H: copy by fresh SHA256, including model and predictions. Save small unique logs/receipts first; this hash check does not prove every smoke file is duplicated. |
| `G:\temp_data\rsna-knee\l1rco-cache-full` | 683.06 | Approximately 341.53 GiB assembled arrays plus 341.53 GiB per-study shards; assembled cache also exists on F: | Keep `F:\rsna-knee-cache\l1rco-cache-full` and the copy/hash evidence. No active training reader observed. |
| `G:\temp_data\rsna-knee\l1rco-cache` | 38.27 | Earlier partial/pilot cache; current completed training used the full cache | Preserve contracts/receipts; verify no pilot is resumed. |

**Priority candidate total: 747.30 GiB.** This is a logical upper bound
before retaining small unique records. The C: primary parent output is already
retained at `.local/scratch/sol-2/rplus-f1-full/full-parent-root/parent/outputs/`.
The 2026-10-12 archive confirms earlier C: pytest offloads were already removed;
do not count the historical ~145 GiB pytest folders again.

## Selective historical cleanup

| Exact group | Candidate GiB | Scope and reason |
|---|---:|---|
| Old `epochs/*.pt` under `G:\temp_data\rsna-knee\.local` | 91.27 | 129 exact files identified. Intermediate R+, refactor, CPU-smoke and older experiment checkpoints; preserve best/final checkpoint copies and result records. |
| `G:\temp_data\rsna-knee\artifacts\logs\EXP048_output` generated `.npy` files | 20.17 | 11 old derived input arrays. Retain small metadata/receipts. |
| `G:\temp_data\rsna-knee\artifacts\experiments\EXP050_CONVNEXT_BASE_CENTER9_FOLD0` non-best checkpoints | 2.62 | 6 `last.pt` / `fold0_e*.pt` files. Keep `best.pt`, predictions and records. Entire folder is 2.95 GiB. |
| `G:\temp_data\rsna-knee\.local\learning-compact-v1\images.npy` | 2.44 | Old generated example pixels; source recipe must remain. |

**Additional selective candidates: 116.50 GiB**, disjoint from the three
priority rows. All `G:\temp_data\rsna-knee\.local` data totals 113.04 GiB;
the 91.27 GiB checkpoint row is a subset, not an additional copy of that total.
The exact selected paths and byte counts are in the local machine-readable inventory.

The remaining ~19.33 GiB of `.local` includes staged packages, best models,
predictions and metadata. Review individual copies before discarding; do not
delete `.local` wholesale. Historical moves used `/MOVE` or `/MOV` without
junctions, so some G: files are the only surviving copies.

## Conditional archive candidates

| Exact path | GiB | Why removal is conditional |
|---|---:|---|
| `G:\temp_data\rsna-knee\artifacts\releases` | 12.70 | Old `EXP169_A172_v2_stage` and `EXP169_B172_v2_stage`. They contain actual model assets, not just throwaway files. Compare final-stage member hashes first. Two `exp169-assets.zip.bin` archives alone occupy 6.08 GiB, already included here. |
| `G:\temp_data\rsna-knee\storage\external\heliosli-s6-209a32c8-compact-r6cnxfsss` | 0.83 | Move log says the S6 package was accidentally downloaded as R6. Confirm it is not the sole pinned copy needed by a final stage, and preserve source provenance. |
| `G:\temp_data\rsna-knee\k1-diag-ckpt-2021` | 0.75 | Single diagnostic checkpoint. Identify/hash its canonical retained counterpart before removal. |

## Preserve for now

| Path / category | GiB | Reason |
|---|---:|---|
| `F:\rsna-knee-cache\l1rco-cache-full` | 341.53 | Latest fast usable cache, the retained counterpart if the G: copy is removed. Local all-data seed1 completed; keep this copy until final results and any authorized recovery are resolved. |
| `G:\temp_data\rsna-h216-oai-v4-v1` | 212.39 | Explicit standing instruction preserves OAI caches. H216 completed and did not improve its matched control, but that is not a revocation of the preservation instruction. |
| Current H: project, final checkpoints/assets, F4 stage/build/review, submission receipts and ledgers | Not measured here | Outside the requested cleanup scope; needed for current work and final reproducibility. |
| `C:\Users\Taeyang\.cache\huggingface\hub` | 28.79 | Shared cache containing unrelated Qwen, Whisper and other project models. Not an RSNA whole-folder deletion candidate. |
| Credential/config directories and files | Excluded | No credential content opened; never include account files in a cleanup group. |
| Other applications, unrelated datasets, and `G:\Kaggle` older competitions | Excluded | Different projects; folder age or disk usage is not ownership evidence for this task. |

The OAI root consists of cases **103.40 GiB**, rendered shards **54.48 GiB**,
assembled root arrays **54.48 GiB**, and small native/anchor/audit records.
Cases can include materialized native inputs; they are not automatically disposable
because a downstream cache exists. Retention policy must be resolved separately
before any OAI-specific cleanup, and at least the intended reusable cache must remain.

The two clearly named temporary directories `kaggle-srcpkg` and
`rsna-consult-fde03eba2f774caea7e27dfdf57853bc` under the user's Windows Temp
were empty. Broad Temp cleanup and shared Hugging Face token/config paths are excluded.

## Evidence and limits

- Metadata-only scan: 202,684 files across six roots (four project roots plus
  two shared library-cache roots), zero access errors, zero reparse traversal.
  No image, label row or radiology report text was inspected.
- Six relevant local job receipts are terminal SUCCESS: H216 cache and pair,
  H235 all-data seed1, Lateral-OA expert, Synovitis-head job and crop-TTA pilot.
  Current activity includes F4 preparation/review and hidden-submission status
  collection. No active local training process was identified in the scoped
  process snapshot; absence of a named process is not a system-wide open-handle proof.
- C: collected parent outputs: 7/7 fresh SHA256 matches against the retained H:
  parent output directory. No need to reread prediction contents.
- G: to F: cache copy: historical 10-21 robocopy and independent check record
  all 7 files and large-array hashes. Current file sizes match and all five small
  metadata files have matching hashes. The 341 GiB pixel array was not rehashed
  again just to prepare this list; the report reuses the existing evidence.
- Scoped internal link survey: 182 links, no normalized target outside the
  project found; 35 inaccessible old workspace directories limit that survey.
  This does not replace the explicit G:/F: paths in old launch cards.
- The GitHub snapshot (rsna-knee-abnormality-detection) intentionally excludes weights and data. Its
  successful publication is **not a backup of these binaries**.

Source records: `docs/handoffs/handoff-archive-to-20261012.md` lines 5360/5420;
`.local/run-scripts/rplus-f1-pair-prep-h.ps1`;
`.local/scratch/slot-b/disk/move-v1.log`, `move-v2.log` and move manifests;
`.local/scratch/slot-b/vast-next/cache-copy-robocopy.log`;
handoff 10-21 14:41/14:50; D-B-244/270; latest F4 build prompt and job receipts.

Detailed local audit:
`.local/scratch/slot-a/external-cleanup-audit-20261022/` contains `inventory.json`,
`files.jsonl`, `cleanup-candidates.json`, `c-parent-counterpart.json`,
`fg-cache-counterpart.json`, `completed-local-jobs.json` and `workspace-links.json`.
It contains no deletion command. No process, scheduler, active claim or scored
artifact was changed. A later deletion operation must use exact literal paths,
recheck those paths and active dependencies, and preserve the explicitly retained files.
