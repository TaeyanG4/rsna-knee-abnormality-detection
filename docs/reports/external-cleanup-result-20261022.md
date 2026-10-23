# External storage cleanup result — 2026-10-22

Version 1.0.0. Completed 2026-10-22 18:55 KST.

Following Taeyang's explicit instruction to delete the previously listed unnecessary
external data, all **150 confirmed targets** were removed: three directory trees
and 147 individually selected files. **13,917 files / 927,505,760,037 logical bytes
(863.81 GiB)** were deleted, with **zero failed deletions**. All 150 target paths
were verified absent. This was permanent removal, not a move to the Recycle Bin.

The earlier [inventory](external-cleanup-inventory-20261022.md) remains a historical
pre-deletion record. Its conditional and protected groups were excluded from this
operation. This report does not claim that all external project data was deleted.

## Removed data

| Target | Logical GiB | Result |
|---|---:|---|
| `C:\kaggle-offload\sol-1\rplus-f1-v121` | 25.98 | Removed after retaining final parent outputs and archiving records |
| `G:\temp_data\rsna-knee\l1rco-cache-full` | 683.06 | Removed assembled copy and shards; F: copy retained |
| `G:\temp_data\rsna-knee\l1rco-cache` | 38.27 | Superseded pilot cache removed |
| 129 exact old epoch checkpoint files under `G:\temp_data\rsna-knee\.local` | 91.27 | Removed selected intermediate files only |
| 11 generated EXP048 arrays | 20.17 | Removed selected arrays; other files retained |
| Six EXP050 non-best checkpoints | 2.62 | Removed; `best.pt`, predictions and records retained |
| Old `learning-compact-v1\images.npy` | 2.44 | Removed; surrounding records retained |

## Preserved evidence and active work

Before removal, **4,913 research-record files / 88,313,306 bytes (84.22 MiB)**
were copied into the current H: workspace and verified by SHA256. These include
small source/configuration files, logs, receipts and result records from the three
deleted trees. Files were copied as opaque bytes; no label rows or radiology text
were displayed. This local archive is excluded from GitHub publication.

Local evidence directory:
`.local/scratch/slot-a/external-cleanup-audit-20261022/`.

- `preserved-records/`: verified local record archive with original drive/path layout.
- `execution-plan.json`: exact 13,917-file snapshot, retained-file metadata and archive hashes.
- `deletion-log.jsonl`: durable per-file deletion ledger.
- `deletion-result.json`: completed operation, target absence and volume-space observations.
- Original inventory, counterpart hash checks, completed-job receipts and process snapshots remain.

Execution-plan SHA256:
`80788e195e2a9257f6c38ee7b47567784881e575608ae4f532b1e76f448c708a`.

The seven R+ parent output files retained on H: were rechecked against the earlier
source SHA256 values before deletion. The F: cache's seven files and the EXP050
best checkpoint were also checked for existence and expected metadata. All 15
retained key files had unchanged size and modification time after removal.
The previous full G-to-F array verification was reused; the 341 GiB pixel array
was not rehashed again during this cleanup.

Preserved locations and work:

- Current `H:\kaggle\competitions\rsna-knee-abnormality-detection` workspace,
  final models/stages, submission receipts/ledgers and current F4 work.
- `F:\rsna-knee-cache\l1rco-cache-full` (341.53 GiB), the retained usable K1 cache.
- `G:\temp_data\rsna-h216-oai-v4-v1` (212.39 GiB), under the existing OAI retention instruction.
- Other G: best/final checkpoints, prediction files, records and unselected artifacts.
- Shared model caches, credentials, unrelated datasets and applications.

No process, GPU job, submission, automation, active compute claim or B-owned
scientific item was changed. Slot A remained released for this bounded cleanup.

## Observed free-space increase

| Volume | Free bytes immediately before deletion | Free bytes after deletion | Observed increase |
|---|---:|---:|---:|
| C: | 540,632,129,536 | 568,191,705,088 | 27,559,575,552 bytes / 25.67 GiB |
| G: | 2,818,806,972,416 | 3,718,453,014,528 | 899,646,042,112 bytes / 837.86 GiB |
| F: | 69,245,227,008 | 69,245,227,008 | 0 bytes |

Combined C:/G: free-space increase was **863.53 GiB**. This differs from logical
deleted bytes because volume allocation and concurrent writes affect the observed
free-space counters. H: stores the preserved records and operation logs; it was
not a deletion target.

## Conditional groups still retained

These three groups total approximately **14.27 GiB** and were already conditional
in the inventory. A complete retained-counterpart/pinned-member check was not
established for them, so they were not silently added to the confirmed deletion set.

| Retained group | GiB | Remaining condition |
|---|---:|---|
| `G:\temp_data\rsna-knee\artifacts\releases` | 12.70 | Old A172/B172 stages contain model assets and packaged code; verify retained member hashes before disposal |
| `G:\temp_data\rsna-knee\storage\external\heliosli-s6-209a32c8-compact-r6cnxfsss` | 0.83 | Confirm a retained equivalent and no unique pinned source dependency |
| `G:\temp_data\rsna-knee\k1-diag-ckpt-2021` | 0.75 | Identify an identical retained canonical checkpoint |

## Validation scope

The exact candidate paths, counts, lengths and modification times matched the
pre-deletion inventory. All resolved paths were within the three named external
trees or the exact selected-file list; reparse traversal was refused. The script
checked active process command lines for the relevant paths and rechecked archive
hashes before deletion. Command-line inspection is not an exhaustive open-handle
audit. File removal used literal paths; directory removal was nonrecursive and
could not consume unexpected new children. No failed file removal was forced.

The private notes repository is a documentation snapshot, not a backup of removed
weights or data. It receives this aggregate report only, not the local record
archive, data, labels, weights or credentials. Scientific results remain pending
according to the separately preserved pre-result report.
