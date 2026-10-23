# Pilot96 preparation result — 2026-10-19 15:33 KST

Status: **96 provisional studies selected; reading release HOLD.**
Actual readings:0. Adjudicated target cells:0. New trained models:0.

| Role | Studies | Intended use |
|---|---:|---|
| Reader calibration | 16 | Resolve definitions; exclude from later training/evaluation |
| Training candidates | 48 |18 random,15 shared contradictions,9 family disagreements,6 rare weak-label cases |
| Random audit/development | 32 | Feasibility/error-yield audit; never promote to locked test |
| Pilot total | 96 | Each selected study requires full available MRI inspection and all12 targets |

The fixed strata filled without fallback. All96 cases have distinct known
components. A separate200-component holdout reserve was frozen before training
case selection; it is conditional/exposed, not certified as a clean test.
No raw images or reports were opened, rendered, copied or exported.

Preparation reused20 saved outputs from two model families (ResNet50 and
ConvNeXt-S, two seeds/five folds each), authenticated against H105 manifests,
prediction hashes, fold rosters and saved model provenance/inventories.
Disease-fit train/heldout components are disjoint. Seed means stay within
family; twenty outputs are not twenty independent opinions. No model weights
were loaded and no inference/metric experiment was rerun.

The original4343-study OOF universe was aligned to current K1's3190 training
and977 holdout partitions. Percentiles use3174 training studies after16
calibration exclusions. Selection uses actual K1 L2 targets
(75%Dread+25%Aquila). A contradiction is only an error suspicion: new expert
readings could confirm an old-label error, a model error, both or neither.

## Local deliverables

Private root: `.local/scratch/slot-b/expert-mri-pilot96-v1/`.

- `coordinator/roster.private.json`: real-ID/component/token/role mapping;
  coordinator access only, excluded from team publication.
- `coordinator/future-test-reserve.private.json`: conditional future reserve;
  no clean-test claim and no new image labels.
- `readers/A-*.blank.csv`, `B-*.blank.csv`: separate blinded orders, each with
 1152 blank study-target rows across calibration/independent phases.
- `readers/adjudication-*.blank.csv`: blank adjudication templates, not a
  completed third reading. No model scores, old labels, realIDs or selection
  reasons are in reader views.
- `PREPARATION-RESULT.json`: aggregate counts, input/source hashes, private
  roster hash and explicit limitations. No label rows or study IDs.

Templates are not a deployed viewer or access-control system. Opaque tokens
are pseudonyms, not permission to distribute images or the mapping. The
companion reader-form document carries study-level orientation/access fields.

## Direct checks and limits

Eleven targeted synthetic tests passed: group/role separation, deterministic
ordering, repeated groups, tie ranks, contradiction signs and middle-label
exclusion, target round-robin, explicit shortages, insufficient groups, hash
tampering, nonfinite scores and blank/blinded form fields. A second direct
check of written outputs confirmed96 unique cases/groups,1152 blank cells per
reader, exact strata and matching private-roster hash. These are B's checks,
not independent review or evidence of clinical/model benefit.

Direct overlap with Gold58 is zero. **Cross-Gold patient linkage remains
unknown:** the4343-study group contract has no entries for the58 Gold IDs.
Complete prior human-image inspection history and patient linkage are also
not established. Old validation images and indirect teacher/encoder ancestry
remain exposed/uncertain. These gaps prevent a clean-evaluation claim.

Reading tasks remain unreleased until qualified MSK readers/adjudication are
available, the target manual is signed off, permitted full-study viewer access
is arranged, and exposure/linkage eligibility is resolved. If the roster must
change, version it before new image labels are seen. No fabricated AI labels,
automatic negative filling or training on these blank forms is allowed.

Next permitted step is staffing/access/eligibility resolution followed by the
16-case calibration. Current GPU training and existing candidate work continue
independently. This pilot has no credible pre-final-freeze score-gain promise.
