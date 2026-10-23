# K1 input audit16 result — 2026-10-19

**No new raw-series/position/window mapping failure was demonstrated in the
fixed16-study sample.** This is an engineering finding, not a diagnosis,
clinical reading or explanation of shared model errors.

## Actual checks

| Check | Observed scope/result |
|---|---|
| Sample |4 shared-contradiction,4 family-disagreement,4 random-training,4 random-audit cases from the existing96 roster |
| Raw headers |2584 geometric DICOM headers; no patient-name, report or label-column reads |
| Series |85 raw series,85 cached series; zero missing or removed series in this sample |
| Metadata |All selected series/view row bounds, slice indices/counts and relative physical positions agree with the raw-derived mapping |
| Views |310 series/view middle frames checked, including85 full views; zero constant frames or constant352-centre crops, zero all-zero frames/crops |
| Inference windows |3090 windows from the exact existing eval-window source; zero cross-series/view boundary errors |
| Visual QA |All16 anonymous cases inspected on four sheets; one sagittal representative per study, comparing raw canonical evidence/full saved view/two locals |
| Resources |One sequential CPU run,66.31s;45,711,360 bytes of cached pixel arrays copied for inspection, not a total disk-I/O measurement; zero model/GPU calls |
| Clinical output |Zero diagnoses, zero corrected labels, zero new training |

The visual comparison showed corresponding anatomy and image orientation in
the inspected representatives, without an obvious blank or wrongly rotated
saved frame. Different display normalization and, where applicable, slab
averaging mean this is not a byte-parity test. It does not establish correct
clinical laterality, lesion preservation, adequacy of every sequence or model
accuracy. This was sparse input QA, **not full-study MRI interpretation**.

## Remaining observations

- **Thin-slab source behavior:** the existing source comment says all slices
  contribute, but its remainder rule can discard a short tail. An exact-source
  synthetic example with61 slices at0.7mm averages12 groups of5 and leaves
  one slice unused. This is a confirmed code behavior/comment limitation,
  not an established disease-relevant defect. The actual16-study sample had
  one thin series and **zero discarded tail slices**. It does not explain any
  observed shared error here and does not justify a cache rebuild/retrain.
- **Crop support:** the352 centre crop retains only part of the384 stored
  frame by design. Nonzero-pixel removal ranged0.65%-16.12% across the310
  middle frames. Nonzero pixels are not a lesion mask; this is not a measured
  percentage of lost pathology or evidence that enlarging the crop helps.
- **Anchors/laterality:**11 cases used `skm_tea_joint`, one used
  `skm_tea_joint_avg2`, four used existing `slab_prior`. These are anatomy-anchor
  fallbacks, not failed disease predictions. Nine side decisions used DICOM
  laterality tags and seven used sagittal position; clinical side correctness
  remains unadjudicated. Existing fallback behavior is not a new proven bug.
- All audited headers used the observed transfer syntax in the private receipt;
  no new claim of compressed hidden-data qualification follows from this work.

## Validation, artifacts and decision

Eight targeted tests passed, including exact-source slab remainder equivalence,
short/thick-series behavior, window boundary checks, border-only crop loss,
constant-versus-zero distinction and the frozen16-case eligibility check.
All per-case counts were reconciled with the aggregate result after completion.
B reviewed all four actual image sheets. This is B direct work, not independent
review or a cross-host VERIFIED finding.

Private evidence root: `.local/scratch/slot-b/k1-input-audit16-v1/`:
`REGISTRATION.json`, `selection.private.json`, `RESULT.json`, `cases/`,
`private-images/`, `VISUAL-REVIEW.json`, `B-RESULT-REVIEW.md`.
Raw result SHA256:
`abad56973fa2036221cf07e64f3b1f004769af81f313f4ef75b869d71fdc9de6`.
No image, medical-ID mapping or label file enters the team snapshot.

Close H214 with no implementation change or training follow-up. Existing
D121/125/134/156 coverage work and parked H212 are not reopened; the source-tail
observation alone does not establish benefit from preserving the remainder.
Record these16 as exposed to an engineering image audit, preserving original
pilot manifests. The expert-label reserve still needs real readers and
eligibility/access resolution. Existing H209/H211 work and H210 training
continue; this audit supplies no new0.959/0.960 evidence.
