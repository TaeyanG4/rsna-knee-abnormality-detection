# H216 exact production-input binding v1.0.0

2026-10-19. Metadata-only continuation of D240, not a new cohort census.
Bind existing1703native-task studies in their original sorted order,3190RSNA
training/977holdout, original L2 target order/hash and fixed400/800 schedules.
Use the pair core's seed21601 permutation:400batches of2 consume800distinct
OAI studies. Prepare only these predeclared inputs; preserve the1703logical
index space and full target array so no permutation or per-study window RNG
changes when the physical cache is smaller. No pathology/label/performance
criterion selects these800. This is not full-cohort OAI exposure.

The ScheduledStudies adapter maps logical to physical cache indices, preserves
original index in collate, and adjusts the frozen loader's seed by logical-minus-
physical index. Thus seed+epoch+index gives exactly the original window stream.
An unscheduled read fails; no replacement draw or partial-cache silent fill.
Both arms use identical predeclared recovery order and original3190RSNA data.

Private outputs contain exact roster, native masks/grades, existing L2 arrays,
series references and a source-bound registration. Only aggregate counts are
reported. Known557evaluation participants are excluded using existing evidence;
this does not certify complete teacher/anchor ancestry or clean validation.
No raw DICOM census, images, checkpoint, fit, GPU or cache allocation occurs.

After the actual queued v4 pilot succeeds and its outputs are checked, reuse any
selected pilot studies and prepare only missing inputs. Bind fresh approved
G:/temp_data capacity and pilot runtime before production allocation. The input
binding itself is not a launch-ready cache or efficacy result.
