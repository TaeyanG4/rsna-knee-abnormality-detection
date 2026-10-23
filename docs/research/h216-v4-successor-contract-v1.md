# H216 next input stage: actual anchors and full-v4 pilot

2026-10-19 17:30 KST. Implementation contract, not an admitted job. This
continues D240 using its exact completed native8 and geometry8 outputs. No
new cohort selection, label census, native decode or preview is needed.

## Exact eight-study operation

Build a versioned adapter under the existing H216 item. Use frozen
`tools/exp189_s6_src/knee_anchor_bank_v2.py` and `px_cache.py`, mainline
anatomy/geometry/input-provenance modules, and the existing offline anatomy
weight from the K1 registration. Bind their original source/model hashes before
launch. The source inspection is complete; the checkpoint has not been loaded
or hashed beside H210. Model loading/execution needs an actual exclusive slot
and the existing gpu_queue/gpu_job admission when CUDA is used.

Read the eight source study JSONs and 31 native volumes by reference. New
study JSON copies may point to those existing files. Use geometry8's verified
per-study geometry directories; no repeated geometry build. Call the actual
AnchorBank.record for every study, preserving its joint/patella coordinates,
method, reasons and per-series projection. Count segmentation calls, expected
absence of sagittal input, slab-prior fallback and errors separately. The
registered case5 has no sagittal series: an explicit slab-prior result is an
existing recipe fallback, not a segmenter success or prediction failure.

Refuse missing geometry, missing/invalid anchor records, numerical errors and
unrecorded field-centre substitution. Record every out-of-field projection and
the exact fallback which the unchanged renderer consumes; a material problem
keeps input readiness on HOLD rather than silently changing crop geometry.
Do not alter ROI thresholds or the slab-tail rule to make this pilot pass.

Run StudySlices.build with size384/views=v4 and the actual anchor directory.
Preserve exact duplicate collapsing and require accounted-for series (including
explicit duplicate removal), all expected view IDs, nonempty uint8 pixel rows,
finite12-column metadata and authenticated K1 slots. Use the existing
geometry_checked crop/padding guard. Save per-study shards with hashes, row
counts, full-v4 timing, anchor method/fallback and source identities. Separately
measure anchor and rendering time; include model startup in all-up timing.
Compare physical-window sampling against the existing compact training loader
on these same outputs, without a disease-model call or label-based selection.

## Storage and cost admission

Pilot destination: private in-project `h216-oai-v4pilot8-v1`, preserving a
freshly checked100GiB H: reserve. Full-cohort output, if later admitted, goes
under the existing approved `G:/temp_data/rsna-h216-oai-v4-v1` overflow. Before
creation verify the resolved absolute path is a strict descendant of
G:/temp_data, not a reparse-point escape, and is either absent or belongs to
the exact registration. Do not move or delete originals or prior OAI caches.

Use the established per-study shard and final mmap format, not a new training
storage framework. Worst-case rendered arrays plus assembly duplication are
2*172.424GiB=344.848GiB before metadata; retained native arrays add104.271GiB.
These are header bounds, not actual sizes. Raw extraction and overhead must be
added from authenticated archive/header metadata before full allocation, along
with a100GiB free-space reserve. The earlier G: free-space reading is stale
for admission; refresh it. No full-cache allocation is authorized by this doc.

Actual pilot timing supplies a conditional full-preparation budget; eight
cases cannot establish full-cohort tail latency. Include metadata checks,
decode, geometry, anchors, rendering, assembly, paired training, inference and
review in the10-21 usefulness decision. Prefer a measured bounded route over
interrupting H210 merely to expose the next unimplemented step. Prepare the
paired trainer in CPU source before any fit; check actual throughput/memory at
an available slot. Do not make a future physical-resource claim.

## Scientific comparison retained

Both arms start from immutable completed seed0 and use400adaptation plus800
RSNA-recovery optimizer steps. Candidate adaptation uses the30masked native
categorical tasks; control uses a temporary RSNA objective/head and its existing
train roster. Restore the original12-target head/objective for recovery, with
identical recovery examples/windows and reset RNG boundaries. Freeze exact
trainable last-stage names, temporary-head design, optimizer/scheduler/EMA
transitions before execution. These code contracts are still missing, so this
document is not a trainer implementation or launch PASS.

Equal update counts do not mean equal RSNA exposure or isolate domain from
supervision changes. Keep original L2 and no target/ratio/epoch search. Assess
the fixed976/four-ruler candidate versus both control and original seed0, with
all targets and seven-lens limitations. Completed actual results guide H35's
authorized useful-candidate selection/submission; no automatic promotion or
forecast of0.960. Existing H209 remains a separate fixed serving candidate.
