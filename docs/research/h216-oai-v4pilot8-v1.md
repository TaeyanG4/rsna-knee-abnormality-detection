# H216 actual-anchor pilot adapter v1.0.0

Implementation of the17:30 successor contract,2026-10-19. B works alone under
the current user override; B checks are not independent review. No new cohort,
label change, original-source modification or full-cohort allocation.

`tools/h216_oai_v4pilot8_v1.py` reuses D240's native8 and geometry8 by reference.
It pins the original three serving/loader modules, binds existing input receipts,
native arrays and geometry, and records the original anatomy-weight SHA without
opening it during CPU preparation. At exclusive execution it verifies that SHA,
loads the unchanged SKM-TEA model, computes actual anchors and invokes the
unchanged384/v4 renderer. No synthetic anchor is used in the actual job.

The adapter rejects model errors hidden inside AnchorBank fallback reasons,
missing geometry/projections, missing or unaccounted series, lost views, invalid
metadata/slots, empty pixels, crop/padding violations and mmap/window drift.
Out-of-field projections retain the exact original field-centre substitution
but produce readiness HOLD, not a repaired crop or training PASS. Explicit
slab-prior fallback is preserved; the no-sagittal case must have zero segmenter
calls. Anchor fallback does not mean disease-prediction failure.

CPU tests exercise adversarial missing/error/NaN/metadata/view/offset paths and
full serving-to-loader window parity. A one-series real-native CPU smoke uses
an explicitly synthetic centre solely to verify mechanical rendering/storage;
it loads no model, initializes no CUDA and is never evidence of actual ROI.

After register/test/smoke and an exact bound B direct precheck, the only actual
entrypoint is `gpu_queue.py` -> `gpu_job.py` -> adapter `run` -> owned `worker`.
The existing exclusive claim, known process identity and fresh device checks
are required. The same900-second zero-progress watchdog remains; telemetry
advances on model forwards, completed render rows, copies and parity checks.
Peak must stay within12GiB and desktop+peak+2GiB within device memory. Pilot
storage reserves100GiB plus4GiB initial room. No self-imposed runtime deadline,
automatic retry, new monitor or physical claim before launch.

Each study retains raw anchor record, method/reasons/projections, full-v4 pixel
and metadata shards, geometry coverage, serving-window hashes and runtime.
The final existing mmap format is checked against the original training loader.
All artifacts containing identities remain private. Source/model startup,
anchors, renderer and all-up time are separate. A successful eight-case pilot
does not establish full-cohort readiness, clinical validity or transfer gain.

Next: actual pilot outcome and bounded full-input preparation, then the fixed
same-start400+800 OAI/RSNA-only pair. The adapter does not train or submit.
