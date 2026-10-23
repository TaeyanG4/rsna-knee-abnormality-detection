# H216 actual OAI input mechanics v1.0.1

2026-10-19. Additive preparation beside the immutable v1 RSNA mechanics check.
Use only the three predetermined fixed800/pilot8 intersections after the pilot's
bound SUCCESS and B result check. Do not read the running producer's outputs,
decode/render images again, select cases, load checkpoints or initialize CUDA.

The missing check is the actual OAI path from qualified pixels through the
logical1703/physical3 adapter into the native30x4 masked loss. Existing dummy
mapping tests and actual RSNA mechanics do not exercise these real OAI inputs.
Validate every selected study's complete training-window tensors against the
frozen window_rows recipe with its original logical-index RNG, then use only
the first two windows per selected study in one CPU random-weight forward and
backward pass. Bind targets privately to returned logical indices, verify
finite outputs/gradients and no early-encoder gradients, and take zero optimizer
steps. Preserve all targets, masks, source/metadata and the fixed recipe.

Write aggregate evidence and hashes only to a fresh private output. Refuse a
second completed run. This is a bounded input/graph check, not fitting, full800
cache qualification, measured GPU memory/throughput, launch admission, clinical
validation or efficacy. B works solo; the result is not independent PASS.

Builder/verification scope follows the existing BUILDER-CHECKLIST.md and
VERIFY-RULES.md, with the latest solo override. No new agent, monitor or GPU job.
