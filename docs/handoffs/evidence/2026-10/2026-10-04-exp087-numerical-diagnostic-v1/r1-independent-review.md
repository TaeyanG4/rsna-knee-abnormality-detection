# EXP087 R1 delta review

- Owner anchor_inputs;2026-10-04 11:33KST. **PASS for root exact decision; execution not authorized.** No blockers remain in this delta.
- Single `stat()` per candidate; only FileNotFoundError skipped; `stat.S_ISREG` counts regular files. PermissionError/other OSError propagate. This resolves the temporary-file replacement race. Current source and `r1-guard.diff` checked.
- New5 syntheticPASS/0.050s receipt and test source read, no tests rerun. Coverage: one stat, atomic replacement, file→directory, unchanged PermissionError/OSError propagation. Prior11tests and prior recipe/failure-semantics review reused.
- Actual guard/test/protocol/plan hashes checked once; all match receipt/bindings/protocol. Runner recorded hash unchanged, not rehashed. Child argv/budget/environment unchanged; output absent. Protocol changes only bindings/revision/prior-registration metadata.
- Protocol `fcec7360942d2dd90d23bd19138aa6732dfca53658e3770b991247159948b51a`; plan `64cdb6b0c503accc7967734cc4923eb500ff2ccabe27c5840d1329c130870dc5`. Full hashes/checks: `r1-independent-review.json`.
- `pre-r1/preservation-manifest.json` has10 prior records including original HOLD; no snapshot rehash sweep. No model/data/GPU access, training imports, source6 reads or central edits.
- Prior limitations remain:240s supervisor timer, sampled resource limits, telemetry timing effects, bounded200 is not full stability/performance. Root separate exact execution decision remains required.
- Additional reviewer hit usage limit before delta findings; no retry. Direct anchor_inputs review completed. B/RN submission score does not change this numerical review; no scoring follow-up.
