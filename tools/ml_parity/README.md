# ml_parity (vendored)

Generic cross-environment parity scripts copied unmodified from the `numerical-parity-skill` source (renamed from `ml-parity-skill` on 2026-10-06)
(`H:/dev/kaggle-data/numerical-parity-skill/scripts`, installed for Claude Code at `~/.claude/skills/numerical-parity-skill`)
so Codex sessions can use them too. Usage: see `SKILL.md` in the skill, or `python <script> --help`.

- `parity_runtime.py` — configure_parity / runtime_info / diff / dump_activations. Not wired into
  `training/src/raptor_runtime.py` (SHA-pinned by frozen B_RN3F contracts; do not edit it).
- `compare_arrays.py` — compare npy/npz/csv/pt with pre-registered tolerances.
- `hash_stage.py` — byte or array-content hash manifests and diffs.

Project-specific score/order check built on the same ideas: `tools/ml_parity_order_check.py`.
