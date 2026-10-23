# EXP087 training preparation handoff

- Owner: dicom_header_io; 2026-10-04T09:05:57.327949+09:00; CPU preparation complete, independent review pending.
- Final protocol: `artifacts/experiments/EXP087_COVERAGE_B8_PROTOCOL_V2/manifest.json`; SHA256 `4682faacbe99267027da6dd189f71dd8d0da4dc6e49e521aa4a3d05c7d551a27`. Source is frozen; see `final-bindings.json`. Original a78e registration is preserved stale source and must not execute.
- Synthetic tests: 12 PASS/0.08 s (`synthetic-tests-final-v2.xml`). No real Windows gate, GPU/model, labels/OOF, decode, census, training, or evaluation executed.
- New runner is a narrow diff from EXP084 B8: unchanged generic initial state/recipe/schedule, coverage wrapper only,32-update pilot with3448 horizon, full fresh generic restart,3448updates/skip0/final-only inference. Existing B comparator reused; no refit.
- Windows implementation: full T3455 membership, zero synthetic labels,8-fixture indices,12 persistent spawn workers/internal1, epoch0->1, separate parent reference, disabled/forced/enabled, cleanup. Actual execution remains pending.
- Root phase decision templates are DRAFT_NOT_AUTHORIZED. Windows180s; GPU pilot180s/full3600s/total3780s;24GiB RSS is a proposed cap for root review; retry0. Projection gate uses last24 whole-update mean <= 0.736212743 s; no speed guarantee.
- GPU runtime reads label/eval CSV then zeros V inference labels; do not describe this as no real-label access at runtime. No intermediate V metric or Gold.
- `runner-baseline.diff`, `implementation-receipt.json`, final bindings and exact execution plans contain full details. Report owner received final handshake; no report scoring performed.
- Next owner root: independent source review -> exact Windows decision -> CPU gate review -> separate pilot/full decisions. No launch by this agent.

Exact Windows command (requires reviewed root decision first):

```text
C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe H:\kaggle\competitions\rsna-knee-abnormality-detection\tools\exp087_training.py run --protocol H:\kaggle\competitions\rsna-knee-abnormality-detection\artifacts\experiments\EXP087_COVERAGE_B8_PROTOCOL_V2\manifest.json --protocol-sha256 4682faacbe99267027da6dd189f71dd8d0da4dc6e49e521aa4a3d05c7d551a27 --mode windows --decision H:\kaggle\competitions\rsna-knee-abnormality-detection\docs\handoffs\evidence\2026-10\2026-10-04-exp087-training-v1\root-windows-execution-decision.json --execute
```
