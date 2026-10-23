# EXP087 numerical diagnostic — R1 repair

- Owner dicom_header_io; 2026-10-04T11:29:23.951226+09:00; CPU repair complete, independent delta review pending. Actual diagnostic remains unauthorized.
- Pre-fix guard/protocol/plan/bindings/review+receipt preserved byte-for-byte with SHA in `pre-r1/preservation-manifest.json`. Original runner and EXP087 source/stage/cache untouched; no old hash sweep.
- Only R1 fixed: `output_bytes` stats each candidate once; ignores only FileNotFoundError from removal; counts regular files by st_mode; PermissionError and other I/O exceptions propagate. `r1-guard.diff`.
- New focused5PASS/0.05s (`synthetic-output-r1.xml`): removal during atomic replacement, file-to-directory replacement, one-stat regular-mode check, PermissionError/OSError identity. Prior11 tests/gates reused without rerun. No model/weights/labels/OOF/pixel/GPU or active trainer imports.
- Refreshed protocol SHA `fcec7360942d2dd90d23bd19138aa6732dfca53658e3770b991247159948b51a`; exactplan SHA `64cdb6b0c503accc7967734cc4923eb500ff2ccabe27c5840d1329c130870dc5`; `bindings.json`. Existing runner SHA reused from saved binding; runner not edited.
- max200attempt/240s/retry0/treeRSS24GiB/12workers/internal1/output16MiB proposal unchanged; output/RSS remain sampled safeguards, not OS quotas. No checkpoint/V inference/score.
- Next: anchor_inputs narrow delta review -> root exact diagnostic decision only. Root template DRAFT_NOT_AUTHORIZED; no launch/retry. New LB observation does not broaden this repair.
- Deadline target11:33/cap11:36 met; root central docs untouched. See `r1-repair-receipt.json`.
