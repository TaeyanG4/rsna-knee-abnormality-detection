# EXP087 numerical diagnostic CPU preparation

- Owner dicom_header_io; 2026-10-04T10:58:12.616059+09:00; CPU preparation complete, actual execution not authorized. Root central docs untouched.
- Read M02/M04/M05/M07/M09/M10/M11 and root-static-failure-triage: original gradient-norm FAIL preserved; no performance conclusion.
- New runner/guard only; frozen source checks all unchanged. Protocol `protocol.json` SHA256 `9e8ae220a40b6849d6d6800b84bcb48d530e3d9c4537ea21a84b11cfdebfd627`; exact `execution-plan.json`; `bindings.json`.
- Focused synthetic11PASS/1.53s + compilePASS. No model/weights/real labels/OOF/pixel/GPU reads or training, old gates/tests not rerun.
- Same generic initial state/seed42/order/strictFP16/TF32off/init_scale1024/optimizer/loss/full3448scheduler. Normal attempt scaler.step/update remains; failure blocks all recovery/update/backoff.
- Durable current/last-completed attempt/update/epoch/batch UID SHA, input/logit/loss flags/scalar. Failure captures scale/found_inf, gradient count/dtype/finite/nonfinite/max/native norm and FP64 reference aggregates only then; writeback0; bare-reraise original child failure; worker cleanup.
- Proposed diagnostic only: max200attempt/240sec/retry0/treeRSS24GiB/12workers/internal1/output16MiB. Parent termination/resource outcome remains separate from numerical observation. No checkpoint/V inference/score.
- Prior8T/315stage/Windows/pilot reused, no rerun. Pilot does not prove longer stability; diagnostic timing may change with telemetry; failure may not reproduce.
- Next root: anchor_inputs independent source review -> reviewed exact decision -> optional actual diagnostic. DRAFT template is not authorization. No auto-launch or retry.
- Root resumed target11:05/cap11:10; old CPU-decision timestamps preserved. See `implementation-receipt.json` and `runner-original.diff`.
