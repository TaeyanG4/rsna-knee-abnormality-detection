# EXP087 numerical diagnostic independent review

- Owner anchor_inputs;2026-10-04 11:21KST. **HOLD — R1 scan/replace race. Execution not authorized.**
- New guard/runner/tests and protocol/plan only. Binding10checksPASS; synthetic11PASS/1.531s reused. No runs, imports, model/data access, oldhash sweep or source edits.
- R1: `tools/exp087_numerical_diagnostic_v1.py:36–37` calls `p.is_file()` then `p.stat()`. Concurrent `durable_json` lines48–50 replaces `current-attempt.json.tmp`; the temp may disappear between those two calls. Supervisor line233 then throws FileNotFoundError, outer handler line238 records failure and finally kills a healthy child.
- Required: single stat; catch only FileNotFoundError for disappeared files; use stat's regular-file flag. Add a synthetic disappearance/replacement check. Refresh changed new bindings and review only the delta.
- Otherwise acceptable: original generic init/seed/order, strictFP16/TF32off/init_scale1024/loss/AdamW/full3448scheduler preserved. Clip remains `error_if_nonfinite=True`; caught failure uses bare raise. FP64 gradient observations are failure-only/read-only; no recovery or postfailure optimizer/scaler/scheduler step. Found_inf pre-step inspection is nonmutating.
- No evaluation/checkpoint; bounded200 success means only failure absent in that prefix. Original full FAIL remains; no stability/performance claim. Exact decision/plan/budget, fresh output and prior receipts are gated.
- Limits:240s begins at supervise, excludes parent validation; RSS/output are100ms samples, not hard quotas. Logs can overshoot between checks; final receipt follows last size measurement. Telemetry alters timing; reproduction/bitwise equality unproven. Resource PASS needs diagnostic receipt/counters/cleanup too.
- Hashes/evidence: `independent-review.json`, `independent-binding-checks.json`. Reviewer numerical_guard_review found R1 independently. Next: narrow fix → delta review → root exact decision. Do not launch this revision.
