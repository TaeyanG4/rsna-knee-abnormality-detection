# H-B-228 / astra-b4 — R6 stride diagnostic and candidate

Version 1.0.0, 2026-10-20. Built locally only. Independent review and Kaggle execution are pending. Neither unchanged predictions nor a two-hour saving has been established. No GPU, network, push, submission, upload, or subagents were used.

## Exact change and raw-row dependency

The INF3 hidden parent is `artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.46-r6w958inf3`; the shared x3 ancestor is `v1.1.64-r6w958k1x3sr20` in the same directory. The current `tools/k1n_stage_v1.py` v1.2.0 packages runtime `tools/k1sr_runtime_v7.py` v7.2.0.

The embedded `exp169_r7_port.py::worker_args` reads `manifest.get("eval_stride", 3)` and always appends `--flip-tta`. `storage/external/heliosli-r6-fe055406-v1/manifest.json` specifies stride 1, size 384, crop 352, v4, slices 3 and phys_mm 3.5. The candidate changes only this embedded argument to `"2"`, plus the embedded port version and chained SHA pins. The original manifest is untouched.

`tools/k1_shared_render_v2.py::SharedR6` originally accepts only R6 stride 1. The candidate's embedded copy changes that exact acceptance tuple to stride 2, with version `2.0.1-hb228-r6s2`. Its builder hook captures the **full raw px/meta/blocks before R6 window selection**. `from_rows` independently selects K1 stride-2 windows. Thus R6 stride 2 must not subsample the raw capture. The original-side rerender removes the forced-side correction temporarily and restores R6 state afterward.

K1 remains the registered v2 + v3g seed0 + v3g seed1 checkpoint set, all stride 2, with O retained:

```text
rank(0.8 * rank(INF3_R6_stride2_float32_csv)
   + 0.2 * rank(mean(per-target rank(K1_model_FP16))))
```

No per-target alpha is introduced. Runtime v7 code, filename, launch/import/pool/thread/timing behavior, renderer and anatomy weights remain unchanged relative to the configured v7 parent. Each stage includes `full-payload-notebook.diff`, `payload-scope.json` with unchanged-member hashes, and the complete inherited `ancestor-v6-to-v7.diff`; inherited v7 changes are not concealed as part of the stride patch.

## New files and stages

New code: `tools/hb228_stage_v1.py`, `tools/hb228_pair_v1.py`, `tools/hb228_compare_v1.py`, `tools/hb228_cpu_smoke_v1.py`.

New config/test: `configs/k1n/hb228-r6s2-k1x3s2.json`, `tests/test_hb228_v1.py`.

All stage paths below are relative to `artifacts/experiments/EXP169_RELEASE_C85R7/`:

| Stage | Kernel ID | Purpose |
| --- | --- | --- |
| `v1.1.68-hb228-r6-pair` | `taeyangg4/knee-hb228-r6-stride-pair-v1` | R6-only stride 1 then 2, one label-free N32 roster; NEVER SUBMIT |
| `v1.1.68-r6s2k1x3s2-n32` | `taeyangg4/knee-lb-r6s2k1x3s2-v1-n32` | Full candidate N32; NEVER SUBMIT |
| `v1.1.68-r6s2k1x3s2` | `taeyangg4/knee-lb-r6s2k1x3s2-v1` | Candidate, unqualified and unpushed |

Preparation registers exact config, embedded source hashes and notebook/metadata hashes. `execution_verified=false`, `independent_review=REQUIRED`, and `qualified_for_submission=false` are intentional. There is no separate register/admit command or GPU admission in this build. Only new versioned files were created; no scored stage, original source, manifest or existing receipt was edited, so no backup of an existing file was needed.

The pair notebook uses the inherited R6 `compact_infer.py`, without K1 injection. It preserves pinned offline wheels, codecs, weights, flip and controller behavior, and launches fresh sequential controllers on the same roster. R/O/R224/K1 inference is not run. Metadata retains inherited dataset mounts, including unused parent/K1 assets. The pair enforces 32 IDs, 12 finite probabilities, zero fallback and the actual stride/flip arguments. It refuses a root `submission.csv`; controller-local submission files remain inside each arm.

Pair outputs in `/kaggle/working/hb228-pair/`:

- `r6-stride1.csv`, `r6-stride2.csv`, `comparison.json`, `receipt.json`
- `n32/selection.json`
- `stride1/timing.json`, `stride2/timing.json`, plus full per-arm provenance and controller outputs

The comparator aligns rows by ID, rejects invalid schemas/duplicates/nonfinite values, and reports maximum absolute difference and per-target Spearman. Constant ranks are explicitly undefined. Optional existing rulers produce only aggregate per-target AUCs/deltas, with missing/single-class cases explicit. Measurements are exposed exploratory evidence, not clean OOF or automatic qualification.

## Executed CPU checks

Use the existing Python, without rebuilding any environment:

```powershell
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:S6PLUS_PIN_DLL = 'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
& $py -c "from tools import s6plus; import pytest; raise SystemExit(pytest.main(['-q','tests/test_hb228_v1.py','tests/test_k1_shared_render_v1.py','tests/test_k1sr_release_v1.py','tests/test_k1sr_release_v2.py','tests/test_k1sr_release_v3.py','tests/test_k1sr_release_v5.py','tests/test_k1sr_release_v6.py','tests/test_k1sr_load_timeout.py','tests/test_k1sr_speedup_v6.py','tests/test_k1n_v7.py','tests/test_k1n_stride_v710.py','--basetemp=.local/pytest-tmp/astra-b4']))"
& $py -m tools.hb228_cpu_smoke_v1
& $py -m tools.hb228_compare_v1 --stride1 .local/scratch/slot-b/vnext-stress-n32-visible/hb116-run/public/p956-run/s6/predictions.csv --stride2 .local/scratch/slot-b/vnext-stress-n32-visible/hb116-run/public/p956-run/s6/predictions.csv --rulers-root . --out .local/scratch/astra-b4/hb228/comparator-cli-cpu-smoke.json
& $py -m tools.hb228_stage_v1 prepare
& $py -m tools.hb228_stage_v1 validate
```

Results:

- Full relevant suite: **388 passed**, one existing constant-input Spearman warning, 44.71 s. Log: `.local/scratch/astra-b4/hb228/pytest.log`. Includes strict package/manifest/adapter contracts and mocked pair execution/export/fallback regressions.
- Real DICOM CPU smoke: **PASS**, 1 study, **177 windows**, original K1 stride-2 versus R6-stride-2 shared capture; images and metadata bit-exact, original-side rerender bit-exact, R6 state restored, CUDA not initialized. Receipt: `.local/scratch/astra-b4/hb228/cpu-smoke/comparison.json`. This is renderer/window coverage, not classifier/FP16/full-parent inference.
- An earlier CPU attempt was invalidated correctly by its source-stability guard while the builder was edited; the frozen-source rerun passed. The final process log is `.local/scratch/astra-b4/hb228/cpu-smoke.log`.
- Real 32-study saved predictions passed comparator CLI and existing-ruler aggregation. Self-comparison maximum absolute delta was **0**; all seven weak rulers loaded, with 11 defined AUC targets each; Gold had no defined AUC on this roster. This validates comparator I/O/aggregation only, **not R6 stride parity**. Output: `.local/scratch/astra-b4/hb228/comparator-cli-cpu-smoke.json`.
- `prepare` and `validate`: **PASS_LOCAL_ONLY** for all three stages. Logs: `.local/scratch/astra-b4/hb228/prepare.json`, `validate.json`. All embedded Python and notebook code cells compile; exact stage inventories and bytes validate.

## B-only integration commands — not executed

After the required independent review and B's resource/quota checks, run the pair diagnostic first. Do not submit either diagnostic kernel.

```powershell
kaggle kernels push -p artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.68-hb228-r6-pair/kernel
```

After B downloads the diagnostic outputs into `.local/scratch/astra-b4/hb228/pair-visible/`, aggregate existing local rulers:

```powershell
& $py -m tools.hb228_compare_v1 --stride1 .local/scratch/astra-b4/hb228/pair-visible/hb228-pair/r6-stride1.csv --stride2 .local/scratch/astra-b4/hb228/pair-visible/hb228-pair/r6-stride2.csv --selection .local/scratch/astra-b4/hb228/pair-visible/hb228-pair/n32/selection.json --rulers-root . --out .local/scratch/astra-b4/hb228/pair-visible/comparison-with-rulers.json
```

That future-output command is intentionally unexecuted: the diagnostic has not run. After reviewing its actual predictions and timing, qualify the full candidate on N32:

```powershell
kaggle kernels push -p artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.68-r6s2k1x3s2-n32/kernel
```

Only after satisfactory independent result review and full-candidate qualification:

```powershell
kaggle kernels push -p artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.68-r6s2k1x3s2/kernel
```

No submission command is provided. Push commands are a handoff to B, not authorization for this lane to execute them.

## Expected savings and unresolved risks

Existing N32 receipt observations: single-K1 parent 1428.603 s / S_R7 983.414 s (**68.84%**); x3-v1 1644.312 / 1189.717 (**72.35%**); x3-v3 1769.555 / 1240.639 (**70.11%**). These S_R7 phases include in-pass K1, and parent totals also include K1. They are **not pure stride-scalable R6 classifier timings**.

The idealized arithmetic `0.69 * 0.5 = 34.5%` corresponds to roughly 2.31–2.38 h of a hypothetical 6.7–6.9 h parent. This is an optimistic scenario, not a measured bound or forecast. Rendering/loading/stackers remain, and K1 stride 2 adds windows relative to stride 3. **About two hours saved is the hypothesis to test.**

Stride changes can alter lesion sampling, probability values, ranks and stacker inputs; close average values alone do not establish acceptable target-level behavior. N32 AUC is noisy and some targets lack both classes. Fixed stride1-then2 order permits cache-warming bias. The R6-only pair isolates R6 behavior but cannot measure combined K1 overhead, candidate runtime, FP16 parity or hidden-scale failure risk. Full-candidate N32 must check real shared-row capture, K1 fallback, codecs, output validity and timing. Hidden runtime <=9 h remains unmeasured for this candidate.

SUB-AGENTS: none.
