# H224 fold assets and ensemble templates v1.0.0

Local CPU packaging only. No GPU, Kaggle/API access, network, credentials,
training, upload, submission, or child agent. New files only; no existing
generator, training source, scored stage, or receipt was edited.

## Files and contract

- `tools/k1_fold_dataset_v1.py`: verify a completed producer `EXPORT.json`
  and `SHA256SUMS`, then copy exactly the deploy checkpoint, frozen serving
  trainer, `asset-manifest.json`, and private `dataset-metadata.json`.
- `tools/k1_fold_configs_v1.py`: verify real base/fold asset hashes, materialize
  a standard `k1n-config/v1` config and a hash-bound runtime projection sidecar.
- `configs/k1n/five-model-folds-stride2.template.json`: v2, s0, s1, f1, f2.
- `configs/k1n/six-model-folds-stride2.template.json`: the above plus f3.
- `tests/test_k1_fold_assets_v1.py`: synthetic export failure cases, actual seed
  checkpoint byte roundtrip, config/stage compatibility, and CLI CPU checks.

Both templates fix stride=2, drop_o=true, alpha=0.25, encoder_chunk=32.
The existing K1N implementation averages per-target ranks across the selected
models, then ranks the 75% selected parent / 25% K1 blend. Templates have a
different schema and no fake hashes; the existing stage generator rejects them.
Only a successful materialization creates an executable config.

The producer exports seven files, excluding its generated trainer. This tool
reconstructs that trainer using the exact fold producer transform from the
SHA-pinned frozen source, verifies the reconstructed hash against `prepare.json`,
and compares the Net/OFLEncoder ASTs. The dataset includes the frozen serving
trainer, as the seed datasets do; it records the distinct training-source hash.
Neither source is imported/executed as a model during packaging.

Export verification hashes the holdout CSV without parsing or printing rows.
No holdout CSV, registration roster, labels, runtime environment file, or
training state is copied into the dataset. The manifest retains hash provenance,
`EXPOSED_EXPLORATORY`, and `clean_oof=false`. Each checkpoint keeps its real
fold filename, e.g. `weights/E_prior_s0_f1.pt`.

All input inventories, receipts, contract/fold/model identities, training
completion counts, and source hashes must agree. Traversal, linked paths,
duplicate JSON/SHA entries, extra files, placeholders, and copy drift fail closed.
Existing destinations are immutable: rerun `verify-dataset` or `validate`.
The packager never deletes a partial output; metadata is written after the
copied asset hashes pass.

## Runtime decision

These are supplied estimates, not measured hidden qualification:

```
one stride-2 K1 model = 0.45 h * 1.44 = 0.648 h
projection = parent - 1.1 h O - included_parent_K1_count * 0.45 h
             + new_model_count * 0.648 h
```

| Meaning of the 6.5–6.9 h baseline | Five models | Six models |
|---|---:|---:|
| K1 not included (conservative default) | 8.640–9.040 h: HOLD | 9.288–9.688 h: HOLD |
| Three stride-3 K1 models already included | 7.290–7.690 h: estimate passes | 7.938–8.338 h: HOLD |

The user prompt did not disambiguate the baseline scope. A clarification was
requested; absent a response, templates retain `parent_k1_models_stride3=0`.
Do not subtract 1.35 h without evidence that the baseline actually includes the
three K1 classifiers. If confirmed, record that evidence in `runtime.basis`
and set `parent_k1_models_stride3=3` in a reviewed template. This is an explicit
baseline accounting change, not a speedup measurement.

The upper endpoint must be <=8.0 h. The decision uses exact decimal arithmetic,
not a rounded displayed value. The six-model template stays HOLD even if the
baseline includes three K1 models. No CLI option relaxes the 8.0 h limit or
lowers supplied timing assumptions. Changed measured timing assumptions require
a versioned update and review. O saving is gross; unmeasured overhead remains.
Estimate PASS is not GPU/submission qualification or an independent review.

## Exact local commands (PowerShell, project root)

```powershell
$py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$task = '.local/scratch/astra-b3/h224-fold-assets-v1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'

& $py -B -m pytest tests/test_k1_fold_assets_v1.py tests/test_k1n_v7.py tests/test_k1n_stride_v710.py --basetemp=.local/pytest-tmp/astra-b3 -q

& $py -B -m tools.k1_fold_dataset_v1 verify-dataset --dataset .local/scratch/astra-b1/k1-shared-render/dataset-k1-v3gs1
& $py -B -m tools.k1_fold_configs_v1 project --template configs/k1n/five-model-folds-stride2.template.json
& $py -B -m tools.k1_fold_configs_v1 project --template configs/k1n/six-model-folds-stride2.template.json
```

`project` prints its projection and returns 2 on HOLD. Current templates both
return 2; this is expected, not a successful candidate qualification.

After B has the **completed** export and all seven manifest members locally:

```powershell
& $py -B -m tools.k1_fold_dataset_v1 verify-export --export .local/scratch/astra-b2/h224-folds-v1/f1/EXPORT.json
& $py -B -m tools.k1_fold_dataset_v1 package --export .local/scratch/astra-b2/h224-folds-v1/f1/EXPORT.json --output "$task/dataset-k1-v3gf1" --dataset-id taeyangg4/knee-k1-v3gf1-v1-assets
& $py -B -m tools.k1_fold_dataset_v1 verify-dataset --dataset "$task/dataset-k1-v3gf1"

& $py -B -m tools.k1_fold_dataset_v1 package --export .local/scratch/astra-b2/h224-folds-v1/f2/EXPORT.json --output "$task/dataset-k1-v3gf2" --dataset-id taeyangg4/knee-k1-v3gf2-v1-assets
& $py -B -m tools.k1_fold_dataset_v1 package --export .local/scratch/astra-b2/h224-folds-v1/f3/EXPORT.json --output "$task/dataset-k1-v3gf3" --dataset-id taeyangg4/knee-k1-v3gf3-v1-assets
```

The f2/f3 paths above assume B restores the remote export tree to those local
paths. If it is preserved elsewhere, pass that actual `EXPORT.json` path;
the packager resolves only its seven allowlisted relative members. It does not
require Linux absolute paths, download anything, or touch a live training job.
Never run these commands against an unfinished export.

Choose unused kernel/version identities before a real release. These explicit
example identities are for the requested templates; these commands currently
return HOLD and create no config:

```powershell
& $py -B -m tools.k1_fold_configs_v1 prepare --template configs/k1n/five-model-folds-stride2.template.json --fold-datasets "$task/dataset-k1-v3gf1" "$task/dataset-k1-v3gf2" --output "$task/config-five" --kernel-id taeyangg4/knee-lb-k1n5folds2p25-v1 --release-version 1.2.0-k1n5folds2p25
& $py -B -m tools.k1_fold_configs_v1 validate --template configs/k1n/five-model-folds-stride2.template.json --fold-datasets "$task/dataset-k1-v3gf1" "$task/dataset-k1-v3gf2" --output "$task/config-five" --kernel-id taeyangg4/knee-lb-k1n5folds2p25-v1 --release-version 1.2.0-k1n5folds2p25
& $py -B -m tools.k1_fold_configs_v1 prepare --template configs/k1n/six-model-folds-stride2.template.json --fold-datasets "$task/dataset-k1-v3gf1" "$task/dataset-k1-v3gf2" "$task/dataset-k1-v3gf3" --output "$task/config-six" --kernel-id taeyangg4/knee-lb-k1n6folds2p25-v1 --release-version 1.2.0-k1n6folds2p25
```

A successful `config.json` uses the unmodified `tools/k1n_stage_v1.py` schema.
No GPU register/admit command or release-stage preparation is added by this
packaging task. CLI success paths are tested against synthetic completed exports;
real H224 completion is not implied.

The resulting four-file dataset directory is suitable for this **B-only**
command (not run by this builder):

```powershell
kaggle datasets create -p "$task/dataset-k1-v3gf1" -r zip
```

Metadata sets `isPrivate=true`; the create command deliberately has no `--public`
flag. B must retain the normal independent review and external publication gates.

SUB-AGENTS: none.
