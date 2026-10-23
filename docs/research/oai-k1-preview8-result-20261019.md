# OAI K1 full-field preview8 result

Completed2026-10-19 16:09 KST. B direct work, not independent review.

Eight preregistered OAI cases from the existing confirmation461 extraction were
processed once by the original K1 native decoder and full-field rendering math.
All32 audited series were retained,1,737 bound DICOM hashes checked and1,168
384x384 uint8 rows produced, with finite12-column input metadata. This12-column
metadata is not12 disease predictions. No decode/geometry error, missing series
or slab-tail omission was observed. Sequential CPU execution took29.63seconds;
no checkpoint/model/GPU call, labels, new denoising or prediction.

Seven targeted tests passed, including byte equality of the selected full-field
component to original v4 rendering on a synthetic fixture, deterministic
selection, ID-only metadata overlay and checkpoint-load rejection. Output hashes
for all8 cases and344 PNG files were checked after execution. B inspected all8
processed sheets/32 middle representatives and spot-checked one native and one
RSNA sheet. This does not establish lesion preservation or clinical adequacy.

The source extraction has no `test.csv`; initial registration failed before
pixel reads or output creation. v1.0.1 supplies the frozen ID-only roster through
a virtual directory-reader overlay and leaves the old extraction untouched.
The first failed source-read assumption is retained here, not a failed model or
scientific negative. No preprocessing run was repeated.

## Private artifacts and opened folders

- Existing RSNA input arrays: `G:/temp_data/rsna-knee/l1rco-cache-full`.
- Original downloaded OAI archives:
  `storage/external_data/oai/raw/V00-redl-20261010`.
- New bounded output: `.local/scratch/slot-b/oai-k1-preview8-v1`.

Within the new output, `rsna-existing-preview` shows8 previously inspected
RSNA cases; `oai-native-preview` shows orientation-corrected native OAI with a
display-only window; `oai-processed` contains fullfield384 arrays,352 center-crop
PNGs, metadata and sheets. Native previews are not unmodified raw DICOMs.
RSNA and OAI patients are different, so these are domain/input comparisons,
not paired disease or quality comparisons. Original OAI source images remain
unchanged. The three actual directories were opened in Explorer on user request.

Result SHA256:
`64e451a436810af965350c8e93de3e8eec54d21c5c6b68045b104de531ea14ee`.
Private registration, selection, native metadata, arrays, receipts and images
are excluded from publication. Only aggregate documentation and source/tests
may enter the private team-notes snapshot.

## Limit and next decision

This completes the8case **full-field preview**, not full OAI preprocessing,
SKM-TEA joint/patella ROI extraction, full K1 v4 input qualification or training.
Native-grade task masks, participant exclusions and a paired transfer run are
still prospective under `oai-k1-transfer-plan-20261019.md`. No efficacy or
0.959/0.960 probability follows. Do not rerun the completed preview/audit.

## Private notebook viewer (2026-10-19 16:20 KST)

The private preview directory contains `OAI_RSNA_Compare.ipynb`,
`compare_viewer.py` and `Open_Viewer.cmd`. Run All activates study/series,
1/3/5/7 slice, spacing, crop, display-window and brightness controls.
Rows show original OAI DICOM, optional canonical native OAI, processed OAI
and existing RSNA. RSNA local ROI views are separately selectable.

All notebook cells and actual widget callbacks executed;32raw representative
decodes and all OAI group counts/selected RSNA view bounds checked. B visually
checked a comparison. This is a viewer check, not clinical or independent review.
Source-to-slab grouping includes right-sagittal reversal. Raw/native show one
middle contributor, not the average. RSNA is a different subject and relative
depth is not anatomical registration. All arrays are read-only memory maps.
Notebook/images/mappings/server logs stay local and out of team snapshots.

## 2026-10-19 16:37 input and label-route clarification

D239 confirms that the reused OAI extraction has no Fat_Suppression column;
all32preview series carry unknown K1 slots. Preserve the completed pixel preview,
but resolve this metadata bridge and missing full-v4 views before any training.
The user also requested OAI-informed L2 improvement. Existing weighting/fill
negatives and the missing target/teacher/domain bridge keep it a design reserve;
no current L2 values or weights change. Full review and next steps:
`oai-rsna-domain-gap-review-20261019.md`.
