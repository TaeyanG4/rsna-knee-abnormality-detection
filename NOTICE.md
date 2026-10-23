# Notice: third-party work, data and images

Our own code and documents in this repository are released under the Apache License 2.0 (see [LICENSE](LICENSE)). Third-party work keeps its own licence and copyright. Nothing listed below is redistributed here unless stated: we publish no weights, no datasets and no third-party notebook outputs.

## Public Kaggle work this project builds on

We thank the authors. Kaggle notebook code is published under the Apache License 2.0; datasets and model weights carry the licence their authors chose. Every Kaggle page below was checked against Kaggle's public listing (`kaggle datasets list` / `kaggle kernels list` / `kaggle models list`, read-only) on Oct 23, 2026; an entry marked "exact page not verified" links the author's profile instead.

### Members of the final submissions and their serving code

| Work | Kaggle page | Author | How we used it |
|---|---|---|---|
| All-data CoAtNet "Raptor" SWA checkpoints at 384 px and 224 px, fine-tuned with OAI images | [datasets/nartaa/rsna-knee-publication-swa-weights-20261007](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) | nartaa | R384 and R224, the R member of every blend from Oct 9; attached to both final kernels |
| Compact ConvNeXt-S 2.5D bundle R6 `fe055406` | [datasets/heliosli/rsna-knee-r6-frozen-fe055406](https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406) (version 1); the bytes we served came from the versioned copy [datasets/rostigor/rsna-knee-r6-fe055406](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406) | heliosli (copy: rostigor) | the compact member R6 in both finals, served from a private copy |
| OrthoFoundation code dataset (compact rendering, inference and training code, anatomy anchor bank) | [datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004) | heliosli | serving code of the compact members; the public compact trainer that K1 started from; attached to both final kernels |
| Public 0.958 notebook "RSNA Knee \| CoAtNet + R6 + 2.5D MIL \| LB 0.958" | [code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) | goodpjw2008 | the per-target weights w_t of P, the R224 share inside the Raptor slot, flip TTA |

### Earlier members, recipes and notebooks we read

| Work | Kaggle page | Author | How we used it |
|---|---|---|---|
| Compact bundle R7 `rsna-knee-compact-r7cnxfsss` (`8c7b666e`) | [datasets/heliosli/rsna-knee-compact-r7cnxfsss](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) | heliosli | compact member Oct 13-16 (C85-R7) |
| Compact bundle S6 `rsna-knee-compact-v1` | [datasets/heliosli/rsna-knee-compact-v1](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-v1) | heliosli | compact member Oct 11-12 (A5q, C85) |
| Notebook "≈0.957 \| Latest dataset \| score in one click!" | [code/heliosli/0-957-latest-dataset-score-in-one-click](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) | heliosli | S-branch serving sources, vendored byte-for-byte |
| Notebook with the 0.9573 public reading (now titled "This one is old version 0.9573 …") | [code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use) | heliosli | the 0.9573 public reading behind the R224 addition (H-B-164) |
| Notebook "This is best public dataset new version =0.9586" | [code/heliosli/this-is-best-public-dataset-new-version-0-9586](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586) | heliosli | read during the Oct 22 scouting; same recipe as our P |
| 2.5D ConvNeXt reader | [datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader) | goodpjw2008 | reader overlay tested (H-B-190) |
| Notebook "RSNA_Versia_7" | [code/evgendvorkin/rsna-versia-7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) | evgendvorkin | read as text only (same recipe as the 0.958 notebook) |
| Notebook "RSNA Knee 0949 Anatomical Mirror" | [code/nartaa/rsna-knee-0949-anatomical-mirror](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) | nartaa | reference for the 0.949 single model; mirror TTA tested (D-B-217) |
| Notebook "RSNA Knee 0945 Efficient 224crop" | [code/nartaa/rsna-knee-0945-efficient-224crop](https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop) | nartaa | read as text (the 224 px model) |
| Notebook "Bend the Knee to Speedy Raptors-The Original" (0.943 stack, "Mattia v34" recipe) | [code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original) | mattiaangeli | 0.943 reference recipe |
| DINOv3-S fold weights `knee-mri-fold-weights` | [datasets/mattiaangeli/knee-mri-fold-weights](https://www.kaggle.com/datasets/mattiaangeli/knee-mri-fold-weights) | mattiaangeli | A5 member tested on Oct 21-22; also an input of the public 0.941 / 0.943 stacks |
| Notebook "RSNA Knee Fast 2xT4 Inference" | [code/jiweiliu/rsna-knee-fast-2xt4-inference](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference) | jiweiliu | our exact copy of the 0.943 stack (Oct 2) |
| Public 0.941 stack, notebook `rsna-knee-0941-restructured` (preset `probe22`) | [kaggle.com/maverickss26](https://www.kaggle.com/maverickss26) (exact page not verified: the slug is no longer listed) | maverickss26 | our starting point (Sep 11-12) |
| Notebook "Bend the Knee to the Dinosaurs" (lineage of the 0.941 stack) | [code/mattiaangeli/bend-the-knee-to-the-dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs) | mattiaangeli | read; the 0.941 stack is a restructured version of it |
| Notebook "Knee MRI: training the twelve-finding model" (Raptor training recipe) | [code/dreaddevelopment/knee-mri-training-the-twelve-finding-model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) | dreaddevelopment | recipe of our own CoAtNet models |

### Label sets

| Work | Kaggle page | Author | How we used it |
|---|---|---|---|
| `rsna-knee-labels` (LLM report-distilled soft labels) | [datasets/dreaddevelopment/rsna-knee-labels](https://www.kaggle.com/datasets/dreaddevelopment/rsna-knee-labels) | dreaddevelopment | the Dread labels and ruler; 75 % of the L2 training labels |
| `jev-knee-labels` | [datasets/riadmohamed42/jev-knee-labels](https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels) | riadmohamed42 | the JEV hybrid ruler |
| `rsna-knee-hybrid-report-labels` | [datasets/flight0234/rsna-knee-hybrid-report-labels](https://www.kaggle.com/datasets/flight0234/rsna-knee-hybrid-report-labels) | flight0234 | flight_v4 labels; consensus member; prior label given to the Aquila reader |
| `rsna-knee-llm-report-labels` | [datasets/stevenleehans/rsna-knee-llm-report-labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) | stevenleehans | steven_v4 labels; consensus member |
| `rsna-knee-llm-labels`, `rsna-knee-report-labels` | [datasets/pilkwang/rsna-knee-llm-labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels), [datasets/pilkwang/rsna-knee-report-labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-report-labels) | pilkwang | pilkwang labels; consensus member |
| `rsna-knee-11-label-evidence-fusion` v2 (CC BY-NC 4.0) | [datasets/omerzalman/rsna-knee-11-label-evidence-fusion](https://www.kaggle.com/datasets/omerzalman/rsna-knee-11-label-evidence-fusion) | omerzalman | an extra evaluation lens (D-B-259); its scripts were not executed |

Other public label sets, used only in the label-set audit on Gold58 (D-B-014):

- barun2104: [datasets/barun2104/rsna-knee-stratified-folds-and-llm-soft-labels](https://www.kaggle.com/datasets/barun2104/rsna-knee-stratified-folds-and-llm-soft-labels)
- freshtime: [datasets/freshtime/rsna-knee-pseudo-labels-betta](https://www.kaggle.com/datasets/freshtime/rsna-knee-pseudo-labels-betta)
- jezrealmomoh: [datasets/jezrealmomoh/rsna-knee-pseudo-labels](https://www.kaggle.com/datasets/jezrealmomoh/rsna-knee-pseudo-labels)
- juanicarlos: [datasets/juanicarlos/rsna-knee-llm-labels](https://www.kaggle.com/datasets/juanicarlos/rsna-knee-llm-labels)
- laymond: [datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels](https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels)
- lixin73: [datasets/lixin73/rsna-knee-llm-report-labels-sol56](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56)
- mohammadsaidulislam: [datasets/mohammadsaidulislam/rsna-knee-super-labels-v1](https://www.kaggle.com/datasets/mohammadsaidulislam/rsna-knee-super-labels-v1)
- tasmeemreza: [datasets/tasmeemreza/rsna-knee-refined-llm-labels](https://www.kaggle.com/datasets/tasmeemreza/rsna-knee-refined-llm-labels)
- tsuyu122: `rsna-knee-mri-balanced-labels-public`, [kaggle.com/tsuyu122](https://www.kaggle.com/tsuyu122) (exact page not verified: no longer listed); never used for training (provenance unresolved, D-B-014)
- yunusgmsoy: [datasets/yunusgmsoy/rsna-knee-llm-report-labels](https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-report-labels), [datasets/yunusgmsoy/rsna-knee-llm-labels-2-source-merged](https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-labels-2-source-merged), [datasets/yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels](https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels), [datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged](https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged)
- zaidaliiq1000: [datasets/zaidaliiq1000/rsna-knee-llm-labels](https://www.kaggle.com/datasets/zaidaliiq1000/rsna-knee-llm-labels)

### Inputs of the public 0.941 / 0.943 stacks (Sep 11 - Oct 7)

Our copies of the public stacks attached these public inputs unchanged (see `kaggle/packages/KNEE_ANCHOR_0943/` and `kaggle/packages/EXP001_*`):

- dreaddevelopment Raptor weights: [raptor-knee-maxspan](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan), [raptor-knee-native384](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384), [raptor-knee-native384dense](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense); [raptor-knee-finespacing](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-finespacing) in one Gold58 probe
- mattiaangeli: [rsna-knee-coat-resgated-ep10-top3](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3), [rsna-knee-coatnet-d4-depthzone-swa3-b2](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2), [rsna-knee-coatnet-global96-top3](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coatnet-global96-top3), [opencv-python-headless-4120088-x86](https://www.kaggle.com/datasets/mattiaangeli/opencv-python-headless-4120088-x86), [knee-mri-fold-weights](https://www.kaggle.com/datasets/mattiaangeli/knee-mri-fold-weights)
- antoinegg1 RadImageNet heads: [rsna-knee-e9-radimagenet-heads-v15](https://www.kaggle.com/datasets/antoinegg1/rsna-knee-e9-radimagenet-heads-v15), [rsna-knee-e11-diverse-heads-v20](https://www.kaggle.com/datasets/antoinegg1/rsna-knee-e11-diverse-heads-v20)
- prvsiyan: [rsna-knee-v52-radimagenet-heads-20260812](https://www.kaggle.com/datasets/prvsiyan/rsna-knee-v52-radimagenet-heads-20260812)
- marwanmath: [resnet-50-radimagenet-marwan](https://www.kaggle.com/datasets/marwanmath/resnet-50-radimagenet-marwan)
- pilkwang: [rsna-knee-weights](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights), [rsna-knee-llm-labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels)
- sofiaanjenje notebooks (kernel sources): [code/sofiaanjenje/rsna-knee-e11-train](https://www.kaggle.com/code/sofiaanjenje/rsna-knee-e11-train), [code/sofiaanjenje/rsna-knee-e13-train](https://www.kaggle.com/code/sofiaanjenje/rsna-knee-e13-train)
- Meta DINOv2 (PyTorch, small): [models/metaresearch/dinov2](https://www.kaggle.com/models/metaresearch/dinov2)

### Pretrained weights used by our own experiments

- **RadImageNet ResNet-50** ([RadImageNet](https://www.radimagenet.com/)): initial weights of our ResNet-50 partner models (part of O), taken from marwanmath's Kaggle mirror [datasets/marwanmath/resnet-50-radimagenet-marwan](https://www.kaggle.com/datasets/marwanmath/resnet-50-radimagenet-marwan), whose Kaggle licence field is recorded in our notes as CC BY-NC-SA 4.0.
- **OrthoDiffusion knee MRI foundation weights**: [datasets/prvsiyan/orthodiffusion-official-knee-mri-foundation](https://www.kaggle.com/datasets/prvsiyan/orthodiffusion-official-knee-mri-foundation) (prvsiyan), fine-tuning scouts EXP039A-C only; not in any final submission.

## Other data, models and software

- **RSNA Knee Abnormality Detection competition data** (images, reports, labels): available to participants under the competition rules at <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection>. Not included here.
- **Osteoarthritis Initiative (OAI)**, <https://nda.nih.gov/oai>: used through nartaa's OAI-fine-tuned checkpoints and in our own OAI experiments and the OAI MOAKS evaluation panel, under the OAI data use agreement. No OAI data is included here.
- Open-source software including [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), [pydicom](https://pydicom.github.io/), [GDCM](https://github.com/malaterre/GDCM) (python-gdcm), [NumPy](https://numpy.org/), [FastAPI](https://fastapi.tiangolo.com/) and [React](https://react.dev/).
- LLM services used to read training reports (permitted by the host, D-B-029): Codex ("astra" lanes) and Claude. No report text is redistributed.

## Images in `docs/figures/knee_mri_examples.png`

The example MRI figure is **not competition data**. It is an adaptation (resized and placed side by side, with added titles and credit lines; image content not cropped) of two openly licensed images from Wikimedia Commons:

1. **"Knee MRI PD TSE FS Sagittal.jpg"** by Ptrump16, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Source: <https://commons.wikimedia.org/wiki/File:Knee_MRI_PD_TSE_FS_Sagittal.jpg>
2. **"Proton density MRI of a grade 2 medial meniscal tear.jpg"** by Nicolas Lefevre, Jean Francois Naouri, Serge Herman, Antoine Gerometta, Shahnaz Klouche and Yoann Bohu, from *Radiology Research and Practice* (2016), doi:[10.1155/2016/8329296](https://doi.org/10.1155/2016/8329296), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: <https://commons.wikimedia.org/wiki/File:Proton_density_MRI_of_a_grade_2_medial_meniscal_tear.jpg>

Because it includes a CC BY-SA 4.0 work, the composite figure `docs/figures/knee_mri_examples.png` is itself licensed under **CC BY-SA 4.0** (share-alike), not under Apache 2.0. Changes made: resizing, side-by-side arrangement, titles and credit lines.

## Image in `docs/figures/banner.png`

The README banner uses image 1 above ("Knee MRI PD TSE FS Sagittal.jpg" by Ptrump16, CC BY-SA 4.0) as a faded, tinted background element on the right. Changes made: cropping to the banner height, resizing, tinting, a horizontal fade and overlaid text. The banner is therefore also licensed under **CC BY-SA 4.0**. It contains no competition data.

The diagrams `docs/figures/*.svg` are our own work (Apache 2.0) and contain no competition data.

## Not included

- Competition data of any kind: DICOM images, labels, report text, study or series identifiers, per-study predictions.
- Model weights (public or ours), private Kaggle dataset contents, Kaggle kernel outputs, run caches and logs.
- Credentials of any kind.
- Third-party notebook sources that we only read for review.
