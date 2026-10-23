<div align="center">

# Head and shoulders, knees and toes

### A proven public 0.936 knee-MRI checkpoint

<img src="https://playbees.firstcry.com/blog/wp-content/uploads/2025/12/Head-Shoulders-Knees-and-Toes-Lyrics-Nursery-Rhymes-For-Kids.jpg" width="780" alt="Head, Shoulders, Knees and Toes nursery-rhyme illustration">

<sub>Cover image supplied by the notebook owner. Source: <a href="https://www.bing.com/images/search?view=detailV2&ccid=R3yueXDs&id=615F980032BC4C1280484744A510D964D1306DA6">Bing Images / FirstCry PlayBees</a>.</sub>

</div>

---

| Public LB | Findings | Inference families | Network | Proven source |
|:--:|:--:|:--:|:--:|:--:|
| **0.936** | **12** | **DINO + RadImageNet + dual CoAtNet** | **offline** | **DINOsaur V10** |

> This release is the reproducible control checkpoint for new competitors. It
> is a source-faithful Apache-2.0 port of [Roman Tamrazov's scored DINOsaur V10](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4?scriptVersionId=344688577).
> Model, preprocessing, and fusion code is unchanged. Three owner-specific
> Kaggle mount strings are translated to the standard public-dataset mounts;
> only those paths, this explanatory markdown, and the H&S identity differ.

## What the checkpoint does

```text
RSNA DICOM studies
       |
       +-- DINOv2/DINOv3 protocol models -----------+
       |                                             |
       +-- RadImageNet diversity + calibration ------+--> cohort-rank fusion
       |                                             |
       +-- MaxSpan v5 + WideDense v4 CoAtNets -------+
                         target-specific complement          |
                                                            v
                                                     submission.csv
```

The V10 change is not a generic model soup. It uses MaxSpan v5 as the primary
Raptor checkpoint, WideDense v4 as a complementary checkpoint, and restricts
the complement to predeclared findings and weights before fusing with the
transformer/RadImageNet anchor.

## Reproducibility contract

- Proven public score: **0.936**.
- Proven source version: **`344688577`**.
- Upstream notebook source SHA-256: **`4504962c3b803a4385c2188084c13c3fe806ad006fd49b2201a03a78c10f460c`**.
- The four prediction cells match V10 except for three audited mount-path
  substitutions across the first two cells.
- Required inputs are kept exactly as declared by the scored source.
- Expected output: `/kaggle/working/submission.csv` with one ID column and the
  twelve official findings.
- This is a competition checkpoint, **not a clinical device**.

## Credit by retained responsibility

| Retained component | Public source |
|---|---|
| Scored V10 inference graph and dual-checkpoint target fusion | [Roman Tamrazov — DINOsaur V10](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4?scriptVersionId=344688577) |
| DINO/RadImageNet reproduction assets | [Tony Li — RSNA Knee reproduction assets](https://www.kaggle.com/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets) |
| MaxSpan and WideDense Raptor checkpoints | [Dread Development — Raptor datasets](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan) |
| Standard DINOv2 ViT-S/14 base | [Meta Research — DINOv2](https://github.com/facebookresearch/dinov2) |

The notebook is packaged and audited here by **prvsiyan** so the exact public
checkpoint can be run, inspected, and extended from one place.