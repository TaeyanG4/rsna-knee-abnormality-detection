# RSNA Knee | DINOsaur V4 🦖 

## 🙏 Additional attribution — Kunal Desale

This DINOsaur V4 experiment was informed by the public Kaggle notebook:

**Kunal Desale — `kunaldesale2408/rsna-knee-abnormality-detectionv1`**  
https://www.kaggle.com/code/kunaldesale2408/rsna-knee-abnormality-detectionv1

Thank you to **Kunal Desale** for publishing the notebook and exposing the interaction between the public DINO/DINOv3/RadImageNet/Raptor stack and an additional CoAtNet residual branch.

### What this fork does — and does not — take from that notebook

- We **do not copy or require** the private CoAtNet `resgated e4/e6/e8` checkpoint bundle.
- We **do not copy** the private-residual runtime.
- We independently implement a **public-only analogue** using predictions from the public Raptor checkpoints that DINOsaur V4 already executes.
- The Renta K. weak-label Medial-Meniscus residual remains credited to `renta0426`.
- The Bend-the-Knee / DINOv3 lineage remains credited to Mattia Angeli and its upstream contributors.
- All original checkpoints, datasets, and code retain their original authorship and licenses.

This is an auditable derived experiment, not a relabeling of someone else's work. Please preserve these credits if you fork it again.


## 🧪 DINOsaur V4 — PublicDual branch

The cells above still reproduce and preserve the known **0.937** Renta-based control.

This branch now rewinds only to the post-DINOv3 / pre-Rad transformer state and rebuilds the downstream path with a more diverse **public-only Rad-dual5** signal:

1. two distinct five-fold public v15/E10 RadImageNet head families;
2. E13 FS-crop specialist at 0.50 inside the Rad block;
3. a second E13 pass on the E11 slot layout at 0.15;
4. the same public four-view Raptor;
5. the proven Renta Medial-Meniscus residual `T30 / R60 / bag10`.

The private CoAtNet `resgated e4/e6/e8` residual from the 0.938 reference is deliberately excluded.


### Optional PublicDual + V6 probe

The previous V6 five-target overlay did **not** beat 0.937 on its own, so it is no longer the default.  
For completeness, this cell writes a separate probe on top of the new PublicDual parent, then restores `submission.csv` to the safer PublicDual candidate.


## 🧪 V4 PublicSynovitis residual

The uploaded 0.938 reference contains a multilingual report-label parser with a specific **Synovitis backoff**: when Synovitis is not explicitly addressed, Effusion and synovial-proxy evidence are used as a weak auxiliary signal.

The exact report parser is **not copied** into this fork. More importantly, the displayed scored inference path in the reference still obtains its major final boost from a private CoAtNet residual, so reproducing the full 0.938 score without that artifact would be misleading.

Instead, this V4 branch transfers only the high-level idea in a test-time-safe way: it uses **MRI-model agreement on Synovitis plus MRI-model agreement on Effusion** as a tiny, decorrelated residual for the Synovitis column only.

The known Renta Medial-Meniscus route and the other 11 targets remain untouched. Three fixed variants are written so the experiment is auditable:

- `safe`: 4% auxiliary residual;
- `main`: 8% auxiliary residual;
- `probe`: 14% auxiliary residual.

`submission.csv` is the conservative `main` candidate.


## 🧪 DINOsaur V4 — Public CoAt checkpoint-diversity residual

The uploaded reference uses an additional **private CoAtNet residual** inside the Raptor family. Its internal blend is around `0.40`, but reproducing that directly would require private checkpoints, so this fork does not do that.

Instead, this branch uses only public Raptor predictions that DINOsaur V4 already computes:

- `maxspan-v5` remains the dominant public Raptor anchor;
- `native384dense-v10` is a different public checkpoint / dense view;
- `native384-v8` is another public checkpoint with a different slice budget and span;
- `reverse-v5` is retained only as a separate view-diversity probe because it shares the v5 checkpoint.

The **main residual uses only v10 + v8**, making it checkpoint-diverse rather than merely another view of the dominant v5 arm. No additional model pass is required.

Outputs:
- `safe`: internal residual alpha `0.08`;
- `main`: alpha `0.16`;
- `probe`: alpha `0.24`;
- `viewprobe`: alpha `0.16` with reverse-v5 included.

`submission.csv` is the `main` candidate. The outer Transformer/Raptor blend remains `0.40 / 0.60`, and the proven Renta Medial-Meniscus formula is re-applied with the modified Raptor rank.
