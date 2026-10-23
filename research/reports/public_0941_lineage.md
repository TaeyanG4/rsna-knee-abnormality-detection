# Public 0.941 lineage audit

Updated: 2026-09-12

## Current 0.941 public frontier candidates

- `mattiaangeli/bend-the-knee-to-the-dinosaurs`
  - Public notebook currently described/displayed as 0.941.
  - Full DINO + A5 + RadImageNet + 4-arm Raptor + residual-gated CoAt lineage.
  - Residual-CoAt blend: public Raptor rank 0.60 + residual-CoAt rank 0.40.
  - In the frozen source audited here, the inner hybrid is not re-ranked before the outer blend.
  - Outer CoAt/Raptor routing is target-specific in the latest lineage.
- `maverickss26/rsna-knee-0941-restructured`
  - Explicit default `probe22` recipe documented as 0.941 public.
  - Outer weights: default 0.60; ACL 0.75; Medial Meniscus 0.80; Lateral Meniscus 1.00; Lateral OA 0.75; Fracture 0.75.
  - Inner public-Raptor/residual-CoAt hybrid is re-ranked before outer blending.
  - Source SHA256: BB34D22E97D29F6CB0638BCF5EFFD1DE9237778E420BABA3D9B4258485855B91.
- `evgendvorkin/rsna-baseline`
  - Describes the same 0.941 target-routing recipe and credits the Mattia lineage.
  - Inner public-Raptor/residual-CoAt hybrid is re-ranked before outer blending.
  - Adds DINOv2-Base as an available model source in metadata in addition to DINOv2-Small.
  - Source SHA256: 5AD37157B36007C33E6314462FA3D6C47A2B796F420188AE514BFDFA16BE9ECA.

## Shared lineage

All three rely on largely the same public assets: Raptor v5/v8/v10, DINO family, RadImageNet encoder/heads, residual-gated CoAt e04/e06/e08, and the same competition input. These should not be counted as independent ensemble diversity merely because they are separate notebooks.

## Meaningful differences

1. Target-specific outer routing is the main documented move from the 0.939 parent toward 0.941.
2. Maverick/Evgen re-rank the strengthened Raptor hybrid internally; the frozen Mattia source does not. This changes scale before the outer blend, even though standalone AUC of the inner hybrid is unchanged by monotonic re-ranking.
3. Lateral Meniscus receives 1.00 Raptor/CoAt routing in the 0.941 recipe, discarding the transformer/Rad branch for that target.
4. The current 0.941 plateau should be treated as a family of near-identical public solutions, not multiple independent SOTA models.

## Execution evidence

- `EXP001_PUBLIC_SOTA_REPRO`: byte-identical source to the frozen Mattia notebook (SHA256 B5A8AB7C5C60B0620BC9455AB9E6F0B7E57BACEFEC875E0D4E7DC1B3A3499CDA) executed successfully on Kaggle T4x2 for the visible 3-study test sample.
- `EXP001A_COAT_COMPONENT_REPRO`: Gold58 rank macro AUC 0.90932684 on Kaggle T4; public saved reference 0.90946964.
- `EXP006A_RAPTOR_BOUNDARY_STAGE_A`: fixing same-slot window boundaries changed maxspan-v5 Gold58 macro AUC 0.91982514 -> 0.92028506; paired bootstrap CI crosses zero.
- `EXP001_PUBLIC0937_REPRO`: failed because Kaggle allocated a Tesla P100 (sm_60) while the current PyTorch image supports sm_70+; this is an accelerator compatibility failure, not a model/data failure.

## Release constraints

The RadImageNet input is still marked CC-BY-NC-SA-4.0 in the downloaded Kaggle metadata. Research/control execution is tracked separately from final release eligibility. The winner-code/model redistribution compatibility question remains on hold pending authoritative confirmation.