# Recomputed evidence tables

Teacher AUC, public prediction rescore, and actual MRI inference are not interchangeable.

## Per target

| target | positive_count | Flight_teacher_AUC | Public_DINO_gold_AUC | Public_CoAt_probability_AUC | Public_CoAt_rank_AUC |
| --- | --- | --- | --- | --- | --- |
| ACL | 24.00000 | 0.98713 | 0.86887 | 0.95711 | 0.95711 |
| MCL | 9.00000 | 0.96825 | 0.88435 | 0.89342 | 0.89116 |
| Medial Meniscus | 26.00000 | 0.94832 | 0.84976 | 0.96034 | 0.96635 |
| Lateral Meniscus | 23.00000 | 0.87888 | 0.64224 | 0.89814 | 0.89379 |
| Medial OA | 15.00000 | 0.93178 | 0.95039 | 0.97829 | 0.97984 |
| Lateral OA | 11.00000 | 0.83269 | 0.70793 | 0.84526 | 0.84913 |
| PF OA | 21.00000 | 0.90154 | 0.80051 | 0.84299 | 0.84170 |
| Effusion | 35.00000 | 0.87702 | 0.94286 | 0.95031 | 0.93975 |
| Synovitis | 27.00000 | 0.79032 | 0.74194 | 0.79809 | 0.80765 |
| Baker's | 12.00000 | 0.94384 | 0.95471 | 0.98732 | 0.98370 |
| Contusion | 19.00000 | 0.85965 | 0.84211 | 0.91633 | 0.90486 |
| Fracture | 18.00000 | 0.87014 | 0.89444 | 0.91111 | 0.89861 |

## Experiments

| experiment_id | metric_kind | status | gold_macro_auc | runtime_sec | keep |
| --- | --- | --- | --- | --- | --- |
| EXP001_PUBLIC_SOTA_REPRO | FULL_PUBLIC_PIPELINE | BLOCKED_LICENSE_VERSION_RUNTIME |  |  | control_candidate |
| EXP001A_COAT_COMPONENT_REPRO | TRAIN_IMAGE_COMPONENT_INFERENCE | RESULT_NOT_RETRIEVED |  |  | diagnostic |
| EXP002_PUBLIC_LABEL_AUDIT | REPORT_TEACHER_ONLY | COMPLETED | 0.89913 |  | teacher_control |
| EXP003_FAMILY4_CONSENSUS | REPORT_TEACHER_ONLY | COMPLETED_NO_GAIN | 0.88007 |  | no |
| EXP003B_UNKNOWN_HALF | REPORT_TEACHER_ONLY | COMPLETED_NO_GAIN | 0.88710 |  | no |
| EXP004_UNCERTAINTY_EXPORT | TARGET_PREPARATION_ONLY | EXPORTED_NOT_TRAINED |  |  | features |
| EXP005_TRAIN_GEOMETRY_AUDIT | DICOM_AUDIT_ONLY | KAGGLE_COMPLETED |  | 47.12400 | audit |
| EXP010_PUBLIC_RANK_VS_PROBABILITY | PUBLIC_PREDICTION_RESCORE | COMPLETED_CI_INCLUDES_ZERO | 0.91156 |  | diagnostic |

## Verified checkpoints

| model_id | version | sha256 | inference_geometry |
| --- | --- | --- | --- |
| W001 | coat_resgated_gold_e04.pt | cefaf007415b5e45ce2a720d5e211a183a4e90f160710ba1f72b1ce7c46239a2 | 384px/130mm/within-series/max76 |
| W002 | coat_resgated_gold_e06.pt | 56eca1f61651362ecfae8aecdd9327125713b95653344b8ad62ef8cff17d08f4 | 384px/130mm/within-series/max76 |
| W003 | coat_resgated_gold_e08.pt | 8d3159102046c127dc0d084fb69c9304ffcbd4b98d1333f9811d4d613fea1b12 | 384px/130mm/within-series/max76 |
| W004 | raptor_ft_coatnet_v5_full_swa.pt | 20d45e0732a0e9120025d0172a4d2dd0170900bc250b06742dfb7ae08a63786a | 140mm/64slices/62windows/span.02-.98 |
| W005 | raptor_ft_coatnet_v8_full_swa.pt | 3db77388662d516bb0ce790dafd24e36f4991f6089af26daf546817fe1c21862 | 384px/140mm/44slices/42windows/span.06-.94 |
| W006 | raptor_ft_coatnet_v10_full.pt | 5add8816a47e040efca89bb44480e1cbd01f92db1253831ff7ceb6c65ee02b1d | 140mm/64slices/62windows/span.02-.98 |
