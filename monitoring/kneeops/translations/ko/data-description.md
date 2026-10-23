## 데이터셋 설명

이 데이터셋에는 인대·반월상연골 손상, 세 구획의 골관절염, 관절삼출, 활막염, 베이커 낭종, 골좌상, 골절의 12가지 흔한 소견이 주석으로 표시된 무릎 MRI 검사가 포함됩니다. 각 study는 한 번의 촬영 세션에서 얻은 여러 MRI 시퀀스로 구성되며, DICOM series 형식으로 제공됩니다. 과제는 study마다 12가지 소견 각각이 존재할 확률을 예측하는 것입니다.

여러 국가의 다양한 영상 촬영 기관에서 수집되어 스캐너, 프로토콜, 인구 집단의 범위가 넓습니다. 학습 study 중 일부에만 소견별 라벨이 있습니다. 원래 영상의학 판독문도 제공하므로 나머지 study의 라벨을 판독문에서 도출하는 데 활용할 수 있습니다.

## 파일

**train.csv** — 학습 study당 한 행입니다.

- `StudyInstanceUID`: study의 고유 식별자이며 `train_series/` 아래의 폴더명과 일치합니다.
- `Report`: 자유 형식 영상의학 판독문입니다. 작성 기관에 따라 여러 언어 중 하나로 작성되어 있을 수 있습니다.

12개 이진 라벨:

- `ACL`: 전방십자인대 손상(0/1).
- `MCL`: 내측측부인대 손상(0/1).
- `Medial Meniscus`: 내측 반월상연골 파열(0/1).
- `Lateral Meniscus`: 외측 반월상연골 파열(0/1).
- `Medial OA`: 내측 경골대퇴 구획의 골관절염(0/1).
- `Lateral OA`: 외측 경골대퇴 구획의 골관절염(0/1).
- `PF OA`: 슬개대퇴 골관절염(0/1).
- `Effusion`: 관절삼출 / 관절액 과다(0/1).
- `Synovitis`: 관절 안쪽 막의 염증인 활막염(0/1).
- `Baker's`: 베이커 낭종(0/1).
- `Contusion`: 골좌상 / 뼈의 타박상(0/1).
- `Fracture`: 골절(0/1).

**train_series.csv** — 학습 series당 한 행입니다. series는 한 번의 MRI acquisition이며 하나의 study는 여러 series로 구성됩니다.

- `StudyInstanceUID`: 이 series가 속한 study입니다.
- `SeriesInstanceUID`: series의 고유 식별자이며 `train_series/<StudyInstanceUID>/` 아래의 폴더명과 일치합니다.
- `Fluid_Sensitive`: 액체 신호를 강조하는 시퀀스(T2, PD, STIR 등)이면 `1`, 아니면 `0`입니다.
- `Fat_Suppression`: 지방 억제를 적용한 시퀀스이면 `1`, 아니면 `0`입니다. 학습셋에서 보듯 `Fluid_Sensitive`와 `Fat_Suppression`은 흔히 상관관계가 있지만 모든 경우에 동일한 것은 아닙니다.
- `Anatomical_Plane`: 영상 평면으로 `Sagittal`(시상면), `Coronal`(관상면), `Axial`(축상면) 중 하나입니다.

**train_series/** — 학습 DICOM은 `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm` 구조로 배치됩니다. 각 `.dcm` 파일은 영상 슬라이스 한 장입니다. series는 일반적으로 20~45장(중앙값 30장)이며, 일부는 수백 장에 이르는 긴 꼬리 분포를 보입니다.

**test.csv** — 공개 테스트셋의 study ID 3개를 담은 예시 테스트 파일입니다. 채점 시 실제 테스트 데이터로 대체됩니다. 테스트셋에는 약 1,300개 study가 있습니다. **테스트 단계에는 `Report` 필드가 제공되지 않습니다.**

- `StudyInstanceUID`: 테스트 study의 고유 식별자입니다.

**test_series.csv** — 예시 테스트 study를 위한 파일이며 `train_series.csv`와 스키마가 같습니다. 채점 시 실제 테스트 series 설명으로 대체됩니다.

**test_series/** — `train_series/`와 같은 구조의 예시 테스트 DICOM입니다. 채점 시 실제 테스트 DICOM으로 대체됩니다.

**sample_submission.csv** — 모든 라벨 열이 `0.5`로 설정된 유효한 제출 파일입니다.

## 데이터셋 분포 안내

각 데이터셋에 모든 이상 소견이 포함되도록 노력했지만, 학습·공개 리더보드·최종 평가 데이터셋에서 각 이상 소견의 유병률이 동일하다고 보장하지 않습니다.

## DICOM 참고 사항

series와 study마다 강도, 방향, 해상도가 다릅니다. 전송 구문은 비압축 Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR Little Endian 등이 혼재합니다. 모든 DICOM은 허용 목록에 있는 메타데이터 태그 86개만 남기고 나머지를 제거했습니다.
