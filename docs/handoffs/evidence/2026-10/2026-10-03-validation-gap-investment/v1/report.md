# 2026-10-03 · 검증 공백과 다음 연구 투자

**추가 GPU 학습 전에 기존 DINO 특징으로 10~15분짜리 cross-fold 유사영상 후보 검색을 한 번 할 가치가 있다.** 원본 영상 전수검사나 새 모델이 필요하지 않다. 이 검사가 음성이어도 환자 독립성이 증명되는 것은 아니며, 모든 성능 연구를 중단할 전제조건으로 삼지 않는다.

## 이미 있는 근거와 없는 근거

- 기존 EXP080 metadata-preflight는 보고서 그룹·Series UID·선택 시리즈 집합의 fold 간 교집합 0을 기록했다. 이는 식별자/텍스트 그룹 검사다. 같은 환자의 다른 촬영, 다른 보고서, 재발급 UID 또는 유사 영상은 배제하지 못한다. 이번에 이 검사를 다시 실행하거나 점수를 재계산하지 않았다.
- `study_meta.csv` 4,410행은 scanner·field·software·coil·sex·laterality를 갖는다. 현재 공통 제외2건을 적용한4,343건에는 scanner fingerprint133개가 있다. fingerprint 정의는 manufacturer/model/software/coil이며 환자ID·영상 해시가 아니다. 생성 소스가 PatientID를 수집하지 않았으므로 **저장표에 환자ID가 없다는 사실을 원본 DICOM에도 없다는 주장으로 바꾸면 안 된다.** 원본의 안정적 환자 연결 필드 존재는 이번에 검사하지 않았다.
- 기존187시리즈 및 EXP081 129시리즈 헤더 영수증은 geometry/frame/파일CRC를 보존한다. 목적 표본의 geometry 감사이고 전체 환자 연결표가 아니다. FrameUID는 시리즈마다 다르므로 동일 환자 연결에 쓰는 근거가 없다. ZIP 인덱스CRC도 DICOM 전체 파일에 대한 값이며, UID/헤더가 바뀐 동일 픽셀을 배제하지 못한다.
- 실제 사용 가능한 외부 DINO-v2 small 특징은 `[4407,6,3,384] float16`, **60,922,496바이트**다. 대응 UID4,407개가 folds의 index 순서와 모두 일치한다. 공통 제외 후4,343건 모두 최소3개 슬롯의 유효 특징이 있다. 약12GB의 원래 compact 픽셀 캐시나 약32GB의 Raptor corpus를 새로 읽을 필요가 없다.
- 특징 생성 소스는 외부 `lvd142m` DINO를 고정한 추출이다. 다만 당시 manifest에 가중치·코드·환경 SHA가 없어 완전한 과거 재현 영수증은 아니다. 이번에는 현재 특징/UID/mask/source를 해시로 고정했다. 이 자산은 현 Raptor의5slot/336/first-series와 다른 과거6slot/9slice/224/longest-series 입력이다. 따라서 검색 후보용으로 사용하고 현 입력의 완전한 중복 감사로 해석하지 않는다.

## 제안하는 검사 하나

[candidate-protocol.json](candidate-protocol.json)의 **기존 외부 DINO 특징 기반 cross-fold 후보 검색**이다. 아직 구현·실행하지 않았다.

1. label/prediction/LB/Gold/Report/픽셀을 읽지 않고4,343 UID·fold·mask·특징 해시를 고정한다.
2. valid mask와 유효 float16 특징 바이트가 같은 그룹을 찾는다. 정확한 표현 중복도 동일 환자·원본 픽셀의 확정 증거는 아니다.
3. 슬롯별 유효3개 triplet 평균을 정규화한 고정2304차원 표현으로 다른fold의 유사도를 계산한다. 같은 유효slot 패턴 안에서만 비교하고, reciprocal nearest neighbor 중 상위20 UID쌍을 미리 정한 동률 규칙으로 보존한다. 결과 기반 임계값·성적 선택은 하지 않는다.
4. 유사도 높은 정상 해부학·scanner 공통성의 거짓 양성과, crop/좌우반전/결측series/추적검사의 거짓 음성을 명시한다. 후보가 있으면 **별도 범위의 가림 소수 영상/헤더 확인** 후에만 실제 중복 가능성을 판단한다. 현재 초안은 추가 픽셀 읽기를 허가하지 않는다.

CPU 최대6thread·1GiB·900초 상한. 연산1~5분, 준비 포함10~15분은 **미실측 예산**이다. 확인된 중복이 있으면 이후 실험용 새 group split을 설계하되, 과거 모델을 새 split 이름으로 재채점해 독립 OOF라고 부르지 않는다. teacher/backbone 노출과 public-LB 반복 선택은 별개의 문제로 남는다.

## 다음 성능 가설 비교

| 축·우선순위 | 최소 개입 | 비용 예산 | 얻는 정보·남는 한계 |
|---|---|---|---|
| **학습 — 첫 GPU 후보** | RN34의 soft-label BCE에서 `pos_weight`만 전부1로 변경. B 입력·라벨·초기값·20epoch·seed·F0/F1 고정. EXP079 두 기준 모델은 모든 공통 계약이 맞을 때 재사용. | 구현/CPU검사30~60분 + 새2fit **25~35 GPU분**. 과거 EXP079 두fit 총25분이 근거이며 제안 시간은 추정. | 현재 `clip((1−prev)/prev,1,10)`이 공유 표현의12소견 순위 학습을 왜곡하는지 분리. 희귀소견 악화도 가능. 단순 확률 보정으로 AUC가 올라간다는 가설이 아니며, private 개선은 확정할 수 없다. |
| **표현 — 다음 설계** | F2 제외 EXP080 B backbone 고정. 같은 K24 특징에서 slot ID embedding head와 원래 head를 **둘 다 다시 학습**해 비교. F2 제외 train3475만 사용. | 구현/CPU검사60~120분 + 특징 추출10~30 GPU분 + head2개5~15 GPU분의 잠정예산. | 현재 pooling이 창의 plane/series 역할을 명시적으로 받지 못하는지 검사. 입력 교체 실패와 다른 가설. 기존 저장 head만 대조로 쓰면 재학습 효과가 섞이므로 두head 대조 필요. |
| **감독 — 근거부터** | train의 미언급/중증도·급성 여부 미해결 MCL 항목을 감사. 실제 근거가 있으면 해당 항목 loss만 제외하고 다른11소견·기존class weight는 고정. | 정의/가림감사60~120분 + CPU집계1~5분. 그 뒤 RN2fit25~35 GPU분 또는 CoAt2fit85~100분. | MCL120의 음성55건 재라벨과 다른 개입. unknown을 영상 음성으로 바꾸지 않는다. 기존weak AUC만으로 임상 정의가 좋아졌다고 판정할 수 없어 지금 바로 fit할 근거는 부족. |

현재 학습 소스는 Gold 진단 경로를 포함하므로, 새 실험에서 Gold 미접근을 요구하면 기존 동결 코드를 보존한 별도 최소 실행 버전이 필요하다. 어떤 후보도 자동 실행·새fold/seed 확대를 뜻하지 않는다. EXP081에서 현재 가중치의 직접 입력 교체를 채택하지 않은 것을 넓은 coverage 가설 전체의 반증으로 확대하지 않는다.

## 관측·출처

[inventory.json](inventory.json)은 실제 파일 SHA/크기/UID 정렬을, [research-api.json](research-api.json)은 localhost API 조회 시각과 수집 자료 revision을 보존한다. 새 Kaggle 수집·학습·추론·다운로드·점수 계산·공통 문서 변경은 없었다.

Kaggle 저장 토론736457은 FrameUID 해석에 관한 참가자 질문이며 저장본에 운영자 답변이 없다. 토론735639의 양쪽 무릎 FOV 사례는 동일 환자 cross-study 중복 또는 반대쪽 정상 라벨의 증거가 아니다. 자료를 환자 독립성 보증으로 쓰지 않았다.
