# 토론 수집 점검과 유효성 판단 (2026-10-02)

수집 상태: 라이브 143개 = DB 143개, 원글·댓글 트리 전부 보유(SDK), 잘린 댓글 0, 빈 원글 0, HTML 잔여 0. 제목에 "teammates제외뒤로 미룸"으로 바꿨다. 댓글이 1개 늘어난 글(740610)은 다음 수집에서 갱신된다. 아래 판단은 원문을 읽고 내린 것이며 참가자 주장은 재현하지 않았다.

## 유효 — 계획(ADR 0003)에 직접 반영

| 토론 | 핵심 사실 | 반영 단계 |
|---|---|---|
| 735304 Best single-model score (68v·112c) | 라벨이 결정적 lever. OOF 예측을 라벨에 섞어 단일 fold CoAtNet 224 → 0.950, 다중 teacher pseudo-label → 0.949, 5-fold ResNet 224 → 0.954(주장). 288이 해상도 sweet spot, FOV 140 mm. Gold58에서 calibration 확인 | R1·R2 |
| 737696 RSNA Raptor Weights (20v) | 공개 체크포인트는 모두 같은 라벨. corpus 기하(span 확대, 80~96장, 384 px)와 창 수(42→62, +0.003)만으로 단일 0.924→0.936~0.938. SWA 효과 없음. 단일 fold 전체 학습 0.941이 5-fold 0.939보다 빠르고 높음 | R3·R4 |
| 733932 "Not addressed언급 없음". 침묵의 의미는 소견마다 다름. LLM 0.878 > regex 0.814 | R2 |
| 741310 Six data traps | InstanceNumber 순서가 1/3 시리즈에서 틀림(우리는 위치 정렬), laterality 절반 결측(flip 금지), non-FS는 여러 대비 혼합, 시리즈 결측 mask 필수, 2.5D 유리. Synovitis 양성 절반이 보고서 침묵, Fracture 7% vs 31% | corpus 검증·R2 |
| 737566 Synovitis stuck 0.6~0.7 | 보고서 16%만 synovitis 언급, teacher AUC 0.79. RadImageNet 사전학습이 0.78로 올림 → 의료 사전학습 멤버가 다양성에 유리 | R2·R5 |
| 734105 SOFT wins 3/3 seeds | soft 라벨이 hard보다 Gold58 +0.014(CI는 0 포함). soft 유지 | R2 |
| 735154 Scaling encoder bought nothing | DINOv2 S→B +0.0011, 노이즈 바닥 0.002. 용량은 병목 아님. 체크포인트의 전처리 계약을 바꾸면 조용히 손해 | R1 (작은 모델로 ablation) |
| 740610 Plateau 0.903 + Tucker Arrants 답 | 최고 단일 공개 모델을 로컬 재현 → 공개 LLM 라벨 → 224 단순 baseline으로 0.94. 우리 R0과 같은 순서 | R0 |
| 742050 200 forks at 0.941 + Mattia 댓글 | per-label map은 public split 튜닝. flat 0.60을 두 번째 최종 선택으로. 분 단위 가중치 튜닝 = public 과적합 | R5·최종 선택 |
| 742327 Shake-up | 메달선 안팎 24%/24%로 public 순위 의미 적음. 0.935+ 자체 모델이 public blend보다 private에 유리(참가자 의견) | 검증 설계 |
| 736635 LB OOF relationship, 744961 | OOF가 LB를 과소평가, 비선형. 누출 없는 CV도 LB가 더 높음 → 작은 LB 차이로 판단 금지 | 검증 설계 |
| 733517 metadata shortcut, 734004 scanner CV | 메타데이터만으로 0.652(random) vs 0.598(scanner-grouped) → 0.05는 site 암기. Siemens/Philips/GE × 1.5T/3T 13개 그룹. PatientSex가 test 헤더에 있고 소견 prior가 성별로 크게 다름 | 검증 설계(스캐너 그룹 진단) |
| 734681 public/private split by site? | 주최측 무응답. 전체 site holdout 가능성 → 스캐너 그룹 OOF를 함께 본다 | 검증 설계 |
| 744230 bf16 PSA | T4/P100은 bf16 에뮬레이션으로 50배 느려짐·시간 가드로 0.5 채움. fp16 사용, `is_bf16_supported(including_emulation=False)` | R6 |
| 743374 Fast submissions | 스레드 DICOM 읽기 372 MB/s, 헤더 직접 파싱, fp16 가중치+half 입력이 autocast보다 2배, uint8로 GPU 전송 | R6 |
| 740375 One fold vs five fold | 전체 학습 단일 모델 ≈ 5-fold 앙상블(LB), 시간 1/10 | R4 |
| 738708·737163·740094 corrupted DICOM·metric edge | 학습 2 study 손상(우리 corpus도 1 study 0 volume), test에도 있을 수 있음 → study별 decode 실패 시 prior 출력. 단일 class 열 처리 미확인 | R6 |
| 735639 Two knees | 7 study에 양쪽 무릎 → 중앙 crop이 무릎을 놓칠 수 있음 | corpus 점검(후순위) |
| 742926 (−4v) Dread 댓글 | 서로 불일치하는 모델이 앙상블에 유리 | R5 |
| 733965 Use of hosted LLMs (주최) | 보고서를 외부 LLM API에 보내 라벨 추출 허용(합리적 비용) | R2 라벨 재추출 옵션 |
| 733343 Overview (주최, 84v) | Gold 라벨은 보고서와 독립된 판독 과정. 임상 배경 | 라벨 해석 |
| 743416·733652·743420·744056·735121·737950 외부 데이터·공개 가중치·라이선스 | 주최 미응답. 공개 가중치 사용 수상 자격·NC 라이선스 불확실 → 최종 제출 전 재확인, 자체 학습 모델 비중 확보 | 위험 관리 |

## 참고만 — 사실은 맞지만 행동 변화 없음

733826 보고서·라벨 불일치 사례(28v), 733491, 737454 공개 라벨표 벤치마크(우리 감사와 중복), 734095 LLM 없는 라벨링, 735826 병목 설문(라벨이 결정적이라는 의견 다수), 740162 helped/didn't(DINOv3<DINOv2, 보고서 보조 loss 해로움), 738495 GRU head(과거 RSNA CT에서 유효, 여기선 미검증), 740164 backbone 역사, 736678 Efficiency LB, 735767 ceiling 예측(금 0.96, 1위 0.975 추정), 733934 ResNet34 baseline, 738096 DINOv2 B vs S, 743148 image→report 생성, 744519 private 모달리티(무응답), 737155 Synovitis 기준(무응답), 733836·734055·734106·735855 데이터 구조 설명.

## 무효 — 노이즈·운영·모집

팀 모집 6개(736498·741459·744511·744495·735599·736497), 742053 meme, 하드웨어·다운로드·실행 오류·자격·결제 질문(743303·743698·734676·742474·741916·742077·739693·733797·733971·737350·735017·735295·734131·737333·734062·743746·738129·739200·743779·742122), 743916 autonomous researchers, 737449 접근성 불만, 735348·737635 포크 윤리, 737820 계정 정지, 736170 개인 MRI 공유, 733497 개인 사례(42v지만 기술 내용 없음), 736301 고릴라(태도 환기), 734357 YOLO, 736999 ROI crop(답 없음), 736348·740624·741469·742267·742770(음수 추천, 내용 없음), 736711·735596 용어·언어 질문, 736098 Drive 전송, 738036 Pillar-0(답 없음), 739228, 742190, 744055.
