# RSNA Knee 출발 노트북 재선정 — 2026-10-03

관측: 2026-10-03 11시대 KST. 앱의 노트북 242개 목록에서 점수 상위200개·최근40개를 조회하고, 선정 후보12개의 최신 소스를 Kaggle에서 다시 수집했다. Kaggle 스킬·기존 수집 adapter·앱 API를 사용했다. 계정에 새 노트북을 만들거나 실행·제출하지 않았다.

## 결정

**제출·개발 기반은 Jiweiliu Fast 2×T4 v11을 유지한다.** 우리 비공개 `taeyangg4/knee-anchor-0943-stack` v1의 제출56773226은 10:49 KST 수집에서 COMPLETE/0.943이었다. 로컬 보관본과 이번 최신 Jiweiliu 소스는 **코드 셀23개 모두 동일**하다. 더 높은 점수의 공개 후보도 조사했지만, 현재 확보 가능한 자산과 우리 계정의 재현 근거를 함께 보면 교체할 이유가 부족하다.

이 선택은 공개0.943가 우승 모델이라는 뜻이 아니다. **검증된 전체 스택은 고정하고, 보완 모델을 추가하거나 특정 구성 모델만 교체한다.** 학습 실험에는 기존 Raptor 도구와 작은 다른 계열을 쓸 수 있으며 전체 추론 노트북을 학습 baseline으로 취급하지 않는다.

## 후보 비교

점수는 별도 표시가 없으면 앱이 10:50 KST에 수집한 Kaggle의 notebook **best public score**다. 최신 source revision의 점수와 연결된다는 보장은 없다. 이번에 새로 실행해 재현한 점수는 없다.

| 후보·현재 버전 | 점수 근거 | 실제 차이와 판정 |
|---|---|---|
| [Jiweiliu Fast 2×T4](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference) v11 | 공개0.943 + **우리 제출0.943** | 선택. DINO20·A5 5fold·RadImageNet·Raptor4view·CoAt4reader. 입력 refs에 빈 항목0. 48 study 초과 시 CoAt 순차 실행, fingerprint·fallback 기록. 로컬 제출본과 코드 일치 |
| [pjmathematician D4 Blend](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-blend) v3 | 공개0.946 | 공개 스택 뒤에 `S_dist/S_coat/S_cnxl` 학생 모델을 추가하고 rank 비중0.45. `rsna-knee-eff6-assets`, `rsna-knee-ens14-assets` 필요. 빈 dataset refs3개, 필요한 추가 자산의 공개 검색 결과0. 현재 출발점으로 채택 보류 |
| [D4 Lite](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-lite) v1 | 공개0.945 | `S_dist`만 사용, 36창·4weights, rank 비중0.45. 동일 추가 자산 의존성 때문에 보류 |
| [S75 w50](https://www.kaggle.com/code/aastikrajan15/knee-s75-w50) / [w60](https://www.kaggle.com/code/aastikrajan15/knee-s75-w60) 각각v1 | 공개0.945 / 0.945 | 별도 ConvNeXt-Tiny 320×24 checkpoint10개를 기존 스택과0.5/0.6 혼합. 추가 `aastikrajan15/knee-ckpt`와 `model.py`가 필요하고 현재 접근 실패. 공개 소스의 base64는 텍스트로만 디코딩해 확인; 실행하지 않음 |
| [Skarin reproduction](https://www.kaggle.com/code/skarin/rsna-knee-reproducing-the-0-943-public-stack) v3 | 공개0.943 | 차선 비교본. 코드23개 중22개가 Jiweiliu와 완전히 같고 나머지는 `_RAD_EXCLUDE`에서 Fracture를 빼는 한 줄. 새로운 모델 계열이 아님 |
| [Mattia Speedy Raptors](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original) v39 | 공개0.943 | upstream 모델·학습 레시피 참고. 코드23개 중21개가 Jiweiliu와 동일. CoAt 실행/집계와 audit 차이를 확인했으며 전체 교체의 성능 이득은 입증되지 않음 |
| [Roman DINOsaur V5](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5) v26 | 공개best0.943; v26 점수 연결 미확인 | 이미 계산한 DINO 신호를 ACL/MCL에 재사용하고 CoAt probability/rank consensus 혼합. 신규 계열은 아니지만 추가 forward 없이 가능한 별도 ablation 후보 |
| [OrthoDiffusion Nasrin standalone](https://www.kaggle.com/code/sushanthtiruvaipati/rsna-knee-orthodiffusion-nasrin-standalone) v1 | 앱 점수 없음 | 다른 표현을 쓰는 보완 후보. 3 plane diffusion encoder + 4fold head. 자산v4 약1.66GB의 메타·파일목록 접근 가능. 성능·decode 누락·전처리 parity·자산 조건 확인 전 기반 교체 불가 |
| [Raptor/CoAt consensus](https://www.kaggle.com/code/sushanthtiruvaipati/rsna-knee-raptor-coat-blend-consensus) v4 / [MedMen/MedOA flip](https://www.kaggle.com/code/sushanthtiruvaipati/rsna-knee-medmen-medoa-raptor-flip) v2 | 앱 점수 없음 | 제외. 전자는 공개 테스트3 UID의 예측을 하드코딩, 후자는 `len(df)==3`을 강제하고 행별 값을 교환. 숨겨진 테스트 추론 기반이 아님 |
| [Ryokucha 0946+ours05](https://www.kaggle.com/code/ryokucha/rsna-knee-d4-blend-0946-ours05) v1 | 제목0.946, 앱 점수 없음 | 기존 학생 모델 단계를 건너뛰면서0.946 유지라고 주석을 적고 새 ConvNeXt5%를 추가. 빈 입력 refs2개. 제목·주석만으로0.946 재현 후보로 볼 수 없음 |

## 높은 점수 후보에서 확인한 한계

빈 입력 ref는 소스 식별이 불가능하다는 관측이며, 그 자체로 비공개인지 삭제됐는지를 확정하지 않는다. MCP 접근과 로컬 인증 Kaggle API를 모두 확인했다. S75의 명시적 `aastikrajan15/knee-ckpt`는 접근 실패했다. pj 자산 두 이름은 검색되지 않았고, 노트북 작성자 소유 경로로 조회해도 접근하지 못했다. **원 소유자까지 확인된 공개 가중치 패키지는 현재 확보하지 못했다**는 것이 정확한 결론이다. 자산이 공개되면 다시 평가한다.

D4 Blend/Lite는 누락 시 public-only로 돌아간다는 로그가 있지만, 뒤에서 `_EFF6 / 'arms'`, `_ENS14 / 'arms'`를 조건 없이 평가한다. 자산 탐색이 None이면 그 지점에서 TypeError가 날 수 있어, 그대로 복사하면 안전하게0.943로 끝난다고도 보장할 수 없다. 이는 정적 코드 관측이며 실제 실패 실행을 재현한 것은 아니다.

앱 runtime277초·358초 같은 수치는 노트북의 일반 실행값이다. 약1,300건 숨겨진 테스트의 full runtime으로 사용하지 않았다. 소개 글의 ‘전체 테스트30분’ 등도 이번 검증 사실에 포함하지 않았다. MCP의 `last_run_time`과 catalog 시간이 다르므로 source version·raw hash·score 관측시각을 따로 보존했다.

## 다음 실험 순서

1. **기준본 동결.** 현재 패키지·입력 버전/hash·각 단계 예측·활성 CoAt reader 수를 기록한다. 실행이 완료됐다는 사실과 전체 모델이 fallback 없이 참여했다는 사실을 구분한다. 코드/정밀도/전처리/혼합을 동시에 바꾸지 않는다.
2. **기존 자체 모델의 보완 가능성부터 확인.** 보유 ConvNeXt/ResNet/Raptor 예측의 UID·12target·라벨·split 계약을 맞추고 standalone 품질과 소견별 오류/상관을 비교한다. 이후 검증된 anchor에 후보 하나를 작은 고정 비중으로 더하는 안을 만든다. 자체 standalone0.943 초과는 전제가 아니다. 낮은 상관만으로 개선을 단정하지 않는다.
3. **저비용 변경 하나를 대조.** Skarin의 Fracture routing 또는 Roman의 기존 신호 재사용 중 하나만 별도 ablation으로 둔다. 기존0.943와 같은 표시점수인 만큼 이를 주된 우승 경로로 과대평가하지 않는다.
4. **새 학습은 깨끗한 평가와 함께.** R2 nested 검증을 보완하되 Raptor 전면교체를 미리 결정하지 않는다. 다른 encoder나 plane/소견 특화 모델이 기존 스택에 없는 정보를 제공하는지 시험한다. OrthoDiffusion은 현재 자산 접근이 가능한 후보지만 우선 작은 parity/품질 감사가 필요하다.
5. **제출 전 패키지·실행시간 검증.** 공개 모델이 학습한 train 예측과 새 모델 OOF를 섞은 점수를 독립 ensemble OOF라고 부르지 않는다. Gold58·public LB 비율 탐색도 피한다. 고정한 가설의 제한된 후보를 검토해 승인 후 제출하며, 채점 대기는 하지 않는다.

S75와 D4의 구조는 ‘검증된 공개 스택 + 별도 모델’ 접근의 참고 사례다. 공개best 점수와 현재 코드의 연결이 미확인인 만큼 +0.002/+0.003의 원인을 이번 분석이 실험적으로 증명한 것은 아니다. 1등을 목표로 한 일정·자원·평가 원칙은 [전체 계획](route-to-first-2026-10-03.md)을 따른다.

## 재현 근거

- [12개 소스 메타·revision·raw SHA256·관측시각](../handoffs/evidence/2026-10/2026-10-03-notebook-selection/sources.json), [점수순 목록](../handoffs/evidence/2026-10/2026-10-03-notebook-selection/score-catalog.json), [최근 목록](../handoffs/evidence/2026-10/2026-10-03-notebook-selection/recent-catalog.json).
- [코드 셀·안전한 디코딩 발췌](../handoffs/evidence/2026-10/2026-10-03-notebook-selection/code-analysis.json), [정확한 코드 비교](../handoffs/evidence/2026-10/2026-10-03-notebook-selection/code-comparisons.json). 원문은 기존 workspace runtime notebook 저장소에 버전별 보존했다.
- [인증 API 자산 확인](../handoffs/evidence/2026-10/2026-10-03-notebook-selection/asset-access.json), [MCP 자산 메타/접근 결과](../handoffs/evidence/2026-10/2026-10-03-notebook-selection/mcp-assets.json), [증거 파일 hash](../handoffs/evidence/2026-10/2026-10-03-notebook-selection/manifest.json).

이번 범위는 자료 재수집·소스 비교·계획 수정이다. 새 가중치 다운로드, GPU 학습/추론, Kaggle 실행·제출·게시를 하지 않았다. 외부 점수 검색 MCP 인증 실패는 앱의 수집된 숫자 점수와 인증된 로컬 source fetch로 보완했다.
