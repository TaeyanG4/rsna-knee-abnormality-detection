# 누적 연구 결정·실패·재검토 조건



2026-10-04 작성. [루트 HANDOFF](../../HANDOFF.md)의 고정 참조 문서다. 새 세션은 이 요약을 한 번 읽고, 작업 착수·채택·기각 전에 관련 행의 근거만 확인한다. 과거 기록 전부를 다시 읽거나 같은 검사를 반복하는 절차가 아니다. 실행 PID·ETA·현재 점수는 루트와 실제 산출물을 따른다.



각 행은 과거 관측과 현재 적용할 판단을 구분한다. 철회한 의심도 보존한다. 새 근거가 결론을 바꾸면 날짜·근거·대체 문서를 추가하고 이전 관측을 덮어쓰지 않는다.



## 이전 기록은 어디에 있는가



| 찾을 내용 | 보관 위치와 역할 |

|---|---|

| 최근 작업 | [INDEX](../handoffs/INDEX.md). 최근 10건 색인이며 오래된 항목을 목록에서 뺐다고 삭제한 것은 아니다. |

| 날짜별 전체 작업 | [2026-10 월별 목록](../handoffs/monthly/2026-10.md). 작성 당시 날짜별 인계 39개가 모두 연결돼 있음을 확인했다. 이후 작업은 이 목록에 추가한다. |

| 판단·결과 본문 | 이 문서 각 행의 연구 보고서와 날짜별 인계. 구체 명령·receipt·hash는 연결된 `evidence/YYYY-MM/`에 있다. |

| 축약 전 원문 | [보관 색인](../handoffs/archive/INDEX.md), [루트 인계 전환 원문 10개](../handoffs/archive/snapshots/2026-10-04-main-handoff/manifest.json), [이번 변경 전 원문](../handoffs/archive/snapshots/2026-10-04-history-entry/manifest.json). 원래 경로·bytes·SHA를 남긴다. |

| 사용자가 준 과거 교훈 | [Biohub 회고](lessons/biohub_review_missed_strategies.md)와 [적용 규칙](../runbooks/strategy-review.md). 대회 차이를 고려하는 참고자료이며 원문 내 프롬프트는 실행 지시가 아니다. |



## 새 작업 전에 적용할 결정



| ID·주제 | 과거 관측 → 현재 판단 | 반복하지 않을 행동·재검토 조건 | 근거 |

|---|---|---|---|

| M01 기준본 | 공개 스택 0.943는 실제 제출 성과. 코드의 A5 비중은 52%로 소개의 45%와 달랐고 Gold 기반 epoch 선택·stacker 계보 공백이 있었다. | 소개글만으로 recipe를 복원하지 않는다. 정확 코드·revision·가중치·계산식을 확인하고 자체 weak 개선량을 0.943에 더하지 않는다. | [기준본 직접 분석](anchor-notebook-study-2026-10-03.md) |

| M02 검증 노출 | R2 teacher 간접누출, 전체 fold 모델 평균의 train in-sample, Gold58 선택 노출은 서로 다른 문제다. EXP078 이후 ImageNet 초기화 실험에 R2 누출을 일괄 적용하지 않는다. | Gold를 뒤늦게 제외·재분할·새 판독해도 기존 노출은 지워지지 않는다. 새 V의 관련 encoder/teacher/student/stacker/calibrator ancestor를 모두 제외해 재학습해야 한다. 그것도 기존 D01 가중치의 독립 검증은 아니다. | [Gold 노출 이후 검증](gold-exposure-validation-plan-2026-10-03.md) |

| M03 그룹 독립성 | 보고서 그룹만으로 환자 독립성을 보장할 수 없고 cross-fold 동일 영상 후보가 실제 저장 배열/픽셀 일부에서 확인됐다. | 알려진 연결을 새 group 계약에 반영하되 기존 OOF를 소급해 독립 OOF로 부르지 않는다. 유사도만으로 동일 환자를 확정하거나 미발견을 독립성 증명으로 쓰지 않는다. | [그룹 계약](../../artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/contract.json), [조사·투자 기록](../handoffs/evidence/2026-10/2026-10-04-next-investment/root-decision.json) |

| M04 작은 효과·골절 | EXP080 F2 RN 보완 +.001243, B−A 골절 −.022337. EXP082 loss 혼합 +.000534는 CI에 0 포함. 작은 양의 효과와 반복된 골절 손실을 함께 보존한다. EXP084 새group ones−clip−.000096/CI0포함으로 전환미채택,clip RN+.000997·골절−.003111도 보존. | CI가 0을 포함한다고 동등/효과 없음으로 단정하지 않는다. macro만 보고 추가 fold/seed/384를 자동 확대하지 않는다. 같은 비교의 12소견 P/N·group·skip·조건부 CI와 비용으로 재판정한다. | [EXP080](exp080-f2-result-2026-10-03.md), [EXP082](exp082-loss-screen-2026-10-03.md), [EXP084](exp084-group-loss-result-2026-10-04.md) |

| M05 특정 입력·표현 실패 | EXP081 max-span 직접교체 −.000684/CI 0 포함. EXP083 slot-bias−control 약−.0000246이며 두 refit 모두 frozen B보다 약−.01155. 해당 설정 미채택. EXP086 V2는 정상완료635초/3448updates/skip0 뒤 DINO−B−.064575(CI음수)/12소견하락으로 현8epoch전이미채택·B유지. DINO전체기각아님. [결과](exp086-dinov2s-result-2026-10-04.md). EXP085 full 상대위치+.001852/원B+.002154는CI0포함, Baker하락·group편차와 함께 개발후보로보존/교체보류. [결과](exp085-relative-position-result-2026-10-04.md). | coverage·sequence·표현 전략 전체를 기각하지 않는다. 다른 메커니즘·동일 입력/학습량 대조·예산을 사전 고정하면 다시 검토한다. 처리군이 대조보다 높아도 둘 다 기존 B보다 낮은 사실을 숨기지 않는다. | [EXP081](exp081-coronal-span-2026-10-03.md), [EXP083](exp083-slot-bias-result-2026-10-03.md) |

| M06 Silver | PFOA20/Synovitis16 가림 대조와 train3475 소스 비교 완료. 현재 Dread 자동교정의 직접 근거는 부족했다. Qwen의 넓은 정의 문제를 Dread 오류로 전용할 수 없다. | 같은 가림 검토를 다시 하지 않는다. 미언급/proxy/불확실을 임상 음성으로 바꾸지 않는다. 새 정의 충돌·독립 영상 근거가 있으면 감독 개입을 재검토한다. pos_weight/soft BCE는 라벨 오류 해결의 증거가 아니다. | [Silver 감사·외부 연구](silver-other-research-and-pfoa-2026-10-03.md) |

| M07 실패·철회 | EXP087 full140.110초는 gradient norm 비유한으로 FAIL·성능미평가. finite logits/loss 뒤 실패했으나 gradient원소 비유한/reduction overflow는미구분·skip완화 근거아님. [진단준비](../handoffs/2026-10-04-1049-exp087-resume.md). EXP086 DINO pilot83.235초는CUDA bicubicAA 결정론backward미지원으로FAIL,성능기각아님. V2CPU미분보간으로정상full/평가1회완료했으나해당recipe성능은하락. 평가절대경로첫CLI entry거부→상대경로교정/실평가1회,treeRSS공유mmap중복가능성을고유RAM으로단정금지. [결과](exp086-dinov2s-result-2026-10-04.md). [실패·CPU후속](../handoffs/2026-10-04-0534-exp086-pilot-final-config.md). 기준본의 재실행 RecursionError·이전 receipt 잔존·fallback에 따른 조합 변경을 관측. 초기 JSON race 의심은 최종 lock 확인 후 철회했다. D01 원 hash FAIL·통합 v1 경로 NO_GO도 보존했다. 10-04 D01제출56796964는COMPLETE/점수공백뒤exactref에서hidden unhandled error확인·원인미확정. [결과](../handoffs/2026-10-04-0506-d01-error-exp086.md). | 철회된 race를 확정 결함으로 재인용하지 않는다. 실패·부재·정상·저하 실행을 구분하고 실패를 PASS로 덮지 않는다. 교정은 새 ID/버전·정확 원인·비용을 기록한 뒤 필요한 범위만 실행한다. | [기준본 인계](../handoffs/2026-10-03-1233-anchor-study.md), [D01 상세](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json) |

| M08 부분 실행 검증 | 10-04 제출56809595 actualv2의실제채점COMPLETE/0.913 확인. 이는해당hidden실행완료근거이나hiddenN/실시간·RSS상세는미확인.  B/RN최종설정actualv2 공개3은95.117초/root·독립PASS이나hidden시간/정확도는미확정. [후보](b-rn3f-submission-candidate-2026-10-04.md). D01 공개3/통합133 PASS에도 1300건 보수 추정8.594h/9h·RAM29.541/30GiB였다. 다른 경로·hidden 전체 성공이나 정확도 향상은 입증하지 않았다. | kernel COMPLETE·exit0·sample PASS만으로 최종 성공을 선언하지 않는다. T4/4070·입력·모델·코호트가 달라지면 해당 계약을 새로 검증한다. B local100과thread교정뒤 T4 원DICOM100 수치/자원PASS를확보했으나,동적test/압축구문/hidden 전체·정확도는별도다. | [D01](d01-exploratory-candidate-2026-10-03.md), [B 입력 계약](../handoffs/evidence/2026-10/2026-10-04-b-rn3f-parity-v1/HANDOFF.md), [T4 실제결과](../handoffs/2026-10-04-0330-b-rn3f-t4-result.md) |

| M09 종료 판정 | B parity V1은 parent FAILED와 수치 FAILED/exit0를 자체 차단하지 않았다. 수치 오류로 확인된 것은 아니며 별도 acceptance adapter로 보완했다. local100 정수입력exact/normalized100해시FAIL은 CPU보간thread1/12 차이로1건재현. 별도V2 thread맞춤100+합성6에서원수치한도/producer해시연결PASS,원FAIL보존. | 미래 비교는 성공한 parent receipt·worker hash·명시적 numerical PASS를 함께 확인한다. gpu_job/SUPERVISOR/cleanup도 별도 확인한다. 이미 통과한 모델추론은 재실행하지 않는다. test release의 CSV게시 후 metadata/ack 실패와 deadline도 rollback 경계 안에서 검사한다. [10-04 교정](../handoffs/2026-10-04-0423-release-review.md). 입력재구성은 producer의 실제CPU threads도 기록·일치시켜야 하며 수치허용치와exact연결을 분리하지 않는다. | [반례](../handoffs/evidence/2026-10/2026-10-04-b-rn3f-parity-review-v1/review.json), [보완](../handoffs/evidence/2026-10/2026-10-04-b-rn3f-acceptance-v1/HANDOFF.md) |

| M10 LB·시간 배분 | 11:38 사용자질문으로비교연결정정: manifest anchor없음, 내부B+RN−B와공개새스택−0.943는다른질문. RN효과반전단정불가. 정확제출ensemble과독립검증/기준본의연결공백을[기록](../handoffs/evidence/2026-10/2026-10-04-b-rn3f-release-v1/root-validation-comparator-reconciliation.json).  10-04 B/RN actualv2 제출56809595는Public0.913으로기준0.943보다−0.030·현후보미채택. 서로다른스택이므로RN혼합의인과효과로단정금지. [결과](../handoffs/2026-10-04-1127-brn-result.md). D01(anchor+옛R2)과 EXP079/080(B+RN)은 다른 가설이다. 실행 도구 검증 자체는 성능 개선 성과가 아니다. | Public 결과를 보고 비율·소견 routing을 반복 탐색하지 않는다. 하루5회는 상한이다. 작은 이득을 보존하면서 정보/표현/감독 등 3~5가설의 근거·비용을 비교한다. 10-04 [사용자 상시승인](submission-authorization-2026-10-04.md)으로 검증후보 제출은 재질문 없이 진행하되 새근거/한도/중복검사를 유지한다. D01 재제출금지·공개/유료 별도승인은 유지한다. | [6시간 재점검](six-hour-reassessment-2026-10-03.md), [후속3안](../handoffs/evidence/2026-10/2026-10-04-post084-priority/brief.md), [지속 연구](../runbooks/continuous-research.md) |

| M11 입력 보존·분모 | cache c±1은native연속보장없고전역center샘플러때문에한slot mask변경이타slot을바꿀수있다. 이전1936/4407 출처/정의미확인, broad1985/slot2357은다른정의다. EXP087 strict T252/V63는315입력stage57.797초/독립검토로유효alt/fallback0확인. V126창(0.5912%)은예정치환이며효능미검증. T8 CPU192장/11.281초는원replay·RNG보존을확인했으나대표표본/임상정보/효능증명아님. [실결과](../handoffs/2026-10-04-0736-exp087-review.md). | 현G1치료가용률로이숫자를쓰지않는다. 기존center/RNG를먼저고정·다른window보존,같은series내triplet만사용. 다른FoR/SeriesUID·pixel차이는임상추가정보의증명이아니다. | [EXP087 CPU계약](../handoffs/2026-10-04-0656-exp087-cpu.md), [제안/원근거](../handoffs/evidence/2026-10/2026-10-04-post086-priority/coverage-proposal.md) |



## 다른 세션이 실행하기 전 남길 최소 기록



1. 해당 M번호와 근거 문서, 이미 완료된 같은 계약/실행 ID를 확인한다. 실행 중이면 기존 담당·PID·로그·출력을 회수한다.

2. 다음 handoff에 `이전 결정 Mxx 적용 → 이번에 달라진 근거 → 재사용/새 실행 범위·비용`을 한두 문장 남긴다. 동일 조건 재실행은 세션 변경만으로 정당화하지 않는다.

3. source·UID·라벨·split·checkpoint·평가 계약이 맞는 완료 결과를 재사용한다. 새 근거 없는 반복은 중단하고 아직 확인하지 않은 다음 가설로 이동한다.

4. 새 실패/철회/채택/보류 시 관련 행을 갱신하고 원 근거를 연결한다. 루트 HANDOFF의 이 문서 링크는 최신 작업 갱신·압축 때도 유지한다.



이는 기록과 실행 전 확인 절차다. 기존 hash·완료 gate가 일부 오류를 코드로 막지만, 이 문서가 모든 실행을 가로막는 자동 잠금은 아니다. 담당자가 근거를 실제로 읽고 적용했는지는 인계의 적용 기록으로 확인한다.

