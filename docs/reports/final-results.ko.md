# 최종 결과 — RSNA 무릎 이상 탐지

[English main](final-results.md) · **한국어** · [결과 발표 전 기록](project-summary-20261022.ko.md)

**최종 결과 기록**

준비 시각: 2026-10-22T19:48:36+09:00. 마지막 결과 확인: 2026-10-23T11:22:26+09:00

결과 예정일 2026-10-23은 Taeyang이 알려 준 날짜이며 발표 시각은 미확인입니다. 실제 결과를 받아 기록할 준비 페이지이며 점수를 자동 조회하지 않습니다.

## 1. 대회 결과

| 항목 | 기록 |
| --- | --- |
| 결과 상태 | 최종 결과 기록 |
| 비공개 점수 | 0.943 |
| 순위 | 312 |
| 전체 팀 수 | 6425 |
| 수상 / 메달 | Silver medal (top 5 %; silver cut-off rank 321) |
| 확인 시각 (KST) | 2026-10-23T10:59:57+09:00 |
| 근거 / 출처 | Kaggle private leaderboard via kaggle CLI (312 / 6,425) and Taeyang's screenshot of the leaderboard row |

대시는 미기록을 뜻하며 0점이나 실패를 뜻하지 않습니다. 대회 전체 점수·순위와 개별 제출 점수를 구분합니다. 최종 두 제출물의 평균을 대회 점수로 계산하지 않습니다.

## 2. 확정된 최종 제출물

선택 상태: **확정**. 기본 조합 X + F는 정책상 기본값이며 실제 선택 완료를 뜻하지 않습니다.

| 슬롯 | 후보 | Submission ref | 원격 버전 |
| --- | --- | --- | --- |
| 1 | X | 57415972 | 1 |
| 2 | F | 57475409 | 1 |

선택 확인 시각 / 근거: 2026-10-23T00:30:00+09:00 / Taeyang's screenshot of My Submissions with both entries checked (received about 00:30 KST); plan.md Update 12

[결과 전 고정 정책](https://github.com/TaeyanG4/rsna-knee-abnormality-detection/blob/4172ccf879887e3fa95f00cb714214e17a2805dd/plan.md) Updates 7–10을 따릅니다. 2026-10-23 08:00 KST의 후보 판단 시각과 기록된 08:59 마감은 최종 결과 발표와 별개입니다. 이 페이지는 조합을 자동 선택하거나 변경하지 않습니다.

## 3. 제출별 결과

기준 공개 표시값은 **2026-10-22 18:15 KST** 기록입니다. 이후 결과는 별도 열에 기록합니다. 접수 사실만으로 비공개 추론 완료를 단정하지 않습니다.

| 후보 | Ref / v | 기준 공개 | 새 추론 상태 | 새 공개 | 비공개 |
| --- | --- | --- | --- | --- | --- |
| X | 57415972 / v1 | 0.958 | 완료 | 0.958 | 0.943 |
| p25 | 57427331 / v1 | 0.958 | 완료 | 0.958 | 0.943 |
| F | 57475409 / v1 | 0.957 | 완료 | 0.957 | 0.943 |
| A5 | 57484279 / v1 | 0.956 | 완료 | 0.956 | 0.941 |
| DF | 57500332 / v1 | 0.955 | 완료 | 0.955 | 0.940 |
| U | 57510427 / v1 | — | 오류 | — | — |
| P4 | 57512878 / v1 | — | 완료 | 0.958 | 0.943 |
| U4 | 57514281 / v1 | — | 오류 | — | — |
| F4 | 57526613 / v1 | — | 오류 | — | — |

F4는 기준 시점에 준비 단계였습니다. 접수 원장으로 정확한 커널/ref/버전을 확인한 뒤 결과를 추가합니다. 복구 제출물은 별도 행에 기록하고 기존 제출 식별자를 덮어쓰지 않습니다.

### 정확한 식별자와 근거

| 후보 | Kernel | 접수 시각 (KST) | 기준 근거 |
| --- | --- | --- | --- |
| X | taeyangg4/knee-lb-r6w958k1x3sr20-v2 | 2026-10-20T10:04:32+09:00 | D-B-249/260/276 |
| p25 | taeyangg4/knee-lb-r6w958k1n3s2p25-v1 | 2026-10-20T16:12:20+09:00 | D-B-260/276 |
| F | taeyangg4/knee-lb-r6w958k1n3s2p25flatpv10-v1 | 2026-10-21T18:23:00+09:00 | D-B-274 |
| A5 | taeyangg4/knee-lb-r6w958k1n3s2p25a5v10-v1 | 2026-10-21T22:40:30+09:00 | D-B-275/276 |
| DF | taeyangg4/knee-lb-r6w958k1n3s2p25a5flatpv10b-v1 | 2026-10-22T08:40:49+09:00 | D-B-280/281 |
| U | taeyangg4/knee-lb-r6w958k1uxpv10-v1 | 2026-10-22T13:58:57+09:00 | handoff 2026-10-22 13:59; acceptance ledger |
| P4 | taeyangg4/knee-lb-r6w958k1n4fbp25v10-v1 | 2026-10-22T15:20:09+09:00 | D-B-278; acceptance ledger |
| U4 | taeyangg4/knee-lb-r6w958k1u4fbv10-v1 | 2026-10-22T15:59:21+09:00 | handoff 2026-10-22 15:59; acceptance ledger |
| F4 | taeyangg4/knee-lb-r6w958k1n4fbp25flatpv10-v1 | 2026-10-22T22:39:20+09:00 | plan.md Update 10; preparation only at baseline cutoff |

| 후보 | 관측 시각 (KST) | 새 결과 근거 |
| --- | --- | --- |
| X | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| p25 | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| F | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| A5 | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| DF | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| U | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST; COMPLETE without a score after &lt;= 9.18 h (hidden run failed, D-B-282) |
| P4 | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST |
| U4 | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST; COMPLETE without a score after &lt;= 9.18 h (hidden run failed, D-B-282) |
| F4 | 2026-10-23T10:59:57+09:00 | kaggle competitions submissions (team account), checked by B 2026-10-23 10:59 KST; COMPLETE without a score after &lt;= 9.21 h (hidden run failed, D-B-282) |

## 4. 결과 확인 후 해석

- 최종 두 제출 모두 private 0.943이었고, 0.943으로 표시된 304팀 중 첫 번째였으며 동메달 기준선(0.942)보다 한 표시 단위 위였습니다. (private leaderboard (kaggle CLI); D-B-282)
- 우리 제출 중 private 0.943을 넘은 것은 없었고, 점수가 난 파이프라인 제출 87개 중 21개가 0.943이었습니다. (docs/data/submissions.csv)
- 10월 22일 오후·저녁 제출 4개 중 3개(U, U4, F4)가 약 9.2시간 만에 hidden 실행에 실패했고, P4는 8.24시간 이내에 완료됐습니다. (D-B-280, D-B-282)

관측값의 비교이며 통제된 인과 효과 추정이 아닙니다. 실행 방식·stride·반올림 표시 차이가 영향을 줄 수 있습니다. 공개·비공개·노출된 로컬 근거를 구분하고 기존 실패와 검토 상태를 보존합니다.

## 5. 갱신 이력

| 시각 (KST) | 출처 | 변경 내용 |
| --- | --- | --- |
| 2026-10-23T10:59:57+09:00 | B (Claude Code) from the Kaggle CLI and Taeyang screenshots | 최종 private 결과(0.943, 312 / 6,425위, 은메달), 확정된 최종 쌍 X + F, 마지막 날 제출 결과와 F4를 기록했습니다. |
| 2026-10-23T11:22:26+09:00 | B (Claude Code), repository maintenance | 고정본 링크를 이름이 바뀐 저장소(TaeyanG4/rsna-knee-abnormality-detection)의 커밋 4172ccf로 바꿨습니다. 2026-10-23에 기록을 커밋 하나로 정리하면서 결과 전 커밋 9e1509f가 사라졌습니다. plan.md의 Updates 7-10과 보고서 본문은 소유자 인용 1곳 삭제와 이 페이지로 가는 두 줄 안내를 빼면 그대로입니다. |

## 6. 이 페이지 갱신 방법

[final-results-data.json](final-results-data.json)을 한 번 수정한 뒤 실행합니다:

```powershell
python tools/render_final_results.py
python tools/render_final_results.py --check
```

하나의 기록에서 영어·한국어 두 페이지를 함께 갱신합니다. 입력 항목·근거·게시 절차는 [갱신 안내](final-results-update-guide.md)에 있습니다.

[시점이 고정된 연구 보고서](https://github.com/TaeyanG4/rsna-knee-abnormality-detection/blob/4172ccf879887e3fa95f00cb714214e17a2805dd/docs/reports/project-summary-20261022.ko.md) · [문서 검수 기록](documentation-audit-20261022.md)

<!-- Generated by tools/render_final_results.py; edit final-results-data.json instead. -->
