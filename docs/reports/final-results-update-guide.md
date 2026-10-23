# Updating final results / 결과 갱신 안내

[English results](final-results.md) · [한국어 결과](final-results.ko.md) · [Input record](final-results-data.json)

Both result pages use one JSON record, so each supplied score is entered once.
The renderer validates before writing and writes only these two pages. It does
not query scores, submit, select finalists or modify historical evidence.

## 1. Collect the evidence

For each result, collect the exact submission ref/version, actual inference status,
displayed public/private score, observation time in KST and a dated source. A source
can be a Taeyang message, local screenshot reference or canonical discovery ID.
Keep credentials and patient material out of evidence. The user expectation of
results on October 23 is not an official release-time check.

한국어: 제출 번호·버전·완료/오류 상태·공개/비공개 점수·확인 시각·출처를 함께 기록합니다.
어느 제출의 점수인지 불명확하면 빈칸을 유지합니다.

## 2. Fill the input record

Edit [final-results-data.json](final-results-data.json). Unknown values stay JSON
`null`; do not use zero, empty strings or guesses. Scores are strings with the
exact displayed decimals, preserving precision and trailing zeroes.

| Field | What to enter / 입력 내용 |
|---|---|
| `updated_at_kst` | Record update time with `+09:00` / 기록 갱신 시각 |
| `candidates[].result.inference_status` | `not_recorded`, `running`, `complete`, `error`, `cancelled` / 관측된 상태 |
| `candidates[].result.public_score` | Newly observed public display or `null` / 새 공개 점수 |
| `candidates[].result.private_score` | This exact submission's private display or `null` / 해당 제출 비공개 점수 |
| `candidates[].result.observed_at_kst` | Actual observation time / 실제 확인 시각 |
| `candidates[].result.evidence` | Dated source/reference / 날짜가 있는 출처 |
| `final_selection.state` | `unconfirmed` until actual selection is confirmed, then `confirmed` / 실제 선택 확인 상태 |
| `final_selection.candidate_ids` | Two distinct accepted IDs in displayed slot order / 확정된 두 후보 |
| `final_selection.observed_at_kst`, `evidence` | Confirmation time and source / 선택 확인 시각과 출처 |
| `competition_outcome.state` | `pending`, `provisional`, `final` / 미기록·잠정·최종 구분 |
| `competition_outcome.private_score`, `rank`, `total_teams`, `award` | Competition outcome; never average the pair / 대회 전체 결과 |
| `competition_outcome.observed_at_kst`, `evidence` | Result observation time/source / 결과 확인 시각과 근거 |
| `interpretation` | Optional entries with `text_en`, `text_ko`, `evidence` / 근거를 붙인 양언어 해석 |
| `change_log` | Append `at_kst`, `source`, `change_en`, `change_ko` / 기존 기록을 보존하고 변경 추가 |

The eight accepted baseline identities and old public displays are frozen. F4 is
an unaccepted placeholder until a receipt supplies its kernel/ref/version/time.
Then set `acceptance` to `accepted` and fill `acceptance_evidence`, with a change-log
entry. Recovery submissions get separate IDs/rows with their own receipt; do not
overwrite an old ref. Keep new rows' `baseline_public` null.

한국어: 기존 8개 후보의 식별자와 기준 공개 점수는 고정입니다. F4는 접수 확인 후 식별자를
채우고 복구 제출은 새 행으로 추가합니다. 원래 제출을 새 버전으로 덮어쓰지 않습니다.

New findings still follow the canonical discovery/evidence workflow and original
review rules; cite those IDs here. This record is the single input for the two
presentation pages, not a replacement for canonical research evidence.

## 3. Render and inspect

Run from the original project root:

```powershell
python tools/render_final_results.py
python tools/render_final_results.py --check
python tools/verify_research_visuals_20261022.py
```

The first command validates before writing. The second checks generated-page
drift without writing. Correct input errors rather than weakening checks. The
tool records an actual confirmed pair; scientific eligibility is still governed
by the existing policy and the user's decision.

한국어: 영어·한국어 페이지가 함께 생성됩니다. 이어서 표시값·식별자·출처·미확인 항목을
읽어 확인합니다. 점수 기록 기능이 최종 선택을 대신하지 않습니다.

## 4. Publish from a clone

The repository is published directly from a clone of
https://github.com/TaeyanG4/rsna-knee-abnormality-detection. The copy-only publisher
of the original workspace (`github-repo-sync-v104.ps1`) was retired on 2026-10-23,
when the repository layout changed; it now stops at once if started.

```powershell
git clone https://github.com/TaeyanG4/rsna-knee-abnormality-detection.git
cd rsna-knee-abnormality-detection
# edit docs/reports/final-results-data.json, then:
python tools/render_final_results.py
python tools/render_final_results.py --check
git add docs/reports
git commit -m "Record verified final results"
git push
```

The repository is public. Before pushing, read the diff for study or series
UIDs, report text, labels, credentials and verbatim owner remarks, and keep every
file under 2 MB. Verify remote main and both results pages after the push. Never
force-push over another session's commits.

한국어: 저장소를 clone해서 JSON을 고치고, 페이지를 다시 만든 뒤 `--check`로 확인하고 푸시합니다.
공개 저장소이므로 푸시 전에 UID·보고서 문장·라벨·인증 정보·소유자 발언 원문이 없는지 diff를 확인합니다.

## 5. Preserve the baseline

- Keep the 18:15 report and four figures at their original scientific cutoff.
- Put later outcomes on the separate results pages; preserve old scores and reasoning.
- Separate public displays, hidden completion, private submission scores and overall rank.
- Update the README results link/status without making the old snapshot appear live.
- Final rank alone does not prove an individual ablation caused the result.

한국어: 결과를 안 뒤 과거 판단을 다시 쓰지 않습니다. 보존된 기준 기록과 새 결과를
나란히 비교하며, 최종 순위만으로 개별 실험의 인과 효과를 단정하지 않습니다.
