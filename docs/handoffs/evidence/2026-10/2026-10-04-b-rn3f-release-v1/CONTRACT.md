# B/RN test-only release V1 · 동결된 CPU 구현 계약

2026-10-04 04:27 KST. 코드·합성52검사·최종 stage v3·설치SDK 검사 완료, 독립 코드검토/실제public3 대기. [root 허용범위](root-cpu-decision.json)와 [독립 인수계획](review-plan.md)을 적용했다. 이 문서의 release는 실행부 버전이며 제출 승인이나 성능 개선 의미가 아니다.

최종 source SHA `4bca01aa59eac625fd8b85707b63e44f845873c0b61aa67425dad446e09356ae`, stage `workspace://.local/b-rn3f-release-staging/20261004-v3`, release-manifest SHA `a2fa8118649bb438bd810ed8b8ab156798f70f124d7ff1aa2be2692d2a6a3f5f`. manifest가 코드·감시기·기존 root/독립100건 검토·codec reference·fixture4개를 묶으며 원본 serving asset SHA를 별도로 고정한다. 이 manifest가 신규 protocol이다. 앞선 CPU stage v1/v2는 보존·실행금지다. v2의 post-rename metadata/ack rollback gap은 [root 교정](root-pre-dispatch-repair.json)에 따라 수정했으며 [이전 verification](pre-v3-verification.json)에 실패 근거를 보존했다.

## 입력·모델·집계

[test_contract](../../../../../../tools/b_rn3f_release.py:95)는 `test.csv`, `test_series.csv`, `test_series/`, `sample_submission.csv`만 요구한다. train 파일·Report·원본 캐시를 열지 않는다. 공식 sample은 `StudyInstanceUID+12target`의 정확한 열 순서, 비어 있지 않은 안전한 문자열 UID, 중복/누락/extra가 없는 test와 동일집합을 요구한다. 예측 계산 순서는 공식 sample 순서로 고정하고 test 원래 순서도 기록한다. sample의 예측값은 사용하지 않는다.

worker는 변경하지 않은 `b_rn3f_serving.py`의 asset 검증/activation·ObservedCorpus·atomic_outputs와 frozen runtime/ensemble/legacy DICOM adapter를 재사용한다. 원본 CLI의100/public3 제한과1800초 제한을 변경하거나 monkeypatch하지 않는다. 새 worker가 독립적으로 전체 test UID를 구성하고 원본 pure functions에 전달한다. exact B8/RNclip20 F0–2, strictFP16, 한 visible T4, batch1/resident6, K24/224/ImageNet/within-slot-valid를 유지한다. 현재 구현은 같은 입력을6개 member에 공유하며 전부 완성된 raw float32 `[6,N,12]`를 저장한다.

최종식은 `mean_f[.9*rank_pct(B_f)+.1*rank_pct(RN_f)]`, 전체N·target/member별 average ties/N·float64, 최종 rerank 없음이다. `validate_result`가 raw의 정확한 재집계와 내부CSV 일치를 확인한다. 신규 wrapper는 로컬/T4 수치 gate를 다시 만들지 않고 이전 정확100 검토를 출처로 묶는다. test3 rank값을100건 rank값에 비교하지 않는다. 신규 architecture/weights·anchor·EXP084/085 replacement·routing·비율 변경 없음.

## 시간과 자원

첫 notebook cell의 `import time` 직후 monotonic을 시작하고 이후 단계마다 초기화하지 않는다. root의 추가 전달대로 **N≤3은600초**, 나머지는32100초의 내부 deadline이다. 실제 저장 요청의 platform timeout은900초이며 **공개3건 qualification용**이다. 향후 의도하는 full platform32400초는 현재 요청에 반영되지 않았다. [timeout 결정](root-timeout-decision.json)과 [설치 client 근거](timeout-client-review.json)에 따라 hidden scoring 상속/override는 UNKNOWN, full readiness는 HOLD다. 내부600초 gate는 notebook 전체를 반드시600초에 종료시키는 플랫폼 hard kill이 아니다. supervisor에는 남은 시간에서 최종검사/압축/게시용30초를 뺀 timeout을 넘긴다. 부족하면 시작하지 않는다.

실제N에 대해 사전 고정 `1.3*(877.354403408*N/100)+300 <=30600`을 검사한다(정수 최대N2656). 이는9h 대비1800초의 readiness 여유이며, full 내부 deadline32100초의300초 여유와 다르다. **운영 화면 검사일 뿐 hidden 전체 성공 증거가 아니다.** 초과 시 자동 coefficient 완화·skip·two-T4·추가시도 없음.

기존 process_resource_receipt가 실제worker와 자손을100ms 간격으로 감시한다. 성공에 요구하는 조건은 고정 supervisor SHA/정확 argv·관측 workerPID, COMPLETE/child0/wrapper0/cleanup 전child0, telemetry 샘플/RSS>0, 접근/샘플 실패0, retry0, cleanup action/error/survivor0, Linux cgroup v2 start/end 완전, 실제 limit 안의 current/peak, max/oom/oom_kill event0이다. timeout·worker의 자연종료 판정 순서와 별개로 resource.elapsed<timeout, 반환 후 전체elapsed, 최종 rename 전후 absolute deadline을 모두 검사한다. cgroup peak는 shared session 범위일 수 있고 RSS 샘플은 순간 peak를 놓칠 수 있다는 기존 한계는 유지된다.

## 최종 환경 codec

기존 pydicom3.0.2 wheel에서 압축3개와 MR_small uncompressed fixture를 그대로 가져왔다. 총24,350bytes이며 원본/hash를 검증했다. pinned pydicom3.0.2/python-gdcm3.2.6/OpenCV4.12/timm1.0.19를 worker의 offline 환경에 설치한 뒤 생산 DICOM과 동일한 기본 `pixel_array` 경로로 실제 decode한다. syntax/shape/dtype/pixel SHA를 기존 참조와 비교하고 MR lossless/uncompressed는 exact array equality를 요구한다. 실제 decode 범위는 .70/.90/.91, .57은 availability만이다. 모든 MRI/hidden 압축 구문을 대표한다는 주장은 없다. 로컬에서는 decode/GPU/모델 추론을 실행하지 않았다.

## receipt와 atomic 최종 파일

1. `work/launch.json`: test metadata SHA/UID·운영 화면 검사·첫 시작/deadline·정확worker argv/timeout.
2. worker: `codec.json`, 내부 `result/members.npz`, `result/predictions.csv`, `result/receipt.json`. 실패 시 `worker-failure.json`. 이 단계는 최종 `submission.csv`를 쓰지 않는다.
3. supervisor 종료 후 parent: worker/raw/CSV·실제resource/cleanup·source/metadata 연결 검사 → `parent-receipt.json(PARENT_COMPLETE_RESOURCE_CLEAN)`과 evidence.zip. 모든 hash를 다시 확인한 뒤에만 최종 임시CSV를 쓴다.
4. 공식UID/열·finite/range·원raw 집계hash와 deadline 검사 후 같은 디렉터리에서 `/kaggle/working/submission.csv`로 rename한다. rename 이후 metadata stat/receipt SHA와 완료ack 전체를 같은 rollback scope에서 처리한다. 임시ack 저장→atomic rename 직후 실제clock/deadline도 확인한다. 늦은 rename/metadata/ack 또는 최종 receipt 저장 실패 시 **이번 실행이 생성했고 원본hash가 그대로인 파일만** 되돌린다. 사전존재/외부변경 final은 보존한다.
5. `work/result.json`은 `PASS_TEST_COHORT_ARTIFACT_ONLY`+finalCSV SHA/bytes+부모/worker/resource/codec/evidence hash를 요구한다. 검증 실패는 FAILED이며 자동 재시도하지 않는다. 강제 세션 종료/파일시스템 고장으로 receipt가 없으면 최종파일이 보이더라도 미완료다. CSV와receipt 여러 파일이 단일 filesystem transaction이라고 주장하지 않는다. result.json의 elapsed_seconds는 ack쓰기 직전 실제clock이다. ack rename 직후 실제clock은 반환값/노트북출력의 post_ack_elapsed_seconds·post_ack_deadline_pass로 별도 기록한다. 자기 write 이후 clock을 같은 파일에 소급 기록한다고 주장하지 않는다.

## 재개·검증 범위

[verification](verification.json): 새focused52검사 PASS0.70초, 동적N1/2/3/100/131·tie/부분member·UID/schema 오류·test-only mount·time 경계/late exit0·OOM/cleanup/telemetry 오류·rename/acknowledge 실패·기존 final 보존. 실제stage의 첫cell clock/AST/private/offline/T4/Docker·압축payload roundtrip도 확인했다. 기존31/37검사·local100·T4100 재실행0. 실제SDK request49,976/script49,292bytes, credential/network0. weights 재업로드0.

독립 source/stage 검토 후 root가 실제quota를 확인하고 [execution-plan](execution-plan.json)의 private push **1회/900초/retry0**만 판단한다. 결과는 정확 remote revision에서 `b-rn3f-release-v1/work/result.json`→지정hash/bytes의evidence.zip→`submission.csv` 순으로 회수한다. 공개3의 실제출력행/UID/raw-rank 재계산·전체 receipt와 finalhash·노트북 출력의 post-ack clock을 독립 확인한다. kernel COMPLETE만으로 승인하지 않는다. 현재900초 버전으로 전체 제출 준비완료/승인요청을 하지 않는다. 별도32400초 버전의 범위·비용 또는 scoring override의 근거가 필요하다. 공개0.943 개선·독립 정확도·hidden full runtime은 미확정이며 대회 제출/공개/유료는 사용자 개별 승인이다.
