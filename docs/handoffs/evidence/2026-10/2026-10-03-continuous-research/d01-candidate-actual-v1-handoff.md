# 2026-10-03 20:42 KST · D01 candidate 실제version1 출력검증

- 상태: 완료. 기록자: Codex(dicom_header_io). root의 실제 version1 COMPLETE 후 하위 출력7개 exact 회수·독립CPU검증 요청. root가 받은 receipt/CSV2개 보존. push/제출/GPU/생산코드 변경 없음.
- 결과: `taeyangg4/knee-d01-candidate-v3` v1, local stage20261003-v2. 총9파일 다운로드version/hash 검증 PASS. 실제nested competition root에서 test3 원본anchor 소cohort분기(CoAt병렬)·자손정리·R2 2+1건×5모델·전체rank90/10 완료. 입력volume/mask3 exact,5raw FP32평균 exact,최종CSV rank 재계산 오차1.11e-16(CSV float64표현).
- 시간·메모리: anchor321.007초/R279.686초/발행직전401.603초,1800초상한 이내. cgroup kernelpeak21.960/30GiB, OOM증가0·잔존자손0·양GPU반환delta0. F0로컬−T4 max.0001595503/mean.0000449226로 고정.002/.0005 통과.
- 근거: [독립검사](d01-candidate-actual-v1-audit.py), [결과](d01-candidate-actual-v1-audit.json) SHA `9b3a3b64d32d6b2018a92b67af4fa71efb58e766523ab4902ba6bd0f2cab4aca`. 최종CSV `180848bb37be7c7b7be01a8c2e00c46ef68b1974ff9b875c2e6df6460121786f`; receipt `096dddefba71b3b891d85dbdce9638fe458003353bfab9340fc5c67bc81264a1`.
- 한계·다음: public3 실행확인 PASS이며 품질향상/hidden full1300 증거 아님. phase는final 내 JSON동일+exact다운로드SHA 연결(별도byteSHA 필드 없음). 기록된asset refs동일,실제dataset/kernel revision미확정은유지. 기존v1입력hashFAIL/통합경로FAIL 보존. root가 20:40 검증대기HOLD를 당시사실로 남기고20:42검증PASS로갱신·제출별도승인 판단.
