# EXP085 결과 · B/RN public3 실행

2026-10-04 04:42 KST · root. M02–M05/M07–M10: 조건부 작은이득과반대근거 보존,실행PASS와성능구분. [직전교정](2026-10-04-0423-release-review.md).

- **EXP085 완료:** 04:30:56 full SUCCESS/resourceCOMPLETE,각3448updates/skip0·pairtrace/초기logits/RNG exact. pilot포함4941.203초≤7200. 04:36고정평가1회·888건/873component·500/500유효. 처리−대조+.001852 CI[−.002560,+.006766],처리−원B+.002154 CI[−.003634,+.006973]. 양의개발후보보존/교체·자동확대보류. Baker−.009430(명목CI음수),all_nonzero654의원B대비−.003585/일부0인234+.017496도함께보존. 골절P29/A없음으로과거B−A손실회복입증불가. [12소견·한계·판정](../research/exp085-relative-position-result-2026-10-04.md). 독립정합성12검사PASS·평가재실행금지.
- **B/RN v3 CPU교정완료:** 52검사재사용·source/manifest/실제payload연결root확인·독립R1/R3 PASS. postrename/ack실패rollback·최종clock보완,원v1/v2보존. [독립검토](evidence/2026-10/2026-10-04-b-rn3f-release-v1/independent-code-review-v3.json).
- **실제 private 원격:** `taeyangg4/knee-b-rn3f-release-v1` version1을04:40:39push,04:41RUNNING. stage`.local/b-rn3f-release-staging/20261004-v3`,manifest`a2fa8118649bb438bd810ed8b8ab156798f70f124d7ff1aa2be2692d2a6a3f5f`. platform900/public내부600·최대0.5GPUh·retry0,04:37quota27.70h. [허가·명령](evidence/2026-10/2026-10-04-b-rn3f-release-v1/root-remote-decision.json),[실행영수증](evidence/2026-10/2026-10-04-b-rn3f-release-v1/remote-push-receipt.json).
- **후속:** 정확version1의`b-rn3f-release-v1/work/result.json`→고정SHA/크기zip→CSV/최종ack로그를선택회수. COMPLETE만PASS금지. public3기능검증이며hidden시간설정/전체성능은HOLD. 원6weights/100결과불변·EXP085가중치대체없음.
- **다음투자:** dicom_header_io가큰미검증표현/정보가설3안과최소비용을읽기전용비교중,04:55목표/15분. 새구현/학습/원격없음. 기존감사재수집금지.
- **재개:** 같은예약04:55(원격ETA04:50–56/queue시작미확인,CPU초안). D01 56796964첫조회05:02/승인사용완료·재제출금지. 새제출/공개/유료별도승인.
