# EXP087 실제 결과 독립 검토 · 2026-10-04 07:50 KST

- **PASS: 8개 고정 T표본의 CPU 입력 경로 검증 완료.** [상세](result-independent-review.json), [저장배열16개 검증](result-independent-array-checks.json). 성능·임상정보·가용률·학습GO 판정은아니다.
- 실행연결: job의child argv·plan/fixture SHA가최종v3와일치한다. supervisor PASS/exit0,11.281초,RSS643,358,720B,cleanup오류0·FAIL파일0·stderr0. root가보고한parent38076과job의child25068모두07:49독립확인시없었다.
- 실제예산:392headers/604,640parserB/192pixel reads/152,746,838payloadB/21,676,032renderedB, child9.531초. 모든상한이내, CPU1/internal1,retry0. block I/O와RSS OS hardquota를측정한것은아니다.
- 기록일치:8개개별receipt와aggregate가동일하고fixture UID순서도일치한다. 원12plane replay8/8, eval8+train64=72검사의no-op/RNG모두정상. evalK24/trainK12와slot2홀수출현치환을확인했다.
- 실제변경창:eval합16개, train8epoch합56개이며선언된교체창과같다. 무교체20검사는baseline/treatment hash동일. active study는fixture안에서eval/train각8개; broad prevalence로해석하면안된다.
- 저장산출물:16개NPY모두uint8[12,336,336],내용hash와재계산mask hash가receipt와일치한다. 각파일SHA도보존했다. NPY총21,678,080B중형식header2,048B를제외한uint8payload가정확히예산과같다.
- 검토한계:원cache/ZIP/rawheader/decoder·model/GPU/label/OOF·기존test재실행0. 원본replay·RNG는이번runner의실행assertion을검토된소스/합성검사와함께수용했으며원본을다시읽어재생산하지않았다.
- 다음: root가후속범위를별도결정한다. 입력차이는임상추가정보나성능이득의증거가아니며, B8전체비교·G1가용률·fullcache/학습·T4처리시간은여전히별도검증대상이다.
