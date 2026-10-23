# EXP087 실행 준비 독립 검토 · 2026-10-04 07:32 KST

- 판정: **freeze-v2 HOLD**. 입력 설계가 아니라 supervisor의 두 실패 경로를 최소 수정해야 한다. [상세](independent-review.json), [root 수정범위](root-pre-pilot-repair-decision.json). 실제pixel/header/model/GPU/label/OOF·fixture 재열거·pilot실행0.
- 원본 보존: 원mask·UID·epoch·seed42로C를 먼저 선택하고 slot2출현1·3·5만 치환한다. trainK12/evalK24, intensity RNG, 다른window는 유지된다. 원StudyWindows와의20개 합성PASS(1.593초)는 재사용했으며 재실행하지 않았다. 실제cache replay는 아직 미실행이다.
- 대안·실패: lexical첫strict시리즈만고정; sorted valid-center의round-half-even·한series triplet·상이native3개를 확인했다. geometry/SOP중복/render동일/유효창없음은원본fallback, 누락/parse/decode/결박불일치/예산초과는FAIL이며다음후보search없다.
- T·source: freeze코드의G1 T3455/V888 digest·T조건·기존header연결, 고정fixture SHA와원ids/masks·picksubset·code/metadata·archive/index/shard stat gate를확인했다. 원12plane replay·CRC/offset/header검증은runtime에실시한다. 전수hash나환자독립증명은아니다.
- 차단1: 마지막poll599.95초→정상exit600.05초(cap600)인scalar반례가 `PASS,wall_seconds=600.05`를재현했다. wait뒤최종elapsed를판정해야한다. 현timer는preflight후시작하므로whole-command하드제한이아니다.
- 차단2: Popen직후job.json의합성OSError에서kill·wait·supervisor기록없이반환했다. post-spawn전체try/finally로자식종료/회수·best-effort FAIL기록이필요하다. 두반례는run만AST추출한isolated fake환경이며실제프로세스/파일/데이터0.
- 예산:8pair/192plane/1GiBpixel/64MiBheader/21,676,032rendered/4GiBRSS/600초/retry0. 100msRSS감시는OS hardquota가아니며순간초과가능성을명시했다. 실패부분출력은보존하고자동재사용하지않는다.
- B8: 이전lineage검토재사용. 같은초기화/G1/8epoch/recipe와no-op이면완료B재사용가능. 실제GPU평가batch4·K24·precision/order/pooling은이CPU검사가보증하지않는다. 노출weakG1·임상정보/성능/유병률미확인한계유지.
- 다음: v2원본보존→최소수정v3와새binding·두회귀만재검토→root가exactargv실행여부결정. 상한07:40유지; 학습/fullcache/GPU는계속미허가.
