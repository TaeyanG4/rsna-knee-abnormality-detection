# EXP087 CPU 준비 독립 검토 계약

- 시작: 2026-10-04 06:55 KST, 담당 `/root/anchor_inputs`; 목표07:35·상한07:40. 근거: [root 결정](root-cpu-implementation-decision.json), [완료 계약 검토](../2026-10-04-post086-priority/coverage-independent-review.json).
- 범위: producer가 완료를 알린 새 helper/runner·focused tests·exact plan의 변경분만 검토한다. scalar/합성 테스트 허용. 실제pixels/header 재감사·fixture/census 중복열거·GPU/model/label/OOF·중앙문서 수정·actual pilot 실행은0.
- 입력 보존: 원 cache/mask/spec/UID/epoch로 original C를 먼저 산출한다. slot2의33..42 출현순번1·3·5만 치환하며 trainK12/evalK24·다른window·순서·intensity/ImageNet 정규화·원 RNG draw를 보존한다. disabled/forced-original/noneligible/m0·1은 exact no-op이어야 한다.
- 대안 계약: 원series와 strict plane/fluid/FS가 같고 기존 모든slot에 미사용인 lexical 첫SeriesUID만 고정한다. round-half-to-even으로 sorted valid-center ordinal에 대응; 하나의triplet은 한series·세유효mask·세상이nativeindex. anatomy 정렬 가정과 다음후보search 금지.
- 근거 연결: producer가 동결한4–8개 G1 T study·exact current/alternate·기존header·archive member/source identity를 읽어 연결의 검증방법을 확인한다. 과거F2/maxspan을 대체사용하지 않으며 교집합·해시 전수산출은 반복하지 않는다. 원B lineage는 완료 검토 재사용.
- 오류 구분: 정상 absence/metadata/읽기성공 geometry/확인된duplicate/valid-triplet없음만 원본fallback. 누락·parse/decode·충돌metadata·identity mismatch·budget초과는hardFAIL. SOP중복/render동일성과 임상정보/near-duplicate는 구분한다.
- 자원/정리: 8pairs·16series·192plane·2048headers·64MiBheader parser·1GiBpixel payload·21,676,032renderedbytes·process-treeRSS4GiB·600초·retry0을 시작전/읽기중/자식종료까지 점검한다. 새출력/no-overwrite·실패표식·자식/임시파일 정리·실효worker receipt와 중단 시 비완료 판정을 확인한다.
- 검사: source를 먼저 읽고 import 부작용을 확인한 후에만 synthetic focused tests. 정상·경계·실패 예제를 확인하고 결과·제한을 own review 파일에 저장한다. producer 구현 수정은 직접 하지 않고 root/producer에게 발견사항을 전달한다.
- 비교/판정: 동일generic init/G1/8epoch/recipe/seed/order에서 의도한slot2 입력만 바뀌고 no-op 검증 및 원batch4/K24/precision/order/pooling이면 B8 재사용 가능. 공통학습계약 변경은matched control 필요. 최종 독립판정은GO/HOLD/FAIL과 exact blocker로 반환하며 actual pilot은root 별도판정 전HOLD.
