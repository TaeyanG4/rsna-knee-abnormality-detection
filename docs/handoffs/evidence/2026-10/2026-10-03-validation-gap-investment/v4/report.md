# 2026-10-03 20:57 KST · 기확인 geometry7쌍 원본 저장픽셀 비교

**정확히 고정된7쌍 모두 원본 PixelData element 바이트와 디코딩한 저장픽셀 배열이 일치했다.** Sagittal3쌍·Coronal3쌍·Axial1쌍이다. v3에서 확인한 과거/현재 캐시 전체행의 일치에 더하여, 두 study경로의 원본 표본에서도 공유 MRI 내용이 확인됐다.

|검사|결과|
|---|---|
|원본 PixelData 바이트|7/7쌍 정확일치|
|디코딩 stored-pixel 배열|7/7쌍 정확일치; 다른원소0개|
|pixel schema와기존geometry|7/7쌍 일치|
|전체DICOM파일 바이트|0/7쌍 일치; 별도study/series/SOP metadata 포함|
|읽은원본/실제decode|고정14개/14회|
|새header검색·후보확장|0회|

## 픽셀의 의미

각 배열은 **320×320 uint16, BitsStored12, unsigned, MONOCHROME2**이며 transfer syntax는 Explicit VR Little Endian(`1.2.840.10008.1.2.1`)이다. 단일frame만 허용했고 `pydicom3.0.2 pixels.pixel_array(ds,raw=True)`를 사용했다. RescaleSlope/Intercept는 없으며, window값은 기록만 했다. rescale·modality/VOI LUT·windowing·반전·색변환·crop·resize·normalization·orientation변환은 적용하지 않았다. 원본 PixelData 바이트 비교는 배열변환과 별도로 수행했다.

따라서 이번 결과는 모델입력224/336uint8 정규화 이후의 동일성과 구분되는 **원본 저장픽셀 표본의 동일성**이다. 원본14개의 full-DICOM SHA256은 각 대응쌍에서 다르지만 PixelData SHA256과 배열 SHA256은 같다. 캐시 생성만으로 생긴 특징 일치라고 설명할 수 없을 만큼, 원본 표본에도 동일성이 존재한다. 다만 7개표본 외 모든slice·모든series까지 같다는 검사는 하지 않았다.

## 계약·검증·비용

- v3에서 geometry가 일치한7쌍만 [새등록](registration.json)에 고정했다. 등록SHA `a922a246053a3016086938cc5f537b2f47d186bf7dd31a39957456e72054d40b`. v2/v3 동결과 v3 중단기록을 보존했다.
- adapter SHA를 확인한 후에만 passive import. existing ZIP index/명시적ZipInfo로14개만읽었다. 중앙디렉터리/전체ZIP스캔·새header탐색 없음. `source.members()`도 호출하지 않았다.
- 모든경고·pydicom로그·stderr를비저장처리하고 예외는 type/phase/순번만기록하게했다. PatientID/Name/BirthDate값이나전체dataset은저장하지 않았다. labels/Gold/Report/prediction 접근·GPU·cache/fold변경 없음.
- 합성uint16데이터에서 raw=True가 slope2/intercept10을적용하지않고 저장픽셀을보존함을 확인. 실행후14경로의등록일치·유일성·7바이트/배열/schema일치·동결source해시를검증했다. [검사](tests-result.txt), [출력검증/해시](verification.json).
- 실제분석 **0.0576초**, 프로세스도구관측0.7651초, peakRSS **53,239,808bytes(50.77MiB)**. 원본2,887,572bytes읽음. 구현·검사·기록은약6분(20:51~20:57), 15분한도이내. [결과](run/result.json), [쌍별영수증 예시](run/pair-01.json).

## 평가에 반영할 범위

fold1↔0의두UID를 **현재모델입력이완전히같고7원본표본도같은교차fold중복쌍**으로표시할수있다. 해당두입력을독립으로간주하면안된다. 이는미래split설계에사용할근거이며이번작업에서기존split·모델·점수는변경하지않았다.

동일환자/전체study동일성을확정하지않는다. v3의PatientIDhash차이는익명화규칙을모르는상태에서환자독립성근거가되지않는다. 이쌍은fold2가아니므로이결과만으로EXP081 F2수치를수정할수없다. 다른환자중복·teacher노출·publicLB과적합·T4/local점수일치는별도문제다.
