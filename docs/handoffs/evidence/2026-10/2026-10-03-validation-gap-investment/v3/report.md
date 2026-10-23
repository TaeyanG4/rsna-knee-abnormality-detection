# 2026-10-03 20:49 KST · 정확 특징 2건의 캐시·원본 연결 조사

## 확인된 범위

두 UID는 **현재 모델 입력에서 교차 fold 완전 중복**이다. 과거6x9x224 캐시 전체 두 행, 현재64x336x336 Raptor 캐시 전체 두 행, 각각의 마스크가 byte-for-byte 같다. 이는 캐시 입력 중복 확인이며 환자 동일성이나 원본 전체 영상 동일성의 확정이 아니다.

|대상|A: UID끝997255…76711|B: UID끝999266…13209|결과|
|---|---:|---:|---|
|fold|1|0|서로 다름|
|과거 원본 행|4394|4404|서로 다름|
|과거 shard7내 행|538|548|전체2,709,504 uint8 각각 동일|
|현재 원본 행|4397|4407|서로 다름|
|현재 shard/내 행|10/397|11/7|전체7,225,344 uint8 각각 동일|

과거 캐시 행 SHA256는 `9bf84ec7798429d13305306ee8c21bcbe11262f2d7c5ff3b47cff26fffc9b36e`, 현재는 `0df659941055357c334a2de20686af336b2d7d54208959d8c64e6dfa67338e7f`이다. [바이트 비교](run/cache-comparison.json), [등록본](registration.json), [검증](verification.json)에 근거를 보존했다.

현재 생성코드 text review에서 EXP019의 study별 새배열·반환idx쓰기, EXP022 반개구간 shard복사, EXP024 idx-a 및 batch_keys 쓰기에 두 행 재사용 오류는 발견하지 못했다. 두 행은 같은 shard의 경계가 아니며 서로 다른 flush 구간이다. 다만 당시 실행 소스/가중치 계보가 완전히 묶여 있지 않아 현재 코드 검토만으로 과거 버그를 배제할 수는 없다.

## 원본 확인은 표본 매핑 모호성에서 중단

사전 등록된 후보는 fluid sagittal/coronal/axial 각3쌍이다. A의 lexical path20/50/80%를 고정하고, B에서 같은 파일크기·가장 가까운 ZIP 압축크기로 후보를 정했다. **이 순위는 물리적 slice 위치를 뜻하지 않는다.** decode 이전에 InstanceNumber·IOP·IPP·PixelSpacing·shape가 모두 같아야 한다는 gate를 뒀다.

첫7쌍은 모두 geometry와 InstanceNumber가 정확히 같았다(Sagittal3, Coronal3, Axial1). 8번째 Axial 후보는 InstanceNumber18↔12, IPP가 달랐다. 따라서 등록된 stop 조건에 따라 교체·재검색 없이 중단했다. 이는 **서로 다른 slice를 대응시킨 표본 선택 실패**이며 영상 내용 불일치의 증거가 아니다. 9번째는 읽지 않았고 원본 pixel decode는0회다.

읽은16개 헤더의 Study/Series UID는 파일 경로와 모두 맞았다. PatientID는 양쪽에 존재하고 salted hash가 달랐다. PatientName·BirthDate는 비어 있었다. 익명화 규칙을 확인하지 않았으므로 PatientID 차이로 서로 다른 환자를 판단하면 안 된다. 원문 식별값은 저장·출력하지 않았다.

## 비용·보존·남은 일

- 분석0.1097초, 프로세스관측0.8333초, peak RSS78,995,456bytes(75.34MiB). 구현·검토·기록 약10분(20:39~20:49).
- header parse16/40, pixel decode0/18, pydicom에 반환한 header bytes23,836. 이는 실제 디스크/압축해제 byte수가 아니다. ZIP index mode=ro; 중앙디렉터리/전체ZIP 스캔·cache 재작성·labels/Gold/Report/prediction 접근 없음.
- v2와 v3 등록·소스·실행 결과는 보존. v3 등록 SHA256 `e8080ed16352e8349b450b90159ca6abccf533b4c0f3c18253f74cc707aba196`.
- 최소 다음 단계는 **이미 geometry가 일치한7쌍만 새 등록본에 고정하고14개 원본을 decode**하는 것이다. 후보 확장이나 새로운 header 탐색이 필요 없다. 픽셀일치는 이7표본 범위의 공유MRI내용을 확인할 수 있지만 전체원본·환자동일성까지 증명하지 않는다. 이번 작업에서는 진행하지 않았다.
- 후속 실행 전에 warning/exception 원문 식별값 출력 방지와 adapter hash검증 후 import를 적용해야 한다. 현재 선택된6개series의 plane/fluid1은 별도metadata검사로 확인했지만 v3코드에는 해당assert가 없어 개선사항으로 남겼다. 분석 중 민감값 경고/오류 출력은 관측되지 않았다.

현재 두 cached model input이 교차fold에서 동일하다는 사실만으로도 해당쌍을 독립 입력으로 볼 수 없다. fold0/1 품질 해석에 이를 표시해야 하며, 자동 사례삭제나 기존fold 변경은 하지 않았다. 두 건은 fold2가 아니므로 이 결과만으로 EXP081 F2 수치를 수정할 근거도 없다.
