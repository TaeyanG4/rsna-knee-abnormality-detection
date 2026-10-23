# 2026-10-04 02:31 KST · local input hash 불일치 단일UID 진단

- 상태: 완료. 담당 Codex `/root/anchor_inputs`. root가 새 실패에 한정해 승인한 CPU 진단; 출력은 이 디렉터리만. [원FAIL](../local-reference-acceptance-v1-failure.json) 보존.
- 범위: 고정 engineering100의 첫 UID1개 cache만 읽어 threads1/12×inference_mode off/on 4조건 비교. producer의 실제 입력 hash와 기존 gate hash를 사용. 모델·GPU·labels/예측배열·원격·100재실행0. Torch CUDA 미초기화 확인.
- 원인: 실제 Torch threads1은 mode와 무관하게 기존 gate hash `83bd512f…`, threads12는 mode와 무관하게 producer hash `b9287d01…`에 exact 일치. interop1·CPU strict 설정. mode 단독 차이는0. windows 단계부터 bitwise 차이가 나므로 CPU bilinear 경로의 thread 조건 불일치가 이 UID 실패를 설명한다. 내부 kernel 분기까지 추적한 것은 아님.

|조건|독립ref 대비 windows max|normalized max|연결|
|---|---:|---:|---|
|threads1 (off/on동일)|1.304290e−7|7.733811e−7|기존 input exact|
|threads12 (off/on동일)|1.374413e−7|7.248802e−7|producer exact|

- threads간 windows max1.788139e−7, normalized max1.192093e−6·916,927원소 차이. 네 조건 모두 기존 windows5e−7/normalized3e−6의 max/p99/mean PASS. 수치 정확성과 동일host 입력 hash 연결은 다른 요구사항이다.
- 결과: [diagnostic.json](diagnostic.json)에 실제threads·mode·hash·원소차이·ref오차·환경·소스SHA, [실행script](diagnose.py). 성공실행2.69초. 최초시도는 Windows 상대경로 key표기 KeyError로 torch import/cache읽기 전 종료; 진단script의 path key만 as_posix로 정정.
- 다음 권고: 새 버전 입력gate에 producer receipt의 실제 selected_threads를 검증·적용·기록하고 동일host exact hash 및 기존 독립ref 허용치를 유지한다. 필요시 strict/backend/mode도 동일하게 고정한다. V1·실패·source·stage·protocol·threshold 변경0. producer를1thread로 재실행하는 대안은 GPU 재실행비용이 있으므로 우선하지 않는다.
- 제한: 이 진단은1UID 원인규명이며 전체100 binding PASS/T4/품질 증거가 아니다. 이후 새버전100 입력검사·acceptance는 root가 별도 결정한다. exact hash 요구를 근사오차로 대체하거나 실패 기록을 PASS로 고치지 않는다. 진행작업/예약 없음.
