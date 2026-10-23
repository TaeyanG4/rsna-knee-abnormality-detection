# 정확6 B/RN · 비공개 T4 engineering100 준비

02:56 KST 상태: **CPU stage·실제설치SDK PASS, 원격미실행**. 최초local linkage FAIL을 보존하고 root의 별도V2 CPU100(02:52:27–57) exact100/threads12 PASS를 연결했다. 기존producer·parityV1·6checkpoint·90/10계산·허용치는 변경하지 않았다. V2 helper `1d6bc7c2…89e21e`와 protocol `820be3d9…79b5f`를 포함해 `.local/b-rn3f-t4-staging/20261004-v1`을 새로 동결했다. 정확경로·argv·비용은 [execution-plan](execution-plan.json), SHA·합성31검사와 실제패키지 결과는 [verification](verification.json)에 보존했다. 기존검사/추론 재실행0.

## 한 세션의 범위

- 데이터셋 예정ID `taeyangg4/knee-b-rn3f-v1-assets`, notebook 예정ID `taeyangg4/knee-b-rn3f-v1-engineering100`, 둘다private. 기존freeGPU 허용범위 내 실제할당을root가 확인한 뒤 단1회 실행. 제출/공개/유료작업이 아니다.
- Kaggle session timeout1800초, 소유프로세스 workflow cap1700초, producer 내부cap1600초. retry0. 세션할당이2T4라면 한GPU만사용해도 할당기준청구일 수 있어 최대예산0.5–1GPUh는 실할당후 확정한다. mount준비/컨테이너startup은 notebook 타이머와다르고 provider1800s와구분한다.
- 기존B8/RNclip20 F0–2 exact6, `mean_f(.9 rank B_f+.1 rank RN_f)` 전체100 동순위/N, anchor·EXP084 replacement·hidden/full release없음. 기존100→133 확대없음.
- 실제T4 DICOM100 producer→같은host CPU analytical 입력100→local/T4 paired acceptance. 입력100의 독립float64 reference 비교·actual producer normalized hash연결, 6×12 raw확률·전역rank·최종CSV계산이모두필요. host별producer receipt의실제selected_threads를새bridge로재현하며thread protocolSHA·adapterSHA·actualThreads·parent/workerbinding·explicitPASS_INPUT_AND_PRODUCER_LINK_ONLY까지추가검사한다. paired확률평가기는기존acceptanceV1그대로다.

## 패키지와 변조 방지

신규 `tools/b_rn3f_t4_trial.py`는CPU `build`와명시적원격 `workflow`만제공한다. publish/push/submit을호출하지않는다. build는local parentCOMPLETE와workerSHA·raw/schema/rank·input100 linkage·resource실제완료를확인한뒤에만 upload/notebook을생성한다. source/helper/protocol/고정100·원본stage hash를묶고source검사를실행때반복한다. 전체weight/header감사나새모델추론은하지않는다.

거대한가중치를다시압축/복제해보관하지않는다. dataset upload디렉터리는frozen assets manifest목록만같은볼륨hardlink로구성한다(원본에write하지않음). Kaggle CLI의`--dir-mode zip`은업로드때하위디렉터리를정리하는기존방식이다. local reference·input JSON·raw NPZ·small helper만tar+LZMA/base85로notebook에내장하여dataset업로드를반복하지않는다. payload파일목록·원본SHA·압축SHA·원본bytes·LZMAbytes·실제SDK변환후script_body bytes를기록한다. SDK의outputs제거/source배열병합/`json.dumps`규칙에더해 `.venv`의실제 `ApiSaveKernelRequest.to_json()` 전체request를offline직렬화해둘다1,000,000bytes미만을필수로한다. Kaggle client/auth/network는생성하지않는다.

mount탐색은두가지기존Kaggle경로의명시dataset만확인하며competition전체DICOM디렉터리를탐색하지않는다. assets manifest·OpenCVwheel은SHA로식별하며불일치/중복은실패다. notebook metadata는기존D01runtime의고정Docker digest/T4 machine shape를재사용하고인터넷off를유지한다. 실제dataset/kernel revision은root의업로드반환에기록하고SHA와별도로보존한다.

## 종료·수거

`process_resource_receipt.py`가workflow parent+descendants를100ms관측한다. cgroup은실제Linux membership경로의start/end·peak/events/current/max이며reset하지않는다. RSS샘플최대는순간최대의하한/중복shared RAM가능성을그대로둔다. 실패·timeout시관측한PID+creation_time만정리한다.

최종 `PASS_ENGINEERING_NUMERICAL_AND_RESOURCE_ONLY`는resource status/child0/접근실패0/정리오류0/생존0/강제정리없음, 실제producer workerPID관측, cgroup완전성, 원본parent와workerSHA, acceptance explicitPASS, numerical explicitPASS·artifactSHA를모두검사한뒤에만쓴다. **exit0나kernel COMPLETE만으로PASS하지않는다.** 실패해도가능한로그/receipt를모아결과를FAILED로남긴다. kernelsetup이helper실행전실패하면result미생성자체가미완료이며성공으로간주하지않는다.

정확한원격산출물:

- `/kaggle/working/b-rn3f-v1/result.json` — 첫수거대상. status·gate hash·최종evidence.zip SHA/bytes.
- `/kaggle/working/b-rn3f-v1/evidence.zip` — resource receipt/log/samples, input/acceptance/paired, producer parent/worker/raw/CSV/실패로그. environment·wheels·가중치복제는제외.
- 내부 `work/producer/result/members.npz`, `work/input.json`, `work/paired.json`, `work/paired-acceptance.json`, `resource/receipt.json` 경로고정.

root는버전고정 `tools/kaggle_output_file.py`로먼저 `b-rn3f-v1/result.json`만회수하고명시status/hash를검사한뒤 `b-rn3f-v1/evidence.zip`을지정SHA/bytes상한으로회수한다. kernel COMPLETE만보고추가시도하거나자동재제출하지않는다. root가학습/할당/로컬완료/bridge 결과와최종stage를검토한뒤upload→processing완료확인→단1회push한다. 장시간작업heartbeat는실제runID/ETA가생긴root가예약한다.

실제build0.375초·SDK check0.328초, 둘다exit0. 동결manifest SHA `8e40c99aa65594f407db584dbb780470ec78cf87910f99182fb234e887932a55`, notebook SHA `bc9365e8a526f9b69f17ec86f0e93b912bfe0d612fd1992541c7cc9f603febfb`. 자산17개 hardlink inode·크기동일 확인, 합계1,154,271,125bytes/추가weight물리복제0. payload raw426,699/LZMA81,008bytes, 실제SDK request104,834/script104,135bytes(<1M). 실제notebook AST·private/offline/T4/Docker·dataset inventory도 확인했다. 이는실행준비결과이며원격실행PASS가아니다. localcache추론28초관측은원격DICOM decoding시간예측에사용하지않는다.
