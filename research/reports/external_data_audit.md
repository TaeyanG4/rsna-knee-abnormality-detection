# External artifact audit

각 Dataset의 게시자, source URL, dataset ID, 선언 license, access, compatibility 상태는 `external_data_audit.csv`에 기록한다. Source code license와 model weights/data license는 동일하지 않을 수 있다.

## 현재 release hold

최강 공개 ensemble이 연결하는 `marwanmath/resnet-50-radimagenet-marwan`의 내려받은 metadata는 CC-BY-NC-SA4를 표시한다. 이것이 대회에서 반드시 금지되었다고 단정한 것은 아니다. 그러나 사용자 지정 정책에 따라 대회 참가 및 winners의 공개/라이선스 의무와 호환되는 근거를 확인하기 전 해당 encoder를 실행하거나 전체 ensemble을 제출하지 않는다.

RadImageNet 원본 repository의 소프트웨어 MIT LICENSE만으로 변환 checkpoint Dataset의 NC-SA 표시가 자동 해소되었다고 판단하지 않는다. Discussion의 참가자 질문은 host의 공식 허가가 아니다.

## 확보한 CC0 선언 artifacts

Raptor v5/v8/v10 Dataset과 Mattia CoAt residual-gated package는 게시자 metadata에 CC0-1.0을 선언한다. 실제 세 CoAt weight와 9개 dependency source의 SHA256을 manifest와 대조했다. Raptor 세 weight도 실제 SHA를 계산했다.

이 hash 검사는 파일 동일성을 증명할 뿐 모든 upstream 권리, 학습 데이터 적법성 또는 악성 코드 부재의 절대 증명은 아니다. 로컬에서는 외부 checkpoints를 pickle로 실행하지 않고 container opcode를 정적으로 검사했다. Kaggle train-only CoAt branch는 pinned source/weights를 확인한 후만 import/load하도록 구성했다.

## Quarantine

Barun의 other/제한 조건, Jez의 unknown, Zaid의 other license는 무조건 사용 가능으로 처리하지 않는다. Tsuyu balanced-label Dataset은 MRNet/OAI를 포함한다는 설명과 RSNA 전체가 gold라는 주장, 접근/외부 cohort provenance를 추가 검증해야 하므로 quarantine했다. Juan의 label 이름 파일은 실제 HTML/CAPTCHA이므로 invalid artifact다.

## Raw / processed corpus

570 GB급 전체 raw DICOM은 내려받지 않았다. MaxSpan Dense Corpus라는 Dataset 이름은 실제 payload가 checkpoint인 경우가 있다. 별도 `knee-raptor-corpus`의 IDs/masks만 확보했으며 pixel volume은 아직 없다. 그 index는 3,200 study, 44-slice 계약이고 gold는 45개만 포함하므로 current 64-slice model의 cache라고 대체 사용하지 않는다.

대규모 train 영상 검사는 competition이 붙은 private Kaggle Notebook에서 실행했다. Train/test report를 외부 서비스로 보내거나 test report를 생성하지 않았다.
