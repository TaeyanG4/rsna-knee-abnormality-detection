# EXP086 V2 CPU 인계

10-04 05:57 KST, dicom_header_io. 중앙문서root담당·GPU0.

## 완료

- 별도 `tools/exp086_dinov2s_v2.py`/runner/후보instance한정adapter. V1source/protocol/FAILED불변, 원4학습source·asset·입력/라벨/분할·loss/optimizer/schedule유지.
- canonical518 `[1,1370,384]` trainable 위치표→CPU FP32 timm bicubicAA→원device/원position dtype복귀. 내부1thread·strict/deterministic=True·warn_only금지. detach/cache/globalpatch없음.
- 초기state `688756a1dff13683a5d55a60dd667e0bbd88b17df324d2598c5baf3122c41ad5`,22,163,224params 동일. 신규15tests PASS2.47초.
- 바뀐연산+synthetic모델1회연결0.969초 PASS: CPU보간값/gradient exact·위치gradient finite/nonzero. 원fullprobe재실행안함, optimizer0/GPU0.
- protocol `workspace://artifacts/experiments/EXP086_DINOV2S_G1_V2_PROTOCOL/manifest.json`, SHA `0383b55ba820c19f262fd36e4cc27bf604d2ae109302d7109f47e5423390cbc3`.
- [검증](verification.json)·[정확한argv](execution-plan.json)·[명령/제한](commands.md)·[CPU근거](cpu-changed-position-probe.json).

## 재개/미완료

anchor_inputs 독립producer/평가기검토후 root V2pilot결정. 새결정예정 `v2/root-pilot-decision.json`, 이후 `root-full-decision.json`. producer작성/실행없음.
V2pilot32Tupdate180초, full8epoch3448update5300초. 원V1실제83.23500000000058초+V2실제pilot+full<=5580초. 비용식 `1.3*(steady*3448)+300<=5300`; retry0. 정확한phase/workerargv·7output·policy는plan/프로토콜참조.
root가active jobs확인·실행결정·완료회수·ETA heartbeat 담당. 같은실패/v1재시도아님. full은성공pilot/실제receiptSHA에묶은별도결정필요.
CPU는실제CUDA↔CPU copy backward미검증. 이를V2pilot에서strictFP16/no-skip/실제시간과확인한다. 위치표shape호환만으로serving동일성아님: 향후factory에같은adapter필요·T4parity별도. 같은노출weakV의임상/private/D01개선주장불가.
