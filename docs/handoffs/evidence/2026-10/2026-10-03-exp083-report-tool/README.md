# EXP083 고정 평가기 준비

새 EXP083 성능값을 열기 전에 구현한 CPU 평가기다. V1 초안은 `v1-draft.py`에 보존했다. V1 pilot의 native FP16 가정 실패 후 root 지시에 따라 V2 native FP32 schema·실제 manifest hash를 등록 시 결합하도록 전환했다. 평가기의 `--prepare`와 실제 `--run`은 별도다. 준비 과정에서는 예측값·라벨값·Gold·GPU를 열지 않는다. 원 EXP083/V2 구현과 protocol은 수정하지 않는다.

주효과는 treatment−재학습 control의12-target macro AUC다. 같은 cache의 frozen B8 대비 차이는 secondary이다. 과거 저장 B OOF의 동등성은 별도 replay gate가 없으므로 주장하지 않는다. 라벨은 기존 canonical hash와 원본 file hash를 모두 확인하고 float32→numpy.rint 순서를 유지한다. 각12소견 P/N·AUC·차이와 Fracture, scanner_group/field/manufacturer/cache mask별 탐색적 결과를 낸다. 부분 target만 평균하지 않는다.

component500회·seed20261003을 고정한다. 각 component draw의 study multiplicity를 세 arm과12target 전체에 공유한다. 어느 target이라도 weighted 양성/음성이 없으면 macro draw 전체를 무효화한다. 유효500개가 될 때까지 추가 추출하지 않는다. 시도·유효·무효·target별 무효수·동일draw hash·조건부 percentile CI를 기록한다.

실행 전 source/protocol/라벨/분할/component 파일 hash, 정확한 UID/target 순서, 성공 job marker와 runner argv, pilot/root authority, 최초 실패시간 포함2700초, cache FP32/encoder/초기parity/RNG trace, 두arm20epoch·1100update·0skip 및 scale 이력을 검증한다. skip이 하나라도 있으면 채점 전에 중단하며 별도 root 검토가 필요하다. CSV↔NPY float32 값이 동일해야 하며 고정최종 작은 head checkpoint도 CPU에서 검사한다. 모델 실행이나 feature 재생성은 없다.

noGoldfile은 B8·연구자·Gold 노출 계보를 지우지 않는다. 한 seed와 선택된 weak F2의 조건부 개발 진단이며 임상/private/anchor0.943/D01 입증이 아니다. 알려지지 않은 환자/중복, 약라벨 오류, 학습seed/선택 불확실성은 CI에 포함되지 않는다. 자동채택·비율튜닝·후속 확장은 없다.

## 명령

root가 확인한 최종V2 manifest로 첫 hash등록을 실행했고, 명시한 full job tag는 처음부터 `EXP083_SLOT_BIAS_V2_FULL`이었다. 이후 root 요청에 따라 code 기본값도 같은 tag로 수정했다. 이전 등록은 `registration-before-default-fix.json`에 그대로 보존했다. 수정한 최종 source로 아래 첫 명령을 실행해 root가 새 registration.json을 고정한다. 실제 채점은 root의 별도 단계다.

```powershell
$cpuPython = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
& $cpuPython tools/exp083_slot_bias_report.py --prepare --protocol artifacts/experiments/EXP083_SLOT_BIAS_F2_PROTOCOL_V2/manifest.json --protocol-sha256 2ae04b602dd7688c6c06625ca3c9e6e257ffb93113fdf478a0b6200db5c2a55b --job-dir artifacts/logs/EXP083_SLOT_BIAS_V2_FULL
$reportRegistrationSHA = (Get-FileHash docs/handoffs/evidence/2026-10/2026-10-03-exp083-report-tool/registration.json -Algorithm SHA256).Hash.ToLower()
& $cpuPython tools/exp083_slot_bias_report.py --run docs/handoffs/evidence/2026-10/2026-10-03-exp083-report-tool/registration.json --registration-sha256 $reportRegistrationSHA
```

`--run`은 성공한 full과 미존재 `evaluation-v1` 출력이 필요하다. registration hash를 등록 직후 root 기록에 남기고 실제 실행 때 그 hash를 사용한다. registration hash를 시작/종료 모두 확인하며 source binding 누락을 거부한다. 산출물은 analysis-plan.json, report.json, SUCCESS.json이다. 실패한 평가 출력은 남고 자동재실행하지 않는다. 준비 중에는 채점 명령을 실행하지 않았다.

합성검사·독립 정적검토와 실제 실행 여부는 `verification.json`, 인계는 `HANDOFF.md`에 기록한다.
