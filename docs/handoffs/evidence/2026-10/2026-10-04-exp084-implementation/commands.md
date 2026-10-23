# EXP084 CPU 구현 명령과 범위

2026-10-04 KST. root가 RN clip20/RN ones20/B8의 새 group 대조를 선택했다. 이 기록의 작업자는 GPU를 시작하지 않았다. 원본4 training source·기존 protocol·중앙 문서는 수정하지 않았다.

## 실행한 검사

PowerShell 작업별 환경은 `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=''`였다. 모델 초기화 확인에는 `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`도 설정했다. 이 CPU 셸 환경을 GPU queue에 그대로 계승하면 안 된다.

```powershell
python -X utf8 -m pytest tests/test_exp084_group_loss.py -q -p no:cacheprovider --basetemp=.local/pytest-exp084-contract-v4
python -X utf8 tools/exp084_group_loss.py --help
```

9 tests PASS. 첫7개는 split/component·T-only BCE weight·optimizer-loop AST 동일성·Gold 및 점수 접근 제거·초기state/runtime gate·완료계약을 확인했다. 추가2개는 기존 output의 재시도 거부와 synthetic timeout에서 소유 child PID만 종료하는 것을 확인했다. 순차수정 후 관련검사를 재실행했으며 GPU/실제 fit을 대신한 검사는 아니다.

runner는 다음의 수동 source renderer 호출로 저장했다. acquisition/training script를 import/eval/exec하지 않는다.

```python
from pathlib import Path
from tools.exp084_group_loss import render_training_source, ENTRYPOINT
root = Path.cwd()
source = (root / 'training/src/raptor_train.py').read_text(encoding='utf-8')
(root / ENTRYPOINT).write_text(render_training_source(source), encoding='utf-8')
```

`ast.parse`와 `Get-FileHash`/Python SHA256으로 신source3개의 구문과 원본4 source 불변을 확인했다. `git diff --stat`은 이 경로가 Git 저장소가 아니어서 사용할 수 없었다. 따라서 변경 증거는 새 파일·frozen source hashes·등록시 생성되는 `trainer.diff`다.

## CPU fresh 초기값 확인

표준 import `torch/timm/pydicom/cv2` 후 side-effect가 확인된 `raptor_common.build_model`만 사용했다. 각 구조마다 Python/NumPy/Torch seed42를 다시 고정하고 `build_model(arch, pretrained=True).to('cpu')`를 수행했다. 각 state의 key 문자열과 CPU contiguous tensor bytes를 key 정렬 순서로 SHA256했다. 로컬 cache 파일 SHA를 먼저 검증했으며 network·pixel·Gold·추론·학습은 없었다.

- RN file `829a220f…` / fresh full-state `093be508…` PASS.
- B file `8558fd38…` / fresh full-state `e4ea3ae6…` PASS.
- `torch.cuda.is_initialized()` false.
- 구체 local snapshot revision·filename·bytes·full SHA·관측시각은 [initial-assets.json](initial-assets.json). revision은 로컬HF cache metadata 근거이며 원격재확인으로 표시하지 않는다.

## 단일 등록과 후속

평가 도구의 source 동결 통보 후 아래 prepare를 한 번 실행했다. 전체 작업의 split은 `RN_GROUP_SCREEN_V1:`+component ID SHA256 big-endian modulo5==2 규칙이다. 기존 grouped eligible4343에만 적용하고 labels/old scores로 salt를 다시 고르지 않았다. 결과 T3455/V888, train component3323/val873,12소견 양음성 존재다.

```powershell
python -X utf8 tools/exp084_group_loss.py prepare --asset-receipt docs/handoffs/evidence/2026-10/2026-10-04-exp084-implementation/initial-assets.json
```

실제 등록 결과와 source/manifest SHA·배정 크기·CPU launch gate 검사는 [CPUverification.json](CPUverification.json)에 기록했다. registered data/source bindings20개+원4source 불변·세arm의 argparse Namespace→validate_arguments gate가PASS다. 새 split의 허용된 weak reference P/N 외에 성능 점수를 계산하지 않았다. GPU queue 실행과 evaluator의 score 개봉은 root가 CPU 검토 후 수행한다. run supervisor는 RN 각1800초/B3600초 및 전체7200초, output/budget 재사용 방지, retry0을 강제한다. 완료 PASS라도 skip<1%가0 skip이라는 뜻은 아니므로 실제 updates/skips를 평가에 보존한다.
