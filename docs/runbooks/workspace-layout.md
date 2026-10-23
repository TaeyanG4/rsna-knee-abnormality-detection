# 작업 공간 경로와 복구

2026-10-01에 경로 의존성을 분리하고 layout v2로 이동했다. 설정의 기준은 config/workspace.toml, PC별 덮어쓰기는 config/local.toml, root 이동은 KNEE_WORKSPACE_ROOT 환경 변수다.

| 현재 폴더 | 내용 |
|---|---|
| training/src | 기존 학습·분석 Python 코드 |
| training/runners | 기존 PowerShell 실행 스크립트 |
| storage/competition | 공식 CSV와 원본 데이터 |
| storage/cache | derived·dicom·pixels cache |
| storage/labels, gold, external, weights | 라벨·평가·외부 자료·모델 |
| artifacts/experiments, oof, predictions, logs | 기존 실험 결과와 로그 |
| research/notebooks, reports, metadata, snapshots | 공개 연구와 조사 기록 |
| kaggle/assets, packages, submissions | Kaggle 배포 입력·코드·출력 |
| knee_workspace | 경로 해석 공통 모듈 |
| monitoring | 대시보드 구현 위치 |
| .local | 로컬 DB·백업·마이그레이션 기록 |

Python 코드는 workspace_root()/workspace_path()로 위치를 찾는다. 예를 들어 `workspace_root() / 'experiments' / 'EXP056_SPECIALIST_POOLED_4FOLD'`는 artifacts/experiments 아래로 해석된다. `workspace://data/train.csv`도 같은 계약이다. 과거 manifest의 H: 절대 경로는 legacy_roots에 일치할 때 현재 root로 옮겨 해석한다. 원본 기록의 숫자와 경로 문자열을 다시 쓰지 않았다.

현재 작동하는 경로 조회 명령:

```powershell
python tools/workspace_path.py data/train.csv
python tools/workspace_path.py experiments/EXP056_SPECIALIST_POOLED_4FOLD/summary.json
```

다른 디렉터리에서 직접 실행하는 training script도 bootstrap이 workspace config를 찾는다. `Path(args.out)` 같은 cwd 의존 입력도 resolver로 바꿨다. PowerShell runner는 Workspace.ps1과 같은 resolver를 사용한다. GPU 학습 전체는 이번 변경 검증을 위해 재실행하지 않았다.

마이그레이션 근거는 `.local/migrations/layout-v2/`에 있다. `before-refactor.zip`은 변경 전 코드와 config, `refactor.json`은 코드 변경 hash, `moves.json`은 23개 폴더 이동과 직접 자식 목록/작은 파일 hash다. 수백 GB의 모든 binary를 새로 해시하지 않았으며 같은 볼륨의 디렉터리 rename으로 이동했다.

복구가 필요하면 실행 중인 관련 작업을 중단한 뒤 `python tools/migrate_layout.py --rollback`으로 디렉터리와 경로 config를 되돌릴 수 있다. 이동 후 같은 예전 경로를 새로 만들었다면 충돌을 감지하고 멈춘다. 코드 원복은 zip의 원본과 현재 변경을 비교한 뒤 선택적으로 수행한다. 복구는 지금 실행하지 않았다.

마이그레이션 당시 검사: 경로·기존 라벨·제출 검증 15개 통과, Python compileall 통과, PowerShell parser 오류 0. 원본 공개 notebook과 과거 산출물은 내용 변경 없이 보존했다.
