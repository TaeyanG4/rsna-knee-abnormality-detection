"""Build docs/learning/colab_knee_ablation.ipynb from colab_knee_baseline.ipynb by swapping in the ablation cells.

Replaces the dataset cell (3), the training cell (5), the evaluation cell (6) and the save cell (7) with
tools/learning/ablation_cells.py CELL_A..CELL_D and rewrites the matching markdown headings.
  python tools/learning/make_colab_ablation.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_cells as C  # noqa: E402

SRC = ROOT / 'docs/learning/colab_knee_baseline_v1.0.1.ipynb'
VERSION = 'v1.2.0'
OUT = ROOT / 'docs/learning' / f'colab_knee_ablation_{VERSION}.ipynb'


def lines(text):
    return text.splitlines(True)


def main():
    nb = json.loads(SRC.read_text(encoding='utf-8'))
    cells = nb['cells']
    code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']
    # code cells in order: 0 nvidia-smi, 1 data, 2 look, 3 dataset, 4 model, 5 train, 6 eval, 7 save
    assert len(code) == 8, len(code)
    for idx, src in ((code[3], C.CELL_A), (code[5], C.CELL_B), (code[6], C.CELL_C), (code[7], C.CELL_D)):
        cells[idx]['source'] = lines(src)
        cells[idx]['outputs'] = []
        cells[idx]['execution_count'] = None
    md = {code[5] - 1: '## 5. 학습 함수 — 실험 하나 = (이름, 증강, seed)\n'
                       '- 학습 설정(epoch, 배치, 학습률, 백본)은 실험 중에 **고정**합니다. 비교할 것 하나만 바꿉니다.\n'
                       '- 결과(검증 예측)는 Drive의 `knee_runs/`에 저장되어, 연결이 끊겨도 끝난 실험은 다시 돌리지 않습니다.\n'
                       '- 이 셀은 함수만 정의합니다. 실제 실행은 맨 아래 7번 셀에서 합니다.\n',
          code[6] - 1: '## 6. 비교 함수 — 정답지 4개로 대조군과 짝 비교\n'
                       '- **흔들림**: 대조군 seed 42와 43의 점수 차이. 같은 설정에서도 이만큼은 우연히 움직입니다.\n'
                       '- **Δ와 구간**: 변형 − 대조군. 구간이 0을 넘고, Δ가 흔들림보다 크고, 네 정답지에서 방향이 같아야 진짜 개선으로 봅니다.\n'
                       '- 합의(consensus) 기준이 가장 깨끗한 자이고, 학습에 쓴 Dread 기준은 Dread에 유리합니다.\n',
          code[7] - 1: '## 7. 실험 실행 — 대조군 2번 + 바꾼 것 하나\n'
                       '- 처음에는 대조군(seed 42, 43)과 좌우 뒤집기(flip)만 돌립니다 (약 1.5~2시간).\n'
                       '- 끝나면 마지막 출력(비교표)을 복사해 대화창에 붙여 주세요. 다음에 무엇을 바꿀지 같이 정합니다.\n'}
    for i, text in md.items():
        assert cells[i]['cell_type'] == 'markdown'
        cells[i]['source'] = lines(text)
    a_md = code[3] - 1
    cells[a_md]['source'] = lines('## 3. 데이터셋과 증강 옵션\n'
                                  '- 모델 입력: 슬라이스 12장 [12, 224, 224], 값 0~1.\n'
                                  '- 증강 종류는 `AUGS`에 있습니다: none, base(대조군), flip, affine, sdrop.\n'
                                  '- 상위권 방법: 증강을 **하나씩** 바꿔 대조군과 비교합니다.\n')
    title = ''.join(cells[0]['source']).replace('· v1.0.1', '· ' + VERSION + ' (증강 비교판)', 1)
    title = title.replace('> 버전 v1.0.1 (2026-10-07)', '> 버전 ' + VERSION + ' (2026-10-08) — v1.1.0 대비: 같은 이름의 결과는 설정(증강·seed·에폭·백본·학습률)까지 같을 때만 재사용, 결과 파일을 원자적으로 저장, seed 수가 대조군과 다른 변형은 비교에서 제외(독립 리뷰 지적 반영). '
                          'v1.1.0 — v1.0.1 대비: 증강 옵션, 학습 함수, 정답지 4개 짝 비교, 실험 계획 셀. 이전 이력: v1.0.1', 1)
    cells[0]['source'] = lines(title)
    cells[2]['source'] = lines(f"print('notebook {VERSION}')\n" + ''.join(cells[2]['source']))
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', OUT, len(cells), 'cells')


if __name__ == '__main__':
    main()
