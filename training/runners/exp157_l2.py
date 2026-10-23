"""EXP157 process-local wrapper around the SHA-bound EXP149 single-head runner.

It changes only registration lookup and the model's initial tensor source. The
original training loop, optimizer, augmentation, evaluation and outputs execute
unchanged through runpy. Validation is CPU only; CUDA is never initialized by
importing this file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def split_initial_state(argv):
    if list(argv).count('--initial-state') != 1:
        raise ValueError('Exactly one initial-state argument required')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--initial-state', required=True)
    args, rest = parser.parse_known_args(argv)
    if not args.initial_state:
        raise ValueError('Exactly one initial-state argument required')
    return Path(args.initial_state), rest


def main(argv=None):
    initial, original_argv = split_initial_state(sys.argv[1:] if argv is None else argv)
    from training.runners import exp149_label_sources as frozen
    frozen.supply_registered_env()
    from tools import exp157_l2_advance as advance
    from tools import exp149_label_sources as guard
    from tools import exp134_aug1_kfold as aug
    import torch

    p = advance.registration(ROOT)
    if initial.resolve() != (ROOT / advance.INITIAL).resolve() or \
            advance.sha256(initial) != p['exp157']['initial_state_file_sha256']:
        raise ValueError('EXP157 initial-state file differs')

    # The frozen runner imports these names after parsing arguments. Replace
    # them in this process only; no SHA-bound source file is edited.
    guard.validate_arguments = advance.validate_arguments
    sys.path.insert(0, str(ROOT / 'training/src'))
    import raptor_radimagenet as rad
    from raptor_common import build_model

    def from_saved_state(arch, weights_path, expected_sha256):
        if arch != p['common_args']['arch'] or \
                Path(weights_path).resolve() != aug.weights_path(ROOT).resolve() or \
                expected_sha256 != p['init_weights']['sha256']:
            raise ValueError('EXP157 model architecture or RadImageNet input differs')
        state = torch.load(initial, map_location='cpu', weights_only=True)
        with torch.device('meta'):
            model = build_model(arch, pretrained=False)
        model.to_empty(device='cpu')
        if set(state) != set(model.state_dict()):
            raise ValueError('EXP157 initial-state keys differ')
        model.load_state_dict(state, strict=True)
        tensor_sha, _ = aug.state_hashes(model)
        if tensor_sha != advance.INITIAL_TENSOR_SHA:
            raise ValueError('EXP157 initial-state tensor hash differs')
        return model, {'source': 'EXP157 saved seed-42 initial-state file',
                       'initial_state_file_sha256': p['exp157']['initial_state_file_sha256'],
                       'full_model_initial_state_sha256': tensor_sha,
                       'weights_sha256': expected_sha256,
                       'weights_file': Path(weights_path).name,
                       'strict_result': 'all state keys matched'}

    rad.build_radimagenet_model = from_saved_state
    saved_argv = sys.argv
    try:
        sys.argv = [str(ROOT / advance.FROZEN_RUNNER), *original_argv]
        runpy.run_path(str(ROOT / advance.FROZEN_RUNNER), run_name='__main__')
    finally:
        sys.argv = saved_argv


if __name__ == '__main__':
    main()
