"""v1.1.0: fixed project-local scratch registration; canonical experiments immutable.

Only destination globals/default paths are routed. The original registration,
validation, scientific checks and CPU CLI execute unmodified. No training entry.
"""
from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp190_coatnet_oai_f0 as arm

CANONICAL = arm.EXP


def canonical_hashes():
    return {str(p.relative_to(ROOT)): arm.HASH(p) for p in CANONICAL.rglob('*') if p.is_file()}


@contextmanager
def route(destination):
    destination = Path(destination).resolve()
    destination.relative_to((ROOT / '.local/scratch/astra-2').resolve())
    reg = destination / 'protocol-v1'
    paths = dict(EXP=destination, PROTOCOL=destination/'PROTOCOL.md', REG=reg,
                 MANIFEST=reg/'manifest-OAI.json', VALIDATION=reg/'validation.json',
                 LABELS=reg/'oai-masked.csv', LABEL_PROVENANCE=reg/'oai-masked.provenance.json',
                 LABEL_QA=reg/'oai-masked.qa.json', RUN_REL=(destination/'OAI').relative_to(ROOT).as_posix())
    originals = {k: getattr(arm, k) for k in paths}
    defaults = arm.load_registration.__defaults__
    validation_defaults = arm.require_validation.__defaults__
    before = canonical_hashes()
    try:
        for name, value in paths.items():
            setattr(arm, name, value)
        arm.load_registration.__defaults__ = (paths['MANIFEST'], *defaults[1:])
        arm.require_validation.__defaults__ = (paths['MANIFEST'],)
        yield arm
    finally:
        arm.load_registration.__defaults__ = defaults
        arm.require_validation.__defaults__ = validation_defaults
        for name, value in originals.items():
            setattr(arm, name, value)
        assert canonical_hashes() == before, 'Canonical OAI artifacts changed'


def parity_and_canonical_rejection(destination):
    """Compare privately, without printing labels or changing any digest checks."""
    old = json.loads((CANONICAL/'protocol-v1/manifest-OAI.json').read_text(encoding='utf-8'))
    with route(destination):
        current = arm.load_registration()
    allowed = {'created', 'run_dirs', 'source_bindings', 'content_sha256'}
    assert set(old) == set(current)
    assert all(old[k] == current[k] for k in old.keys() - allowed), 'Scientific registration parity differs'
    drift = {name for name, digest in old['source_bindings'].items()
             if digest != arm.source_bindings().get(name)}
    assert drift <= {'tools/gpu_job.py', 'tools/gpu_queue.py'}, 'Unexpected historical binding drift'
    if drift:
        try:
            arm.load_registration()
        except ValueError as exc:
            assert str(exc) == 'EXP190 control/recipe/source binding changed'
        else:
            raise AssertionError('Stale canonical registration accepted')
    else:
        arm.load_registration()  # permits a separately authorized later canonical repair
    return current


@lru_cache(maxsize=1)
def private_registration():
    destination = Path(os.environ['OAI_REGISTRATION_OUT'])
    destination.resolve().relative_to((ROOT/'.local/scratch/astra-2').resolve())
    destination.mkdir()
    (destination/'PROTOCOL.md').write_bytes((CANONICAL/'PROTOCOL.md').read_bytes())
    with route(destination):
        arm.main(['register'])
    parity_and_canonical_rejection(destination)
    return destination


def registration():
    destination = private_registration()
    with route(destination):
        return arm.load_registration()


def main():
    destination, *argv = sys.argv[1:]
    if not argv or argv[0] not in {'register', 'validate', 'admit', 'smoke'}:
        raise ValueError('Test fixture allows only CPU register/validate/admit/smoke')
    if argv[0] == 'smoke' and '--device' in argv and argv[argv.index('--device')+1] != 'cpu':
        raise ValueError('CPU fixture only')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with route(destination):
        if argv[0] == 'register':
            arm.EXP.mkdir(parents=True, exist_ok=True)
            with arm.PROTOCOL.open('xb') as stream:
                stream.write((CANONICAL/'PROTOCOL.md').read_bytes())
        arm.main(argv)
    parity_and_canonical_rejection(destination)


if __name__ == '__main__':
    main()
