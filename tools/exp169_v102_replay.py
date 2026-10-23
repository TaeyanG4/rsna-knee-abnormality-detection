"""Reproduce the qualified EXP169 v1.0.2 notebook, without running it.

This is a byte-level generator regression, not a runtime qualification. The
original generator and source notebook remain frozen. No target notebook,
model weights, archive contents, labels, or external services are read here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


GENERATOR_PATH = "tools/exp169_partner_bundle_release.py"
GENERATOR_SHA256 = "c48aac04128f524116097077a7f1ffd1365f421e3af123c91bb228da09d51665"
TRIO_STAGE_PATH = "artifacts/releases/EXP155_S42v1_1_stage_v12"
SOURCE_NOTEBOOK_PATH = TRIO_STAGE_PATH + "/blend3/release.ipynb"
SOURCE_NOTEBOOK_SHA256 = "b364c14e72c6bb241c044e7a6770875bd3757cae47d347d39c2daf8cdfb8ea7b"
BASELINE_STAGE_PATH = "artifacts/releases/EXP169_PARTNER_BUNDLE_A1_stage"
BUILD_RECEIPT_SHA256 = "ff164789e69fabfbe620980569d132ce00c5c6b401e658fb9c621d01e507cb48"
DATASET = "taeyangg4/knee-exp169-partner-bundle-assets"
ARCHIVE_SHA256 = "8a96ac91534144dd68444fc57d308b8c94a4d902fa439aa288992bf25d3dbb95"
MANIFEST_SHA256 = "dea8d3d636884cdc9be5763aa39c81ef470c7d474d80bdcce01e3ed81aa7f433"
TRIO_MANIFEST_SHA256 = "f3ab31e13e489fb8a6abf439f13799b8a0ce39e25c933b3fc0320fdee05e5619"
ORIGINAL_NOTEBOOK_SHA256 = "e255df5e9ab060064ae0455e17a32fb672249450988523353d69a7fdf9877f67"
QUALIFIED_NOTEBOOK_SHA256 = "e96a6c854751efd1684c25553a61ed49a1c70217a814fb0df3c25dddc8d39f75"

# Independently hash-checked against the frozen stage's source_code inventory.
FROZEN_CODE_SHA256 = {
    "exp169_partner_bundle_release.py": GENERATOR_SHA256,
    "raptor_common.py": "bc6db5b3e8cd13b52fbff8c039d3674cdf573f86c1959e22fe9b2fd94255d7dc",
    "raptor_dicom.py": "497a41f85a7b70cba8029c0f726ed2133653a2ce9673cd0bf8de42a85ac5634b",
    "raptor_runtime.py": "afe394b631c681e3826ad976e71c793214dee81c49c0d9cd76046590cc4513ad",
    "raptor_radimagenet.py": "8b368bcfc104f26615934d525dbf37786d1f95c1532601ce8cf4701faa3b6112",
    "raptor_data.py": "0b922ea0359962a2bb6531d5630639f2c9002abdb80fe1353991087d6bcdea01",
    "exp160_base_cpu_pos.py": "c42688595071a46355fd64ccecdb7775bb1ab7586ed271dbaa3b4dbe24c122c5",
    "exp160_base_patch_mil.py": "ede8fcf968d4a3ccc1c681abec0bc378299b0565ddf25c17899fbdb8804ceb58",
}

_CUDNN_SHIM = """    # B v1.0.1 shim: public-serving.py (trio) sets cudnn.benchmark=True globally; under it the RadR50 L2 convs
    # picked different algorithms in pilot and full (max 1.36e-4 > registered 2e-5). Fix the partner path only.
    _torch.backends.cudnn.benchmark = False
    _torch.backends.cudnn.deterministic = True
    print('EXP169 release v1.0.2: partner cudnn.benchmark', _torch.backends.cudnn.benchmark, flush=True)
"""
_SERVE = """    _e169.serve(_partner, _work, Path('/kaggle/working'),
                Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'))
"""
_RETURNED_FALLBACK_GUARD = """    # B v1.0.2: fail loudly for the public reading \u2014 a trio-only fallback would score like blend3 and hide the failure.
    _r = _e169.read_json(Path('/kaggle/working/exp169-executed-candidate.json'))
    if _r.get('status') != 'PASS_BUNDLE':
        raise RuntimeError('EXP169 bundle did not pass: ' + str(_r.get('stages', {}).get('partner')))
"""
_ORIGINAL_FAILURE = """except BaseException as _bundle_error:
    import shutil as _shutil, json as _json, hashlib as _hashlib
    _fallback = Path('/kaggle/working/submission.csv')
    if _fallback.exists(): _fallback.unlink()
    _shutil.copyfile(_work/'submission.csv', _fallback)
    def _digest(p):
        with p.open('rb') as f: return _hashlib.file_digest(f, 'sha256').hexdigest()
    (_fallback.parent/'exp169-executed-candidate.json').write_text(_json.dumps({
        'schema': 'exp169-executed-candidate-v1', 'status': 'FAILED_BUNDLE',
        'candidate': 'TRIO_ONLY_FALLBACK', 'executed_arms': ['EXP155_BLEND3'],
        'reason': type(_bundle_error).__name__ + ': ' + str(_bundle_error),
        'trio_sha256': _digest(_work/'submission.csv'), 'submission_sha256': _digest(_fallback)
    }, indent=2) + '\\n')
    print('EXP169 FAILED_BUNDLE:', type(_bundle_error).__name__, str(_bundle_error), flush=True)
"""
_QUALIFIED_FAILURE = """except BaseException as _bundle_error:
    # B v1.0.2: no trio-only fallback CSV; remove any root CSV so the run fails visibly.
    _fallback = Path('/kaggle/working/submission.csv')
    if _fallback.exists(): _fallback.unlink()
    print('EXP169 FAILED_BUNDLE (no submission written):', type(_bundle_error).__name__, str(_bundle_error), flush=True)
    raise
"""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked_file(path: Path, expected: str) -> None:
    if _digest(path.read_bytes()) != expected:
        raise ValueError("Frozen v1.0.2 dependency changed: " + str(path))


def _replace_once(source: str, before: str, after: str, label: str) -> str:
    if source.count(before) != 1:
        raise ValueError("Expected exactly one v1.0.2 transform anchor: " + label)
    return source.replace(before, after, 1)


def regenerate(root: str | Path) -> bytes:
    """Generate the exact qualified notebook bytes without reading its target.

    Archive and manifest digests are the frozen generator arguments. This
    helper does not revalidate their on-disk contents; a staging caller must
    separately check the artifacts it uses. No notebook is written or run.
    """
    root = Path(root).resolve()
    generator = root / GENERATOR_PATH
    _checked_file(generator, GENERATOR_SHA256)
    _checked_file(root / SOURCE_NOTEBOOK_PATH, SOURCE_NOTEBOOK_SHA256)

    # Lazy import: baseline notebook generation itself imports only stdlib.
    from tools import exp169_partner_bundle_release as baseline

    if Path(baseline.__file__).resolve() != generator.resolve():
        raise ValueError("Imported baseline generator is outside the requested root")
    notebook = baseline.notebook(root / TRIO_STAGE_PATH, DATASET,
                                 ARCHIVE_SHA256, MANIFEST_SHA256)
    original_bytes = (json.dumps(notebook, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if _digest(original_bytes) != ORIGINAL_NOTEBOOK_SHA256:
        raise ValueError("Original staged notebook generation changed")

    source = notebook["cells"][-1]["source"]
    source = _replace_once(source, "    import torch as _torch\n",
                           "    import torch as _torch\n" + _CUDNN_SHIM, "cuDNN reset")
    source = _replace_once(source, _SERVE, _SERVE + _RETURNED_FALLBACK_GUARD,
                           "returned internal fallback")
    source = _replace_once(source, _ORIGINAL_FAILURE, _QUALIFIED_FAILURE,
                           "outer fail-loud handler")
    compile(source, "EXP169 qualified v1.0.2 replay", "exec")
    notebook["cells"][-1]["source"] = source.splitlines(keepends=True) + [""]

    markdown = notebook["cells"][0]["source"]
    markdown = _replace_once(markdown, "# EXP169 fixed", "# EXP169 v1.0.2 fixed",
                             "release heading")
    markdown = _replace_once(
        markdown, "Any bundle failure is recorded as FAILED_BUNDLE.",
        "Any bundle failure stops the run without a submission file "
        "(v1.0.2: fail loudly; no trio-only fallback).", "failure description")
    notebook["cells"][0]["source"] = [markdown]

    # The qualified Windows edit used indent=1, literal Unicode, CRLF JSON
    # whitespace, and no trailing newline. Embedded source newlines remain
    # JSON-escaped LF. The code source list ends with an empty string; the
    # markdown source is a one-item list. The other three cells retain strings.
    result = json.dumps(notebook, indent=1, ensure_ascii=False,
                        allow_nan=False).replace("\n", "\r\n").encode("utf-8")
    if _digest(result) != QUALIFIED_NOTEBOOK_SHA256:
        raise ValueError("Qualified v1.0.2 replay bytes changed")
    return result
