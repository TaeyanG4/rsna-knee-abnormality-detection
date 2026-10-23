"""Entry point for raptor_corpus.py with the extension contracts registered (raptor_contracts_ext.py).

Runs raptor_corpus.main() unchanged (same builder, same arguments, resume and --verify behaviour); the only difference is that
--contract also accepts raptor80_336_c140_s02. After a build run the manifest gets a "contract_registry" record (module and
entry-point sha256) so a cache built through this path says where its contract came from. raptor_corpus.main() rewrites the
manifest on resume, so the record is re-added after every run.

  python training/src/raptor_corpus_ext.py --contract raptor80_336_c140_s02 --workers 3
  python training/src/raptor_corpus_ext.py --contract raptor80_336_c140_s02 --verify
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import raptor_contracts_ext  # noqa: E402,F401  (registers the extension contracts before argparse reads CONTRACTS)
import raptor_corpus  # noqa: E402
from raptor_common import CACHE, sha256_file  # noqa: E402


def _arg(argv, flag, default=None):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


def annotate(out_dir: Path):
    path = Path(out_dir) / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["contract_registry"] = {
        "module": "training/src/raptor_contracts_ext.py", "module_sha256": sha256_file(Path(raptor_contracts_ext.__file__)),
        "entry_point": "training/src/raptor_corpus_ext.py", "entry_point_sha256": sha256_file(Path(__file__)),
        "builder": "training/src/raptor_corpus.py", "builder_sha256": sha256_file(Path(raptor_corpus.__file__)),
        "raptor_common_sha256": sha256_file(_HERE / "raptor_common.py"),
        "note": "contract added to raptor_common.CONTRACTS at import; raptor_common.py and raptor_corpus.py unchanged"}
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sys.argv = [sys.argv[0], *argv]
    raptor_corpus.main()  # --verify exits inside (SystemExit), so only a build run reaches the annotation
    contract = _arg(argv, "--contract", "raptor64_336_c140_s02")
    out = _arg(argv, "--out")
    annotate(Path(out) if out else CACHE / contract.upper())


if __name__ == "__main__":
    mp.freeze_support()
    main()
