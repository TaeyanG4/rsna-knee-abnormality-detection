"""Hash pipeline stages so two environments can prove they saw identical bytes.

    python hash_stage.py make cache_dir/ submission.csv --out manifest_local.json
    python hash_stage.py make cache_dir/ --arrays --out manifest_t4.json   # hash array content, not file bytes
    python hash_stage.py diff manifest_local.json manifest_t4.json

--arrays hashes each .npy/.npz member as (dtype, shape, raw bytes), so files that differ only in
zip metadata or compression still match. Exit code of `diff`: 0 identical, 1 different.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(arr):
    import numpy as np
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256(json.dumps({"dtype": arr.dtype.str, "shape": list(arr.shape)}).encode() + b"\n")
    h.update(arr.tobytes())
    return h.hexdigest()


def entries(paths, arrays):
    """Keys are relative to each given directory (or the bare file name), so machines with
    different absolute paths produce comparable manifests."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += [(x, x.relative_to(p).as_posix()) for x in sorted(p.rglob("*")) if x.is_file()]
        else:
            files.append((p, p.name))
    out = {}
    for f, key in files:
        if key in out:
            raise ValueError(f"duplicate manifest key {key}; hash these inputs separately")
        if arrays and f.suffix in {".npy", ".npz"}:
            import numpy as np
            if f.suffix == ".npy":
                out[key] = array_sha(np.load(f, allow_pickle=False))
            else:
                with np.load(f, allow_pickle=False) as z:
                    for k in sorted(z.files):
                        out[f"{key}::{k}"] = array_sha(z[k])
        else:
            out[key] = file_sha(f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("paths", nargs="+"); m.add_argument("--arrays", action="store_true")
    m.add_argument("--out")
    d = sub.add_parser("diff"); d.add_argument("a"); d.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "make":
        man = entries(args.paths, args.arrays)
        text = json.dumps({"mode": "arrays" if args.arrays else "bytes", "entries": man}, indent=2)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"{len(man)} entries" + (f" -> {args.out}" if args.out else ""))
        if not args.out:
            print(text)
        return
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.a, args.b))
    if a.get("mode") != b.get("mode"):
        print(f"WARNING: mode differs ({a.get('mode')} vs {b.get('mode')})")
    ea, eb = a["entries"], b["entries"]
    only_a, only_b = sorted(set(ea) - set(eb)), sorted(set(eb) - set(ea))
    diff = sorted(k for k in set(ea) & set(eb) if ea[k] != eb[k])
    same = len(set(ea) & set(eb)) - len(diff)
    print(f"identical={same} different={len(diff)} only_a={len(only_a)} only_b={len(only_b)}")
    for k in diff[:20]:
        print("  DIFF", k)
    for k in only_a[:10]:
        print("  ONLY_A", k)
    for k in only_b[:10]:
        print("  ONLY_B", k)
    sys.exit(1 if diff or only_a or only_b else 0)


if __name__ == "__main__":
    main()
