"""v1.2.0: bounded CPU CLI smoke for strict_variant_gate; no release execution."""
from __future__ import annotations
import ast
import base64
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import strict_variant_gate as g

SCRATCH = ROOT / ".local/scratch/astra-b5/strict-variant-gate-v120"
BASE = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0"


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else g.encoded(data))


def seal(stage):
    names = ["kernel/release.ipynb", "kernel/kernel-metadata.json", "registration.json"]
    put(stage / "SHA256SUMS", "".join(g.sha((stage / n).read_bytes()) + "  " + n + "\n" for n in names).encode())


def build_fixture(stage, rename=False):
    nb = g.parse((BASE / "kernel/release.ipynb").read_bytes())
    meta = g.parse((BASE / "kernel/kernel-metadata.json").read_bytes())
    slug = "strict-gate-smoke-v110" + ("-renamed" if rename else "")
    meta.update(id="synthetic/" + slug, title=slug)
    if rename:
        source = "".join(nb["cells"][-1]["source"])
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("BLOB", "PINS")}
        original = base64.b64decode(values["BLOB"])
        stream = io.BytesIO()
        pins = dict(values["PINS"])
        pins["RENAMED_runtime.py"] = pins.pop("exp169_release_c85r7codec.py")
        with zipfile.ZipFile(io.BytesIO(original)) as old, zipfile.ZipFile(stream, "w") as new:
            for entry in old.infolist():
                data = old.read(entry)
                if entry.filename == "exp169_release_c85r7codec.py":
                    entry.filename = "RENAMED_runtime.py"
                new.writestr(entry, data)
        blob = stream.getvalue()
        source = source.replace(repr(values["BLOB"]), repr(base64.b64encode(blob).decode())).replace(repr(values["PINS"]), repr(pins)).replace(g.sha(original), g.sha(blob))
        nb["cells"][-1]["source"] = source.splitlines(True)
    put(stage / "kernel/release.ipynb", (json.dumps(nb, indent=2) + "\n").encode() if rename else (BASE / "kernel/release.ipynb").read_bytes())
    put(stage / "kernel/kernel-metadata.json", (json.dumps(meta, indent=2) + "\n").encode())
    put(stage / "registration.json", {"kernel_id": meta["id"], "test_fixture_only": True})
    seal(stage)


def main():
    scratch = SCRATCH / "c85-cli"
    runs = []
    for version in g.HISTORICAL:
        short = version.split("-", 1)[1]
        output = scratch / "bind-cli" / (version + "-gate-v110")
        review = ROOT / (g.PRESUB + short + "-review-out.md")
        bind_cmd = [sys.executable, "tools/strict_variant_gate.py", "bind", "--stage", str(BASE.parent / version),
                    "--parent", str(BASE), "--output", str(output), "--builder-receipt",
                    str(ROOT / (g.PRESUB + short + "-build-out.md.receipt.json")), "--review", str(review), "--fast"]
        # Bind is create-only. Existing fixed copies are revalidated, never patched.
        if not output.exists():
            result = subprocess.run(bind_cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            put(scratch / (short + "-bind.log"), (result.stdout + result.stderr).encode())
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        commands = [(short, output, None)]
        for case in ("cell0", "title"):
            stage = scratch / (short + "-" + case)
            for name in ("kernel/release.ipynb", "kernel/kernel-metadata.json", "registration.json", "SHA256SUMS", "allowlist.json"):
                put(stage / name, (output / name).read_bytes())
            if case == "cell0":
                nb = g.parse((stage / "kernel/release.ipynb").read_bytes())
                nb["cells"][0]["source"].append("FAST = True\n")
                put(stage / "kernel/release.ipynb", nb)
            else:
                meta = g.parse((stage / "kernel/kernel-metadata.json").read_bytes())
                meta["title"] = "taeyangg4-" + meta["title"]
                put(stage / "kernel/kernel-metadata.json", meta)
            seal(stage)
            policy = g.parse((stage / "allowlist.json").read_bytes())
            policy["stage_sha256sums"] = g.sha((stage / "SHA256SUMS").read_bytes())
            put(stage / "allowlist.json", policy)
            commands.append((short + "-" + case, stage, "cell 0 hash" if case == "cell0" else "title must"))
        renamed = scratch / (short + "-renamed")
        build_fixture(renamed, rename=True)
        policy = g.parse((output / "allowlist.json").read_bytes())
        policy.pop("fast_path")
        policy["stage_sha256sums"] = g.sha((renamed / "SHA256SUMS").read_bytes())
        put(renamed / "allowlist.json", policy)
        commands.append((short + "-renamed", renamed, "renamed"))
        for name, stage, expected in commands:
            cmd = [sys.executable, "tools/strict_variant_gate.py", "--stage", str(stage), "--parent", str(BASE),
                   "--allow", str(stage / "allowlist.json"), "--review", str(review), "--check-only"]
            run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            put(scratch / (name + ".log"), (run.stdout + run.stderr).encode())
            summary, _ = json.JSONDecoder().raw_decode(run.stdout)
            errors = summary.get("errors", [summary.get("error", "")])
            matched = run.returncode == (0 if expected is None else 1) and (expected is None or any(expected in x for x in errors))
            runs.append(dict(case=name, exit_code=run.returncode, matched=matched, expected=expected or "PASS", errors=errors, command=cmd))
    put(scratch / "results.json", runs)
    print(json.dumps([dict(case=r["case"], exit_code=r["exit_code"], matched=r["matched"]) for r in runs], indent=2))
    return 0 if all(r["matched"] for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
