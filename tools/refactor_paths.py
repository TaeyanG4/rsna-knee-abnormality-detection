"""One-time audited source conversion. No training or downloaded code execution."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / ".local/migrations/layout-v2"
IMPORT = "try:\n    from ._workspace import workspace_root, workspace_path\nexcept ImportError:\n    from _workspace import workspace_root, workspace_path\n"


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    archive = BACKUP / "before-refactor.zip"
    sources = [*sorted((ROOT / "src").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py")),
               *sorted((ROOT / "scripts").glob("*.ps1")), ROOT / "config/workspace.toml"]
    if not archive.exists():
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sources:
                z.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        originals = {name: z.read(name) for name in z.namelist()}
    receipt = []
    for p in sources:
        before = originals[p.relative_to(ROOT).as_posix()]
        text = before.decode("utf-8-sig")
        if p.suffix == ".py" and p.name != "_workspace.py":
            converted = re.sub(r"Path\(__file__\)\.resolve\(\)\.parents\[1\]", "workspace_root()", text)
            converted = re.sub(r"Path\(r?['\"]H:\\kaggle\\competitions\\rsna-knee-abnormality-detection['\"]\)", "workspace_root()", converted)
            converted = re.sub(r"((?:ROOT|root)\s*=\s*)Path\(['\"]\.['\"]\)", r"\1workspace_root()", converted)
            converted = re.sub(r"Path\((args\.(?:out|cache_dir|features)|a\.cache_dir)\)", r"workspace_path(\1)", converted)
            converted = converted.replace("root=Path(args.root)", "root=workspace_root(args.root)")
            if p.name == "_patch_exp042_v2.py":
                converted = converted.replace("Path('src/train_adjacent_2p5d_clean.py')", "workspace_path('src/train_adjacent_2p5d_clean.py')")
            if p.name == "finalize_audit.py":
                converted = converted.replace("Path(r['path'])", "workspace_path(r['path'])")
            if converted != text:
                tree = ast.parse(converted)
                offset = 0
                for node in tree.body:
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        offset = node.end_lineno
                    elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
                        offset = node.end_lineno
                    else:
                        break
                lines = converted.splitlines(keepends=True)
                lines.insert(offset, IMPORT)
                converted = "".join(lines)
                ast.parse(converted)
            text = converted
        elif p.suffix == ".ps1":
            # Keep parameter override support and source-relative discovery.
            text = text.replace("'H:\\kaggle\\competitions\\rsna-knee-abnormality-detection'", "(Get-KneeWorkspaceRoot)")
            if text != before.decode("utf-8-sig"):
                if text.lstrip().startswith("param("):
                    text = text.replace("[string]$Root = (Get-KneeWorkspaceRoot)", "[string]$Root = ''")
                    close = re.match(r"param\([\s\S]*?\)", text).end()
                    text = text[:close] + '\n. (Join-Path $PSScriptRoot "Workspace.ps1")\nif (-not $Root) { $Root = Get-KneeWorkspaceRoot }\n' + text[close:]
                else:
                    text = '. (Join-Path $PSScriptRoot "Workspace.ps1")\n' + text
            # Runner arguments become canonical paths, while old result records stay immutable.
            replacements = {".\\src\\": ".\\training\\src\\", ".\\scripts\\": ".\\training\\runners\\",
                            "from src.": "from training.src.", "experiments\\": "artifacts\\experiments\\",
                            "logs\\": "artifacts\\logs\\", "cache\\": "storage\\cache\\derived\\"}
            for old, new in replacements.items():
                text = text.replace(old, new)
        after = text.encode("utf-8")
        if after != before:
            p.write_bytes(after)
            receipt.append({"path": p.relative_to(ROOT).as_posix(), "before_sha256": hashlib.sha256(before).hexdigest(),
                            "after_sha256": hashlib.sha256(after).hexdigest()})
    (BACKUP / "refactor.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"changed_files": len(receipt), "backup": str(archive)}))


if __name__ == "__main__":
    main()
