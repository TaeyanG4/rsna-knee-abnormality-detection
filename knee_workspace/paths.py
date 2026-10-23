from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def discover_root(start: str | Path | None = None) -> Path:
    explicit = os.environ.get("KNEE_WORKSPACE_ROOT")
    candidate = Path(explicit or start or Path(__file__).resolve().parents[1]).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in [candidate, *candidate.parents]:
        if (parent / "config" / "workspace.toml").is_file():
            return parent
    raise FileNotFoundError(f"Workspace config not found above {candidate}; set KNEE_WORKSPACE_ROOT")


@dataclass(frozen=True)
class Workspace:
    root: Path
    config: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path | None = None) -> "Workspace":
        base = discover_root(root)
        with (base / "config/workspace.toml").open("rb") as f:
            data = tomllib.load(f)
        local = base / "config/local.toml"
        if local.is_file():
            with local.open("rb") as f:
                overrides = tomllib.load(f)
            for key, value in overrides.items():
                if isinstance(value, dict):
                    data.setdefault(key, {}).update(value)
                else:
                    data[key] = value
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported workspace schema_version")
        return cls(base, data)

    def resolve(self, value: str | os.PathLike = "") -> Path:
        raw = os.fspath(value).replace("\\", "/")
        if raw.startswith("workspace://"):
            raw = raw[len("workspace://"):]
        # Legacy absolute paths in old manifests remain readable after relocation.
        prefixes = [str(self.root).replace("\\", "/"), *self.config.get("legacy_roots", [])]
        for prefix in sorted(prefixes, key=len, reverse=True):
            prefix = prefix.replace("\\", "/").rstrip("/")
            if raw.casefold() == prefix.casefold():
                raw = ""
                break
            if raw.casefold().startswith(prefix.casefold() + "/"):
                raw = raw[len(prefix) + 1:]
                break
        native = Path(raw).expanduser()
        if native.is_absolute():
            return native.resolve()
        if len(raw) > 1 and raw[1] == ":":
            raise ValueError("Unmapped foreign absolute path; configure its legacy root")
        parts = [p for p in raw.split("/") if p not in ("", ".")]
        if ".." in parts:
            raise ValueError("Workspace-relative paths cannot contain '..'")
        if not parts:
            return self.root
        aliases = self.config.get("paths", {})
        head = Path(aliases.get(parts[0], parts[0])).expanduser()
        base = head if head.is_absolute() else self.root / head
        return base.joinpath(*parts[1:]).resolve()

    def uri(self, path: str | os.PathLike) -> str:
        resolved = self.resolve(path)
        for alias in sorted(self.config.get("paths", {})):
            base = self.resolve(alias)
            if resolved.is_relative_to(base):
                tail = resolved.relative_to(base).as_posix()
                return f"workspace://{alias}" + ("/" + tail if tail != "." else "")
        if resolved.is_relative_to(self.root):
            return "workspace://" + resolved.relative_to(self.root).as_posix()
        raise ValueError("Path is outside configured workspace locations")


class WorkspaceRoot(os.PathLike):
    """Compatibility root: first path component resolves a configured location.

    It is deliberately not a pathlib subclass. Once an alias has resolved, normal
    Path joins apply; data filenames cannot accidentally become new root aliases.
    """
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def __fspath__(self) -> str:
        return str(self.workspace.root)

    def __str__(self) -> str:
        return os.fspath(self)

    def __truediv__(self, child: str | os.PathLike) -> Path:
        return self.workspace.resolve(child)

    def __getattr__(self, name: str):
        return getattr(self.workspace.root, name)


def workspace_root(start: str | Path | None = None) -> WorkspaceRoot:
    # An old --root . argument means the configured project, not arbitrary cwd.
    return WorkspaceRoot(Workspace.load(None if start in (None, "", ".") else start))


def workspace_path(value: str | os.PathLike = "", *, root: str | Path | None = None) -> Path:
    return Workspace.load(root).resolve(value)
