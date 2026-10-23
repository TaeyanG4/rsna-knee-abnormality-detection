"""Bootstrap for direct script execution and package imports; no working-dir change."""
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "config/workspace.toml").is_file():
        for _path in (str(_parent), str(Path(__file__).resolve().parent)):
            if _path not in sys.path:
                sys.path.insert(0, _path)
        break
else:
    raise RuntimeError("Workspace config missing; set up the portable project first")

from knee_workspace import workspace_path, workspace_root  # noqa: E402
