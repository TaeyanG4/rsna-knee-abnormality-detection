"""Allow runners to load the shared workspace package from any working directory."""
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "config/workspace.toml").is_file():
        sys.path.insert(0, str(_parent))
        break
else:
    raise RuntimeError("Workspace config missing")

from knee_workspace import workspace_path, workspace_root  # noqa: E402
