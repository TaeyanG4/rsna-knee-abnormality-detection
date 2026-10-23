"""Small dependency-free path command, also used by PowerShell runners."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee_workspace import Workspace

parser = argparse.ArgumentParser()
parser.add_argument("path", nargs="?", default="")
parser.add_argument("--root")
args = parser.parse_args()
print(Workspace.load(args.root).resolve(args.path))
