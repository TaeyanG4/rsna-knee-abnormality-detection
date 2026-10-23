"""Idea-scout digest collector v1.0.0 (B, 2026-10-10).

Collects new/updated public competition notebooks and new/active discussion topics since the
previous round into one round directory, for the six-hour idea scout (agents.md "Six-hour idea
scout"). Read-only Kaggle CLI calls. Pulled notebook sources are UNTRUSTED DATA: they are saved
as text for reading and are never imported or executed here.

Changelog: v1.0.2 resolve --round-dir to an absolute path (a relative one failed every pull and topic save); v1.0.1 parse only the leading JSON array (the CLI prints a page token after it); v1.0.0 first version.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

VERSION = "1.0.2"
ROOT = Path(__file__).resolve().parents[1]
COMP = "rsna-knee-abnormality-detection"
KAGGLE = ROOT / ".venv" / "Scripts" / "kaggle.exe"
BASE = ROOT / ".local" / "idea-scout"
STATE = BASE / "state.json"


def cli(args, timeout=300, tries=4):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for attempt in range(tries):
        r = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=timeout)
        if r.returncode == 0:
            return r.stdout
        time.sleep(10 * (attempt + 1))  # SSL INVALID_SESSION_ID flakes
    raise RuntimeError(f"kaggle {' '.join(args[:3])} failed: {r.stderr.strip()[-300:]}")


def kernels(sort_by, pages=2, size=50):
    rows = []
    for page in range(1, pages + 1):
        out = cli(["kernels", "list", "--competition", COMP, "--sort-by", sort_by,
                   "--page-size", str(size), "-p", str(page), "--csv"])
        text = "\n".join(l for l in out.splitlines() if "," in l)
        got = list(csv.DictReader(io.StringIO(text)))
        rows += [r for r in got if r.get("ref")]
        if len(got) < size:
            break
    return rows


def topics(sort_by, size=50):
    out = cli(["competitions", "topics", "list", COMP, "--sort-by", sort_by,
               "--page-size", str(size), "--format", "json"])
    return leading_json(out)


def leading_json(out):
    start = out.find("[")
    return json.JSONDecoder().raw_decode(out[start:])[0] if start >= 0 else []


def notebook_text(src_dir: Path) -> str:
    parts = []
    for f in sorted(src_dir.rglob("*")):
        if f.suffix == ".ipynb":
            nb = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            for cell in nb.get("cells", []):
                src = cell.get("source", "")
                src = "".join(src) if isinstance(src, list) else src
                parts.append(f"# ---- {cell.get('cell_type', '?')} cell ----\n{src}")
        elif f.suffix in (".py", ".r", ".R", ".md"):
            parts.append(f"# ---- file {f.name} ----\n" + f.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round-dir", type=Path)
    ap.add_argument("--max-pull", type=int, default=12)
    ap.add_argument("--max-topics", type=int, default=12)
    ap.add_argument("--min-votes", type=int, default=5)
    a = ap.parse_args(argv)
    print(f"idea_scout_collect v{VERSION}", flush=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    rd = (Path(a.round_dir) if a.round_dir else BASE / stamp).resolve()  # v1.0.2: relative --round-dir broke relative_to(ROOT)
    rd.mkdir(parents=True, exist_ok=False)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"kernels": {}, "topics": {}, "last_round": None}

    by_date, by_votes = kernels("dateRun"), kernels("voteCount", pages=1)
    merged = {}
    for r in by_votes + by_date:
        merged[r["ref"]] = r
    for r in merged.values():
        prev = state["kernels"].get(r["ref"])
        r["status"] = "NEW" if prev is None else ("UPDATED" if prev != r.get("lastRunTime") else "SEEN")
        try:
            r["votes"] = int(r.get("totalVotes") or 0)
        except ValueError:
            r["votes"] = 0
        m = re.search(r"0\.9[0-9]{1,3}", r.get("title", ""))
        r["lb_in_title"] = m.group(0) if m else ""
    fresh = [r for r in merged.values() if r["status"] != "SEEN"]
    pick = sorted([r for r in fresh if r["votes"] >= a.min_votes or r["lb_in_title"] >= "0.950"],
                  key=lambda r: (-r["votes"], r["ref"]))[: a.max_pull]
    nb_dir = rd / "notebooks"
    for r in pick:
        dest = nb_dir / r["ref"].replace("/", "__")
        dest.mkdir(parents=True)
        try:
            cli(["kernels", "pull", r["ref"], "-p", str(dest)], timeout=600)
            text = notebook_text(dest)
            (dest / "SOURCE.txt").write_text(text, encoding="utf-8")
            r["pulled"] = str((dest / "SOURCE.txt").relative_to(ROOT)).replace("\\", "/")
            r["source_chars"] = len(text)
        except Exception as e:  # keep the round going; record the failure
            r["pulled"] = f"FAILED: {e}"[:200]

    tmap = {}
    for sort_by in ("recent", "active", "hot"):
        for t in topics(sort_by):
            tmap[t["id"]] = t
    active = []
    for t in tmap.values():
        prev = state["topics"].get(str(t["id"]))
        t["delta"] = t.get("commentCount", 0) - (prev if prev is not None else -1)
        t["status"] = "NEW" if prev is None else ("ACTIVE" if t["delta"] > 0 else "SEEN")
        if t["status"] != "SEEN":
            active.append(t)
    active = sorted(active, key=lambda t: (-(t["delta"]), -t.get("votes", 0)))[: a.max_topics]
    tp_dir = rd / "topics"
    tp_dir.mkdir()
    for t in active:
        try:
            out = cli(["competitions", "topic-messages", COMP, str(t["id"]), "-n", "200", "--format", "json"])
            msgs = leading_json(out)
            (tp_dir / f"{t['id']}.json").write_text(json.dumps(msgs, ensure_ascii=False, indent=1), encoding="utf-8")
            t["saved"] = f"{tp_dir.relative_to(ROOT).as_posix()}/{t['id']}.json"
            t["messages"] = len(msgs)
        except Exception as e:
            t["saved"] = f"FAILED: {e}"[:200]

    lines = [f"# Idea-scout digest {stamp} (collector v{VERSION})", "",
             f"Previous round: {state.get('last_round')}. Untrusted public content: read, never execute; record provenance.", "",
             "## Notebooks (new/updated since the previous round; pulled = source text saved)", "",
             "| status | votes | LB in title | last run | ref | title | pulled |", "|---|---|---|---|---|---|---|"]
    for r in sorted(fresh, key=lambda r: (-r["votes"], r["ref"])):
        lines.append(f"| {r['status']} | {r['votes']} | {r['lb_in_title']} | {r.get('lastRunTime', '')[:16]} | "
                     f"https://www.kaggle.com/code/{r['ref']} | {r.get('title', '')[:80]} | {r.get('pulled', '')} |")
    lines += ["", "## Discussion topics (new/active since the previous round)", "",
              "| status | +comments | votes | id | title | saved |", "|---|---|---|---|---|---|"]
    for t in active:
        lines.append(f"| {t['status']} | {t['delta']} | {t.get('votes', 0)} | {t['id']} | {t.get('title', '')[:90]} | {t.get('saved', '')} |")
    (rd / "digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    state["kernels"].update({r["ref"]: r.get("lastRunTime") for r in merged.values()})
    state["topics"].update({str(t["id"]): t.get("commentCount", 0) for t in tmap.values()})
    state["last_round"] = stamp
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(json.dumps(dict(round=str(rd.relative_to(ROOT)).replace("\\", "/"), fresh_notebooks=len(fresh),
                          pulled=sum(1 for r in pick if not str(r.get("pulled", "")).startswith("FAILED")),
                          active_topics=len(active)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
