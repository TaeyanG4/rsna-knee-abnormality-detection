"""L3 variant F v1.0.0: private, resumable training relabel throughput pilot.

Changelog v1.0.0: reuse EXP147 F and EXP145 15-slot montage, exclude Gold58,
strict categorical severity output, durable paid-call accounting and review gates.
No GPU, Kaggle, Gold scoring, report persistence, or raw provider-output files.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import queue
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp109_relabel_v1 as e109
from tools import exp145_vlm_reading_pilot as e145
from tools import exp147_input_ab as e147

VERSION = "1.0.0"
WORK = ROOT / ".local/scratch/l3-variant-f/P1"
UID = "StudyInstanceUID"
TARGETS = tuple(e109.T)
WEAK = ("Lateral Meniscus", "PF OA", "Synovitis", "Effusion", "Baker's")
SEVERITY = ("absent", "below_threshold", "threshold_met", "uncertain", "not_assessable")
SEED = 20261016
FOLDS = ROOT / "artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/folds.csv"
FOLDS_SHA = "21bec603e6130f9afcb6fd81dcdfe4bdd98128fc3d229ceceeb4356175f4ded0"
TRAIN = ROOT / "storage/competition/train.csv"
MODEL = "gpt-6-astra"
MAX_ATTEMPTS = 2
PILOT_CEILING = 4_000_000
RESERVATION = 50_000
SOURCE_FILES = (Path(__file__), ROOT / "tools/exp109_relabel_v1.py",
                ROOT / "tools/exp145_vlm_reading_pilot.py", ROOT / "tools/exp147_input_ab.py",
                ROOT / "training/src/raptor_common.py", e147.RULE_DOC)


class GuardError(RuntimeError):
    """Only fixed non-sensitive reason codes are exposed by the CLI."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def retry_io(fn):
    for attempt in range(40):
        try:
            return fn()
        except OSError as exc:
            if not isinstance(exc, PermissionError) and getattr(exc, "winerror", None) not in (5, 32):
                raise
            if attempt == 39:
                raise GuardError("PERSISTENT_OUTPUT_IO") from None
            time.sleep(min(.05 * (attempt + 1), 1.0))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    retry_io(lambda: os.replace(tmp, path))


def immutable(path, value):
    if Path(path).exists():
        if load(path) != value:
            raise GuardError("IMMUTABLE_ARTIFACT_CHANGED")
    else:
        atomic(path, value)


def fixed_work(path):
    path = Path(path).resolve()
    if path != WORK.resolve():
        raise GuardError("FIXED_WORK_REQUIRED")
    return path


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise GuardError("CUDA_VISIBLE_DEVICES_MUST_BE_EMPTY")


def schema(code):
    pair = {"type": "object", "properties": {
        "p": {"type": "number", "minimum": 0, "maximum": 1},
        "severity": {"type": "string", "enum": list(SEVERITY)}},
        "required": ["p", "severity"], "additionalProperties": False}
    findings = {"type": "object", "properties": {t: pair for t in TARGETS},
                "required": list(TARGETS), "additionalProperties": False}
    return {"type": "object", "properties": {code: findings}, "required": [code],
            "additionalProperties": False}


def strict_parse(text, code):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise GuardError("DUPLICATE_JSON_KEY")
            out[key] = value
        return out
    def bad_constant(_):
        raise GuardError("NONFINITE_JSON")
    try:
        obj = json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)
    except (ValueError, TypeError):
        raise GuardError("INVALID_JSON") from None
    if not isinstance(obj, dict) or set(obj) != {code}:
        raise GuardError("EXAM_KEYS")
    row = obj[code]
    if not isinstance(row, dict) or set(row) != set(TARGETS):
        raise GuardError("TARGET_KEYS")
    for value in row.values():
        if not isinstance(value, dict) or set(value) != {"p", "severity"}:
            raise GuardError("FINDING_KEYS")
        p = value["p"]
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise GuardError("PROBABILITY_RANGE")
        if value["severity"] not in SEVERITY:
            raise GuardError("SEVERITY_ENUM")
    return row


def prompt_f(code, report, labels, retry=False):
    base = e147._prompt_f(code, report, {code: labels})
    old = f'Return ONLY the JSON object {{"{code}": {{...12 findings...}}}}.\n'
    if base.count(old) != 1:
        raise GuardError("F_TEMPLATE_CHANGED")
    ending = ("For each of the 12 findings, output exactly two fields: p (a finite plain number in [0,1]) "
              "and severity (one of absent, below_threshold, threshold_met, uncertain, not_assessable). "
              "The severity note is a categorical observation relative to the official target definition: "
              "absent=no finding, below_threshold=finding present below the official positive threshold, "
              "threshold_met=official threshold met, uncertain=threshold unresolved, "
              "not_assessable=insufficient image/report evidence. Do not quote or paraphrase report prose. "
              f'Return ONLY {{"{code}": {{"ACL": {{"p": 0.12, "severity": "uncertain"}}, ...12 findings...}}}}.\n')
    result = base.replace(old, ending)
    if retry:
        result += "FORMAT REMINDER: return the exact JSON schema, no Markdown, no prose, no extra keys.\n"
    return result


def corpus():
    sys.path.insert(0, str(ROOT / "training/src"))
    from raptor_common import Corpus
    obj = Corpus(e145.CORPUS)
    if obj.contract.lower() != "raptor64_336_c140_s02":
        raise GuardError("CORPUS_CONTRACT")
    return obj


def inputs():
    import numpy as np
    import pandas as pd
    # No report or Gold target is opened here, apart from the sanctioned membership helper.
    train = pd.read_csv(TRAIN, usecols=[UID], dtype=str)
    if train[UID].isna().any() or train[UID].duplicated().any():
        raise GuardError("TRAIN_UIDS")
    gold = e109.gold_uids(ROOT)
    if len(gold) != 58:
        raise GuardError("GOLD_COUNT")
    ids = sorted(set(train[UID]) - gold)
    if sha(FOLDS) != FOLDS_SHA:
        raise GuardError("FOLDS_HASH")
    fold_df = pd.read_csv(FOLDS, usecols=[UID, "fold"], dtype={UID: str})
    if fold_df[UID].duplicated().any() or not set(fold_df["fold"]) <= set(range(5)):
        raise GuardError("FOLD_VALUES")
    folds = dict(zip(fold_df[UID], fold_df["fold"].astype(int)))
    cols = [UID, *TARGETS, *(t + "__conf" for t in TARGETS)]
    flight = pd.read_csv(e145.FLIGHT, usecols=cols, dtype={UID: str})
    if flight[UID].duplicated().any():
        raise GuardError("FLIGHT_UIDS")
    flight = flight.set_index(UID)
    c = corpus()
    rows = []
    for index, uid in enumerate(ids):
        reasons = []
        if uid not in flight.index:
            reasons.append("flight_missing")
        else:
            arr = flight.loc[uid, cols[1:]].to_numpy(float)
            if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
                reasons.append("flight_invalid")
        if uid not in c:
            reasons.append("corpus_missing")
        elif not np.asarray(c.mask(uid), bool).any():
            reasons.append("unavailable_images")
        rows.append({"uid": uid, "code": f"T{index + 1:05d}", "fold": folds.get(uid, -1),
                     "ineligible": reasons})
    return rows, flight, c, gold


def probability_bin(p):
    return "negative" if p < .4 else "positive" if p > .6 else "uncertain"


def pick_pilot(rows, flight):
    """Forty per fold; greedy coverage of five target x three probability bins.

    At each step, favor cells with low selected counts relative to attainable
    equal-bin quota 40/3. Ties use a single fixed seeded permutation of UIDs.
    Each selected exam covers all five target bins; no replacement or Gold use.
    """
    eligible = [r for r in rows if not r["ineligible"] and r["fold"] in range(5)]
    rng = random.Random(SEED)
    selected = []
    for fold in range(5):
        candidates = sorted([r for r in eligible if r["fold"] == fold], key=lambda r: r["code"])
        rng.shuffle(candidates)
        if len(candidates) < 40:
            raise GuardError("PILOT_FOLD_SHORTAGE")
        bins = {r["code"]: {t: probability_bin(float(flight.loc[r["uid"], t])) for t in WEAK}
                for r in candidates}
        count = {(t, b): 0 for t in WEAK for b in ("negative", "uncertain", "positive")}
        population = {(t, b): sum(bins[r["code"]][t] == b for r in candidates) for t, b in count}
        goal = {cell: min(40/3, n) for cell, n in population.items()}
        for _ in range(40):
            def score(r):
                return sum(max(0, goal[(t, bins[r["code"]][t])]-count[(t, bins[r["code"]][t])]) /
                           max(1, goal[(t, bins[r["code"]][t])]) for t in WEAK)
            row = max(candidates, key=score)
            candidates.remove(row)
            for t, b in bins[row["code"]].items():
                count[t, b] += 1
            selected.append({"code": row["code"], "fold": fold, "bins": bins[row["code"]]})
    if len(selected) != 200:
        raise GuardError("PILOT_SIZE")
    return selected


def source_hashes():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in SOURCE_FILES}


def register(work=WORK):
    work = fixed_work(work)
    rows, flight, c, gold = inputs()
    selection = pick_pilot(rows, flight)
    sources = source_hashes()
    # Hash train.csv bytes without decoding report text. UID-only and Gold membership were read above.
    files = [TRAIN, FOLDS, e145.FLIGHT, e145.CORPUS / "manifest.json",
             e145.CORPUS / "ids.npy", e145.CORPUS / "masks.npy"]
    reg = {"version": VERSION, "candidate": "P1-L3-variant-F", "model": MODEL,
           "reasoning_effort": "xhigh", "seed": SEED, "sources": sources,
           "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files},
           "gold_membership_sha256": digest(sorted(gold)), "gold_excluded": len(gold),
           "rows": rows, "pilot": selection,
           "pilot_count": 200, "pilot_lanes": 2, "pilot_token_ceiling": PILOT_CEILING,
           "reservation_tokens": RESERVATION, "max_attempts": MAX_ATTEMPTS,
           "strata": "40/fold; seeded greedy coverage of five targets x NEG(p<.4)/UNC(.4<=p<=.6)/POS(p>.6); no severity truth",
           "output": "12 probabilities and categorical severity; no Gold labels or report prose",
           "interpretation": "flight agreement is descriptive, not accuracy or training benefit; EXP147 exploratory CHALLENGED",
           "budget_rule": "No new call if spent+reserved+50000>ceiling; missing usage or overrun halts; active calls may overshoot",
           "full_requires_separate_review": True,
           "full_eligible": all(not r["ineligible"] for r in rows),
           "counts": {"training": len(rows) + len(gold), "non_gold": len(rows),
                      "unassigned_fold": sum(r["fold"] == -1 for r in rows),
                      "not_ready": sum(bool(r["ineligible"]) for r in rows)},
           "sampling_coverage": [{"fold": f, "target": t, "bin": b,
                "available": sum(r["fold"] == f and not r["ineligible"] and probability_bin(float(flight.loc[r["uid"],t])) == b for r in rows),
                "selected": sum(r["fold"] == f and r["bins"][t] == b for r in selection)}
                for f in range(5) for t in WEAK for b in ("negative", "uncertain", "positive")],
           "cli_identity": cli_identity()}
    immutable(work / "registration.json", reg)
    return {"registration_sha256": sha(work / "registration.json"), **reg["counts"],
            "pilot": 200, "full_eligible": reg["full_eligible"]}


def verify_registration(work=WORK):
    reg = load(work / "registration.json")
    if reg["version"] != VERSION or reg["sources"] != source_hashes():
        raise GuardError("SOURCE_IDENTITY_CHANGED")
    if reg["cli_identity"] != cli_identity():
        raise GuardError("CLI_IDENTITY_CHANGED")
    for name, expected in reg["inputs"].items():
        if sha(ROOT / name) != expected:
            raise GuardError("INPUT_IDENTITY_CHANGED")
    if len(reg["pilot"]) != 200 or len({r["code"] for r in reg["pilot"]}) != 200:
        raise GuardError("REGISTERED_PILOT")
    gold = e109.gold_uids(ROOT)
    if digest(sorted(gold)) != reg["gold_membership_sha256"] or any(r["uid"] in gold for r in reg["rows"]):
        raise GuardError("GOLD_EXCLUSION")
    return reg


def select_slots(mask, plane):
    import numpy as np
    for start, width in e145.GROUPS[plane]:
        filled = [s for s in range(start, start + width) if mask[s]]
        if len(filled) >= e145.COUNT[plane]:
            lo, hi = e145.SPAN[plane]
            return [filled[int(round(k))] for k in np.linspace(lo * (len(filled) - 1),
                    hi * (len(filled) - 1), e145.COUNT[plane])]
    return []


def prepare_one(work, c, row):
    import cv2
    import numpy as np
    code, uid = row["code"], row["uid"]
    out = work / "images" / code
    receipt = work / "prepared" / (code + ".json")
    if receipt.exists():
        rec = load(receipt)
        if rec["registration_sha256"] != sha(work / "registration.json"):
            raise GuardError("PREPARED_REGISTRATION")
        if any(sha(work / f) != h for f, h in rec["files"].items()):
            raise GuardError("PREPARED_IMAGE_CHANGED")
        return rec
    out.mkdir(parents=True, exist_ok=True)
    vol, mask = c.volume(uid), np.asarray(c.mask(uid), bool)
    if tuple(vol.shape) != (64, 336, 336):
        raise GuardError("VOLUME_SHAPE")
    slots, files = {}, {}
    for plane in ("sag", "cor", "ax"):
        chosen = select_slots(mask, plane)
        slots[plane] = chosen
        cols, rows = e145.GRID[plane]
        canvas = np.zeros((rows * 336, cols * 336), np.uint8)
        for i, s in enumerate(chosen):
            rr, cc = divmod(i, cols)
            tile = np.array(vol[s], np.uint8)
            cv2.putText(tile, f"{plane.upper()} {i+1}/{len(chosen)}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, 255, 1, cv2.LINE_AA)
            canvas[rr*336:(rr+1)*336, cc*336:(cc+1)*336] = tile
        if not chosen:
            cv2.putText(canvas, f"{plane.upper()}: not available", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        path = out / f"{code}_{plane}.png"
        ok, png = cv2.imencode(".png", canvas)
        if not ok:
            raise GuardError("PNG_ENCODE")
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            f.write(png.tobytes()); f.flush(); os.fsync(f.fileno())
        retry_io(lambda: os.replace(tmp, path))
        files[str(path.relative_to(work)).replace("\\", "/")] = sha(path)
    schema_path = out / "schema.json"
    immutable(schema_path, schema(code))
    files[str(schema_path.relative_to(work)).replace("\\", "/")] = sha(schema_path)
    rec = {"code": code, "registration_sha256": sha(work / "registration.json"),
           "files": files, "slots": slots, "missing_planes": sum(not v for v in slots.values()),
           "volume_sha256": hashlib.sha256(np.ascontiguousarray(vol).tobytes()).hexdigest()}
    immutable(receipt, rec)
    return rec


def prepare(work=WORK, scope="pilot", limit=None):
    reg = verify_registration(work)
    codes = {x["code"] for x in reg["pilot"]} if scope == "pilot" else {x["code"] for x in reg["rows"]}
    rows = [r for r in reg["rows"] if r["code"] in codes]
    if any(r["ineligible"] for r in rows):
        raise GuardError("SELECTED_INPUTS_NOT_READY")
    c = corpus()
    selected = rows[:limit] if limit else rows
    recs = [prepare_one(work, c, r) for r in selected]
    return {"prepared": len(recs), "scope": scope, "missing_planes": sum(r["missing_planes"] for r in recs),
            "image_bytes": sum((work/f).stat().st_size for r in recs for f in r["files"] if f.endswith(".png"))}


def validate(work=WORK, scope="pilot"):
    reg = verify_registration(work)
    codes = [x["code"] for x in (reg["pilot"] if scope == "pilot" else reg["rows"])]
    for code in codes:
        rec = load(work / "prepared" / (code + ".json"))
        if rec["registration_sha256"] != sha(work / "registration.json"):
            raise GuardError("PREPARED_REGISTRATION")
        if any(sha(work/f) != h for f, h in rec["files"].items()):
            raise GuardError("PREPARED_IMAGE_CHANGED")
    result = {"version": VERSION, "scope": scope, "validated": len(codes),
              "registration_sha256": sha(work / "registration.json"), "cuda_visible_devices_empty": True,
              "gold_overlap": 0, "full_eligible": reg["full_eligible"]}
    atomic(work / f"validate-{scope}.json", result)
    return result


class Ledger:
    """SQLite FULL-sync transactions serialize reservations across processes.

    Inflight/unknown paid calls are never automatically retried. A completion and
    its parsed payload commit together before exporting the per-exam receipt.
    """
    def __init__(self, directory, registration_hash, ceiling, scope):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "ledger.sqlite3"
        with self.connect() as con:
            con.executescript("""CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (code TEXT NOT NULL, attempt INTEGER NOT NULL,
                state TEXT NOT NULL, reserved INTEGER NOT NULL, input_tokens INTEGER, output_tokens INTEGER,
                cached_tokens INTEGER, started REAL NOT NULL, elapsed REAL, reason TEXT, payload TEXT,
                PRIMARY KEY(code,attempt));
                CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, started REAL NOT NULL,
                finished REAL, lanes INTEGER NOT NULL, elapsed REAL);""")
            expected = {"registration": registration_hash, "ceiling": str(ceiling), "scope": scope}
            with self.tx(con):
                for key, value in expected.items():
                    old = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
                    if old and old[0] != value:
                        raise GuardError("LEDGER_IDENTITY_OR_CEILING")
                    con.execute("INSERT OR IGNORE INTO meta VALUES (?,?)", (key, value))

    @contextlib.contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA synchronous=FULL")
        con.execute("PRAGMA busy_timeout=30000")
        try:
            yield con
        finally:
            con.close()

    @contextlib.contextmanager
    def tx(self, con):
        con.execute("BEGIN IMMEDIATE")
        try:
            yield
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise

    def reserve(self, code):
        with self.connect() as con, self.tx(con):
            if con.execute("SELECT 1 FROM meta WHERE key='halt'").fetchone():
                return "halt", None
            previous = list(con.execute("SELECT * FROM attempts WHERE code=? ORDER BY attempt", (code,)))
            if previous and previous[-1]["state"] == "complete":
                return "complete", None
            if any(r["state"] in ("inflight", "unknown") for r in previous):
                return "blocked", None
            if len(previous) >= MAX_ATTEMPTS:
                return "exhausted", None
            spent, reserved = con.execute("SELECT COALESCE(SUM(input_tokens+output_tokens),0), "
                                         "COALESCE(SUM(CASE WHEN state='inflight' THEN reserved ELSE 0 END),0) FROM attempts").fetchone()
            ceiling = int(con.execute("SELECT value FROM meta WHERE key='ceiling'").fetchone()[0])
            if spent + reserved + RESERVATION > ceiling:
                return "budget", None
            attempt = len(previous) + 1
            con.execute("INSERT INTO attempts(code,attempt,state,reserved,started) VALUES(?,?,?,?,?)",
                        (code, attempt, "inflight", RESERVATION, time.time()))
            return "reserved", attempt

    def finish(self, code, attempt, usage, elapsed, payload=None, reason=None):
        with self.connect() as con, self.tx(con):
            existing = con.execute("SELECT * FROM attempts WHERE code=? AND attempt=?", (code, attempt)).fetchone()
            if not existing or existing["state"] != "inflight":
                raise GuardError("ATTEMPT_STATE")
            state = "unknown" if usage is None else ("complete" if payload is not None else "invalid")
            ins, outs, cached = ((None, None, None) if usage is None else
                                 (usage["input_tokens"], usage["output_tokens"], usage.get("cached_input_tokens", 0)))
            con.execute("UPDATE attempts SET state=?,input_tokens=?,output_tokens=?,cached_tokens=?,elapsed=?,reason=?,payload=? "
                        "WHERE code=? AND attempt=?", (state, ins, outs, cached, elapsed, reason,
                         json.dumps(payload, allow_nan=False) if payload is not None else None, code, attempt))
            if usage is None or ins + outs > existing["reserved"] or reason == "FORBIDDEN_TOOL_EVENT":
                con.execute("INSERT OR REPLACE INTO meta VALUES('halt',?)",
                            ("UNKNOWN_USAGE" if usage is None else (reason or "RESERVATION_OVERRUN"),))
        try:
            self.export_receipt(code)
        except (OSError, GuardError):
            pass  # recoverable export; transactional completion is already durable.

    def export_receipt(self, code):
        with self.connect() as con:
            rows = [dict(r) for r in con.execute("SELECT * FROM attempts WHERE code=? ORDER BY attempt", (code,))]
            meta = dict(con.execute("SELECT key,value FROM meta"))
        if not rows:
            return
        for row in rows:
            row.pop("payload", None)
        atomic(self.directory / "receipts" / f"{code}.json", {"version": VERSION, "code": code,
               "registration_sha256": meta["registration"], "scope": meta["scope"], "attempts": rows})
        with self.connect() as con:
            row = con.execute("SELECT payload FROM attempts WHERE code=? AND state='complete'", (code,)).fetchone()
        if row:
            immutable(self.directory / "labels" / f"{code}.json", {"code": code, "findings": json.loads(row[0]),
                      "registration_sha256": meta["registration"], "version": VERSION})

    def summary(self):
        with self.connect() as con:
            counts = dict(con.execute("SELECT state,COUNT(*) FROM attempts GROUP BY state"))
            n = con.execute("SELECT COUNT(DISTINCT code) FROM attempts").fetchone()[0]
            retries = con.execute("SELECT COUNT(*),COUNT(DISTINCT code) FROM attempts WHERE attempt>1").fetchone()
            usage = con.execute("SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),"
                                "COALESCE(SUM(cached_tokens),0),COALESCE(SUM(elapsed),0) FROM attempts").fetchone()
            meta = dict(con.execute("SELECT key,value FROM meta"))
            wall = con.execute("SELECT COALESCE(SUM(elapsed),0),COUNT(*),"
                               "SUM(CASE WHEN finished IS NULL THEN 1 ELSE 0 END) FROM sessions").fetchone()
        return {"version": VERSION, "scope": meta["scope"], "registration_sha256": meta["registration"],
                "states": counts, "attempted_exams": n, "input_tokens": usage[0], "output_tokens": usage[1],
                "attempts": sum(counts.values()), "retry_attempts": retries[0], "retried_exams": retries[1],
                "completed_exams": counts.get("complete", 0), "unresolved_exams": n-counts.get("complete", 0),
                "cached_input_tokens": usage[2], "total_tokens": usage[0]+usage[1], "reader_seconds": usage[3],
                "token_ceiling": int(meta["ceiling"]), "halt": meta.get("halt"),
                "wall_seconds_all_sessions": wall[0], "sessions": wall[1],
                "unfinished_sessions": wall[2] or 0,
                "unknown_requires_manual_reconciliation": bool(counts.get("unknown") or counts.get("inflight"))}

    def start_session(self, lanes):
        with self.connect() as con, self.tx(con):
            session = "local:" + str(time.time_ns())
            con.execute("INSERT INTO sessions(id,started,lanes) VALUES(?,?,?)", (session, time.time(), lanes))
        return session

    def finish_session(self, session, elapsed):
        with self.connect() as con, self.tx(con):
            con.execute("UPDATE sessions SET finished=?,elapsed=? WHERE id=? AND finished IS NULL",
                        (time.time(), elapsed, session))

    def reuse_pilot(self, pilot, work):
        """Import immutable paid-call history once, including spend, retry counts and wall time.

        Full ceiling is cumulative, including reused pilot spend. Source receipts
        stay intact; the manifest links every imported attempt and label to them.
        """
        pilot.startup_guard()
        with pilot.connect() as con:
            attempts = [dict(r) for r in con.execute("SELECT * FROM attempts ORDER BY code,attempt")]
            sessions = [dict(r) for r in con.execute("SELECT * FROM sessions ORDER BY id")]
        references = []
        for code in sorted({r["code"] for r in attempts}):
            pilot.export_receipt(code)
            receipt = pilot.directory / "receipts" / (code + ".json")
            label = pilot.directory / "labels" / (code + ".json")
            references.append({"code": code, "receipt": str(receipt.relative_to(work)),
                               "receipt_sha256": sha(receipt),
                               "label": str(label.relative_to(work)) if label.exists() else None,
                               "label_sha256": sha(label) if label.exists() else None})
        with self.connect() as con, self.tx(con):
            for row in attempts:
                old = con.execute("SELECT * FROM attempts WHERE code=? AND attempt=?",
                                  (row["code"], row["attempt"])).fetchone()
                if old and dict(old) != row:
                    raise GuardError("PILOT_REUSE_CONFLICT")
                if not old:
                    keys = list(row)
                    con.execute("INSERT INTO attempts(" + ",".join(keys) + ") VALUES(" +
                                ",".join("?" for _ in keys) + ")", tuple(row[k] for k in keys))
            for row in sessions:
                row["id"] = "pilot:" + row["id"]
                old = con.execute("SELECT * FROM sessions WHERE id=?", (row["id"],)).fetchone()
                if old and dict(old) != row:
                    raise GuardError("PILOT_SESSION_CONFLICT")
                con.execute("INSERT OR IGNORE INTO sessions VALUES(?,?,?,?,?)", tuple(row.values()))
        immutable(self.directory / "reused-pilot.json", {"sources": references,
                  "registration_sha256": pilot.summary()["registration_sha256"],
                  "budget_includes_pilot_spend": True})
        for code in sorted({r["code"] for r in attempts}):
            self.export_receipt(code)
        return sum(r["state"] == "complete" for r in attempts)

    def startup_guard(self):
        # A process-level run lock ensures every pre-existing inflight call is a crash, not an active peer.
        state = self.summary()
        if state["unknown_requires_manual_reconciliation"] or state["halt"]:
            raise GuardError("LEDGER_RECONCILIATION_REQUIRED")


@contextlib.contextmanager
def run_lock(work):
    """OS-owned byte lock is released on process death; never delete a stale lock file."""
    path = work / "run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        f.seek(0); f.write(b"0"); f.flush(); f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise GuardError("RUN_ALREADY_ACTIVE") from None
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def reader_command(code, work=WORK):
    img = work / "images" / code
    exe = shutil.which("codex")
    if not exe:
        raise GuardError("CODEX_NOT_FOUND")
    args = [exe, "exec", "-m", MODEL, "-s", "read-only", "-C", str(img),
            "--skip-git-repo-check", "--ephemeral", "--json", "--color", "never",
            "--output-schema", str(img / "schema.json"),
            "-c", 'model_reasoning_effort="xhigh"', "-c", 'approval_policy="never"',
            "-c", "project_doc_max_bytes=0", "-c", "mcp_servers={}", "-c", 'web_search="disabled"']
    for feature in ("shell_tool", "unified_exec", "code_mode", "code_mode_host", "apps", "browser_use",
                    "browser_use_external", "computer_use", "multi_agent", "plugins", "hooks", "view_image",
                    "image_generation", "skill_search", "skill_mcp_dependency_install", "tool_search"):
        args += ["--disable", feature]
    for plane in ("sag", "cor", "ax"):
        args += ["-i", str(img / f"{code}_{plane}.png")]
    return args + ["-"]


def cli_identity():
    exe = shutil.which("codex")
    if not exe:
        raise GuardError("CODEX_NOT_FOUND")
    return {"executable": str(Path(exe).resolve()), "executable_sha256": sha(exe),
            "command_sha256": digest(reader_command("SCHEMA_CODE")),
            "model": MODEL, "reasoning_effort": "xhigh", "ephemeral": True}


def cli_preflight(work=WORK):
    """Validate the exact config/feature flag syntax without any reader/API call."""
    cmd = reader_command("SCHEMA_CODE", work)
    flags = []
    for i, arg in enumerate(cmd):
        if arg in ("-c", "--disable"):
            flags.extend([arg, cmd[i+1]])
    result = subprocess.run([cmd[0], *flags, "features", "list"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30)
    wanted = [cmd[i+1] for i, arg in enumerate(cmd) if arg == "--disable"]
    actual = {}
    for line in result.stdout.splitlines():
        words = line.split()
        if words:
            actual[words[0]] = words[-1]
    # This installed CLI forces the unified_exec backend true even when disabled.
    # shell_tool=false is the separate tool-exposure switch. Preserve the actual
    # state and require its exact receipt in independent review; never claim all
    # requested switches took effect or absolute config isolation.
    forced = [name for name in wanted if actual.get(name) != "false"]
    if result.returncode or any(name != "unified_exec" for name in forced) or actual.get("unified_exec") not in ("false", "true"):
        raise GuardError("CLI_PREFLIGHT_FLAGS")
    rec = {"passed": True, "requested_disabled_features": wanted,
           "effective_features": {name: actual[name] for name in wanted},
           "forced_backend_features": forced, "cli_identity": cli_identity(),
           "calls_made": 0, "config_isolation": "User route/auth config inherited without inspection; shell_tool=false; unified_exec backend may remain forced true; tool event fail-closed"}
    atomic(work / "cli-preflight.json", rec)
    return rec


def parse_events(lines):
    """Parse ONLY safe event metadata and final answer in memory; never persist raw JSONL."""
    usage, messages, forbidden, malformed = None, [], False, False
    for line in lines.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            malformed = True
            continue
        if not isinstance(event, dict):
            malformed = True
            continue
        item = event.get("item", {})
        kind = item.get("type", "") if isinstance(item, dict) else ""
        if kind and kind not in ("agent_message", "reasoning"):
            forbidden = True
        if event.get("type") == "item.completed" and kind == "agent_message":
            messages.append(item.get("text", ""))
        if event.get("type") == "turn.completed":
            u = event.get("usage", {})
            valid = isinstance(u, dict) and all(type(u.get(k)) is int and u[k] >= 0
                    for k in ("input_tokens", "output_tokens"))
            cached = u.get("cached_input_tokens", 0) if isinstance(u, dict) else None
            valid = valid and type(cached) is int and 0 <= cached <= u["input_tokens"]
            if usage is not None or not valid:
                malformed = True
            else:
                usage = {"input_tokens": u["input_tokens"], "output_tokens": u["output_tokens"],
                         "cached_input_tokens": cached}
    if malformed:
        return None, "", "EVENT_STREAM_INVALID"
    return usage, messages[-1] if len(messages) == 1 else "", ("FORBIDDEN_TOOL_EVENT" if forbidden else None)


def invoke_reader(args, prompt):
    import psutil
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Prompt and provider text remain in process pipes/memory; ephemeral prevents Codex session persistence.
    # Existing EXP147 1200-second call bound, not a scientific/GPU runtime limit.
    proc, owned, lines, error = None, {}, [], None
    events = queue.Queue()
    def drain_stdout(pipe):
        try:
            for line in pipe:
                events.put(line)
        finally:
            events.put(None)
    def discard_stderr(pipe):
        for _ in pipe:
            pass
    try:
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", env=env)
        parent = psutil.Process(proc.pid)
        owned[parent.pid] = parent
        t1 = threading.Thread(target=drain_stdout, args=(proc.stdout,), daemon=True)
        t2 = threading.Thread(target=discard_stderr, args=(proc.stderr,), daemon=True)
        t1.start(); t2.start()
        proc.stdin.write(prompt); proc.stdin.close()
        started, size, last_scan = time.monotonic(), 0, 0
        eof = False
        while not eof:
            if time.monotonic()-started >= 1200:
                error = "READER_TIMEOUT_UNKNOWN"
                break
            if time.monotonic()-last_scan > 1:
                last_scan = time.monotonic()
                try:
                    for child in parent.children(recursive=True):
                        owned[child.pid] = child
                except psutil.Error:
                    pass
            try:
                line = events.get(timeout=.2)
            except queue.Empty:
                continue
            if line is None:
                eof = True
                continue
            size += len(line)
            if size > 8_000_000:
                error = "READER_OUTPUT_LIMIT_UNKNOWN"
                break
            lines.append(line)
            try:
                event = json.loads(line)
                item = event.get("item", {}) if isinstance(event, dict) else {}
                kind = item.get("type", "") if isinstance(item, dict) else ""
                if kind and kind not in ("agent_message", "reasoning"):
                    error = "FORBIDDEN_TOOL_EVENT"
                    break
            except ValueError:
                pass
        if error is None:
            proc.wait(timeout=10)
    except (OSError, subprocess.SubprocessError, psutil.Error):
        error = "READER_TRANSPORT_UNKNOWN"
    finally:
        # Terminate only this invocation's verified child process objects, even on interruption.
        for child in reversed(list(owned.values())):
            try:
                if child.is_running():
                    child.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(list(owned.values()), timeout=2)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except subprocess.SubprocessError:
                pass
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream and not stream.closed:
                    stream.close()
    usage, answer, parsed_error = parse_events("".join(lines))
    error = error or parsed_error
    if proc is not None and proc.returncode and error is None:
        error = "READER_NONZERO"
    return usage, answer, error


def private_report(uid):
    # Sole report access: selecting one exam for stdin, no printing, files, exception payload, or raw logs.
    import pandas as pd
    for chunk in pd.read_csv(TRAIN, usecols=[UID, "Report"], dtype=str, chunksize=256):
        matches = chunk.loc[chunk[UID] == uid, "Report"]
        if len(matches):
            value = matches.iloc[0]
            if not isinstance(value, str) or not value.strip():
                raise GuardError("REPORT_UNAVAILABLE")
            return value
    raise GuardError("REPORT_UNAVAILABLE")


def exam(work, row, flight, ledger, reader=invoke_reader):
    code = row["code"]
    for _ in range(MAX_ATTEMPTS):
        status, attempt = ledger.reserve(code)
        if status != "reserved":
            if status == "complete":
                ledger.export_receipt(code)
            return status
        started = time.monotonic()
        usage, payload, reason = None, None, None
        try:
            labels = {t: {"p": round(float(flight.loc[row["uid"], t]), 3),
                          "conf": round(float(flight.loc[row["uid"], t + "__conf"]), 3)} for t in TARGETS}
            prompt = prompt_f(code, private_report(row["uid"]), labels, retry=attempt > 1)
            usage, answer, reason = reader(reader_command(code, work), prompt)
            if usage is not None and reason is None:
                try:
                    payload = strict_parse(answer, code)
                except GuardError as exc:
                    reason = str(exc)
            del prompt
        except BaseException as exc:
            # Never repr/str arbitrary errors: they can include reports or label rows.
            reason = str(exc) if isinstance(exc, GuardError) else "LOCAL_EXCEPTION_UNKNOWN"
        ledger.finish(code, attempt, usage, time.monotonic()-started, payload, reason)
        progress(ledger)
        if payload is not None:
            return "complete"
        if usage is None or reason == "FORBIDDEN_TOOL_EVENT":
            return "halt"
    return "exhausted"


def progress(ledger):
    try:
        atomic(ledger.directory / "progress.json", {**ledger.summary(), "updated_unix": time.time()})
    except (OSError, GuardError):
        pass  # durable ledger is authoritative; heartbeat/export failures do not repeat paid calls.


def review_gate(work, review_path, scope):
    if not review_path:
        raise GuardError("INDEPENDENT_REVIEW_REQUIRED")
    path = Path(review_path).resolve()
    if not path.is_relative_to(work.resolve()):
        raise GuardError("REVIEW_OUTSIDE_WORK")
    rec = load(path)
    expected = {"verdict": "PASS", "scope": scope, "version": VERSION,
                "code_sha256": sha(Path(__file__)), "registration_sha256": sha(work / "registration.json"),
                "cpu_smoke_sha256": sha(work / "cpu-smoke.json"),
                "cli_preflight_sha256": sha(work / "cli-preflight.json")}
    if any(rec.get(k) != v for k, v in expected.items()):
        raise GuardError("REVIEW_IDENTITY")
    if not rec.get("reviewer") or rec["reviewer"] in ("l3_builder", "/root/l3_builder"):
        raise GuardError("INDEPENDENT_REVIEWER")
    report = (work / rec["report"]).resolve()
    if not report.is_relative_to(work.resolve()) or sha(report) != rec.get("report_sha256"):
        raise GuardError("REVIEW_REPORT_IDENTITY")
    smoke = load(work / "cpu-smoke.json")
    if smoke.get("registration_sha256") != expected["registration_sha256"] or not smoke.get("passed"):
        raise GuardError("SMOKE_IDENTITY")
    return rec


def run(work=WORK, scope="pilot", lanes=2, ceiling=PILOT_CEILING, review_path=None, confirm_full=False, limit=None):
    cpu_only()
    reg = verify_registration(work)
    if not 1 <= lanes <= 6 or ceiling < RESERVATION:
        raise GuardError("RUN_LIMITS")
    if scope == "pilot" and (lanes > reg["pilot_lanes"] or ceiling != reg["pilot_token_ceiling"]):
        raise GuardError("PILOT_LIMITS")
    if limit is not None and (scope != "pilot" or limit != 2):
        raise GuardError("ONLY_FIRST_TWO_PILOT_SUBSET_ALLOWED")
    if scope == "full" and (not confirm_full or not reg["full_eligible"]):
        raise GuardError("FULL_SCOPE_NOT_AUTHORIZED_OR_READY")
    cli_preflight(work)
    review_gate(work, review_path, scope)
    validate(work, scope)
    rows, flight, _, _ = inputs()
    selected = {x["code"] for x in (reg["pilot"][:limit] if limit else reg["pilot"])} if scope == "pilot" else {x["code"] for x in rows}
    destination = work / scope
    destination.mkdir(parents=True, exist_ok=True)
    with run_lock(work):
        if scope == "pilot" and (work / "full/ledger.sqlite3").exists():
            raise GuardError("PILOT_FROZEN_BY_FULL_SCOPE")
        ledger = Ledger(destination, sha(work / "registration.json"), ceiling, scope)
        ledger.startup_guard()
        # Full scope can reuse pilot completions, preventing double payment and keeping the original receipt.
        reused = 0
        if scope == "full" and (work / "pilot/ledger.sqlite3").exists():
            pilot = Ledger(work / "pilot", sha(work / "registration.json"), PILOT_CEILING, "pilot")
            reused = ledger.reuse_pilot(pilot, work)
        todo = [r for r in rows if r["code"] in selected]
        started = time.time()
        session_id = ledger.start_session(lanes)
        atomic(destination / "run-session.json", {"started": started, "lanes": lanes, "scope": scope,
               "registration_sha256": sha(work / "registration.json"), "reused_pilot": reused,
               "selected": len(selected), "code_sha256": sha(Path(__file__))})
        stop = threading.Event()
        def one(row):
            if stop.is_set():
                return "not_started"
            result = exam(work, row, flight, ledger)
            if result in ("halt", "budget", "blocked"):
                stop.set()
            return result
        counts = {}
        try:
            with ThreadPoolExecutor(max_workers=lanes) as pool:
                for state in pool.map(one, todo):
                    counts[state] = counts.get(state, 0) + 1
        finally:
            elapsed = time.time() - started
            ledger.finish_session(session_id, elapsed)
        summary = {**ledger.summary(), "session_seconds": elapsed, "lanes": lanes,
                   "session_states": counts, "reused_pilot": reused, "selected": len(selected),
                   "session_finished": time.time()}
        atomic(destination / "run-summary.json", summary)
        progress(ledger)
    return summary


def cpu_smoke(work=WORK):
    """Two real image studies; synthetic report/answers only, never live calls."""
    reg = verify_registration(work)
    cli_preflight(work)
    codes = {r["code"] for r in reg["pilot"][:2]}
    rows = [r for r in reg["rows"] if r["code"] in codes]
    c = corpus()
    recs = [prepare_one(work, c, r) for r in rows]
    smoke_dir = work / "cpu-smoke-engine"
    # Stable smoke IDs make reruns idempotent without deleting any artifacts.
    ledger = Ledger(smoke_dir, sha(work / "registration.json"), 500_000, "synthetic")
    for row in rows:
        code = row["code"]
        findings = {t: {"p": .5, "severity": "uncertain"} for t in TARGETS}
        labels = {t: {"p": .5, "conf": .5} for t in TARGETS}
        prompt = prompt_f(code, "SYNTHETIC CPU SMOKE; no real report accessed.", labels)
        if "Official definitions:" not in prompt:
            raise GuardError("SMOKE_PROMPT")
        answer = json.dumps({code: findings})
        stream = "\n".join([json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": answer}}),
                             json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 30,
                                                                        "cached_input_tokens": 0}})])
        usage, parsed, err = parse_events(stream)
        payload = strict_parse(parsed, code)
        state, attempt = ledger.reserve(code)
        if state == "reserved":
            ledger.finish(code, attempt, usage, 0., payload, err)
        elif state != "complete":
            raise GuardError("SMOKE_LEDGER")
        ledger.export_receipt(code)
        if ledger.reserve(code)[0] != "complete":
            raise GuardError("SMOKE_IDEMPOTENCY")
    rec = {"version": VERSION, "passed": True, "real_studies": len(recs), "live_reader_calls": 0,
           "reports_accessed": 0, "synthetic_payload": True, "cuda_visible_devices_empty": True,
           "registration_sha256": sha(work / "registration.json"), "code_sha256": sha(Path(__file__)),
           "prepared_hashes": [sha(work / "prepared" / (r["code"] + ".json")) for r in rows],
           "checked": ["real 15-slot montage", "F prompt synthetic report", "strict schema", "event token parsing",
                       "durable reservation", "completion receipt", "idempotent replay"],
           "not_checked": ["live Astra calls", "transport telemetry", "parallel reader throughput"]}
    atomic(work / "cpu-smoke.json", rec)
    return {k: v for k, v in rec.items() if k not in ("prepared_hashes",)}


def dry_run(work=WORK):
    reg = verify_registration(work)
    validate(work, "pilot")
    result = {"version": VERSION, "registration_sha256": sha(work / "registration.json"),
              "pilot_count": 200, "lanes": 2, "pilot_token_ceiling": PILOT_CEILING,
              "reservation_tokens": RESERVATION, "max_attempts": MAX_ATTEMPTS,
              "calls_made": 0, "full_started": False, "gpu_admit": False,
              "independent_review_required": True, "full_eligible": reg["full_eligible"]}
    atomic(work / "dry-run.json", result)
    return result


def report(work=WORK, scope="pilot", lanes_projection=4, input_usd_million=None, output_usd_million=None):
    # Reports read finished artifacts only; the same lock prevents reading an active run.
    with run_lock(work):
        return finished_report(work, scope, lanes_projection, input_usd_million, output_usd_million)


def finished_report(work, scope, lanes_projection, input_usd_million, output_usd_million):
    import numpy as np
    from scipy.stats import spearmanr
    reg = verify_registration(work)
    if lanes_projection < 1:
        raise GuardError("PROJECTION_LANES")
    directory = work / scope
    if not (directory / "ledger.sqlite3").exists():
        raise GuardError("NO_READER_RESULTS")
    with sqlite3.connect(directory / "ledger.sqlite3") as con:
        ceiling = int(con.execute("SELECT value FROM meta WHERE key='ceiling'").fetchone()[0])
    ledger = Ledger(directory, sha(work / "registration.json"), ceiling, scope)
    with ledger.connect() as con:
        complete = [dict(r) for r in con.execute("SELECT * FROM attempts WHERE state='complete' ORDER BY code")]
        attempts = [dict(r) for r in con.execute("SELECT * FROM attempts ORDER BY code,attempt")]
    if not complete:
        raise GuardError("NO_COMPLETED_EXAMS")
    _, flight, _, _ = inputs()
    by_code = {r["code"]: r for r in reg["rows"]}
    x = np.asarray([[json.loads(r["payload"])[t]["p"] for t in TARGETS] for r in complete])
    y = np.asarray([[float(flight.loc[by_code[r["code"]]["uid"], t]) for t in TARGETS] for r in complete])
    per_target = {}
    for i, target in enumerate(TARGETS):
        a, b = x[:, i], y[:, i]
        corr = float(spearmanr(a, b).statistic) if len(set(a)) > 1 and len(set(b)) > 1 else None
        per_target[target] = {"n": len(a), "mae": float(np.mean(np.abs(a-b))),
                              "binary_agreement_at_0_5": float(np.mean((a >= .5) == (b >= .5))),
                              "spearman": corr, "mean_probability_change": float(np.mean(a-b))}
    elapsed = np.asarray([r["elapsed"] for r in attempts if r["elapsed"] is not None], float)
    summary = ledger.summary()
    spent = summary["total_tokens"]
    n = len(complete)
    total = reg["counts"]["non_gold"]
    tok_per = spent/n
    seconds_per = float(elapsed.sum()/n)
    price = None
    if input_usd_million is not None and output_usd_million is not None:
        if min(input_usd_million, output_usd_million) < 0:
            raise GuardError("NEGATIVE_PRICE")
        price = (summary["input_tokens"]*input_usd_million + summary["output_tokens"]*output_usd_million)/1e6/n*total
    session = load(directory / "run-summary.json") if (directory / "run-summary.json").exists() else {}
    out = {**summary, "completed_exams": n, "requested_pilot": 200 if scope == "pilot" else None,
           "complete_pilot": scope == "pilot" and n == 200, "wall_seconds_latest_session": session.get("session_seconds"),
           "wall_seconds_per_completed_exam": summary["wall_seconds_all_sessions"]/n,
           "wall_accounting_complete": summary["unfinished_sessions"] == 0,
           "reader_seconds_per_completed_exam_including_failures": seconds_per,
           "reader_seconds_p50_per_attempt": float(np.median(elapsed)),
           "reader_seconds_p95_per_attempt": float(np.percentile(elapsed, 95)),
           "tokens_per_completed_exam_including_failures": tok_per,
           "failure_attempt_rate": sum(r["state"] != "complete" for r in attempts)/len(attempts),
           "unresolved_exam_rate": (summary["attempted_exams"]-n)/summary["attempted_exams"],
           "per_target_flight_agreement": per_target,
           "full_projection": {"exams": total, "tokens": tok_per*total,
               "usage_accounting_complete": not summary["unknown_requires_manual_reconciliation"],
               "lanes": lanes_projection, "ideal_hours": seconds_per*total/lanes_projection/3600,
               "remaining_exams_after_pilot": total-n, "remaining_ideal_hours": seconds_per*(total-n)/lanes_projection/3600,
               "usd": price, "usd_formula": "(projected_input_tokens * input_USD_per_M + projected_output_tokens * output_USD_per_M)/1e6",
               "projected_input_tokens": summary["input_tokens"]/n*total,
               "projected_output_tokens": summary["output_tokens"]/n*total,
               "caveat": "Ideal linear lane scaling is not measured; stratified pilot overrepresents difficult/positive cases. Unknown/inflight usage is excluded from totals and invalidates cost projection."},
           "interpretation": reg["interpretation"], "no_gold_scoring": True}
    atomic(directory / "report.json", out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["register", "prepare", "validate", "cpu-smoke", "dry-run", "run-pilot", "full", "report"])
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--token-ceiling", type=int, default=PILOT_CEILING)
    ap.add_argument("--review", type=Path)
    ap.add_argument("--confirm-full", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--input-usd-per-million", type=float)
    ap.add_argument("--output-usd-per-million", type=float)
    args = ap.parse_args(argv)
    try:
        cpu_only()
        work = fixed_work(args.work)
        if args.command == "register":
            result = register(work)
        elif args.command == "prepare":
            result = prepare(work, args.scope, args.limit)
        elif args.command == "validate":
            result = validate(work, args.scope)
        elif args.command == "cpu-smoke":
            result = cpu_smoke(work)
        elif args.command == "dry-run":
            result = dry_run(work)
        elif args.command in ("run-pilot", "full"):
            result = run(work, "pilot" if args.command == "run-pilot" else "full", args.lanes,
                         args.token_ceiling, args.review, args.confirm_full, args.limit)
        else:
            result = report(work, args.scope, args.lanes, args.input_usd_per_million, args.output_usd_per_million)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except BaseException as exc:
        # No traceback or arbitrary error message may expose private input/provider text.
        reason = str(exc) if isinstance(exc, GuardError) else type(exc).__name__.upper()
        print(json.dumps({"version": VERSION, "status": "BLOCKED", "reason": reason}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
