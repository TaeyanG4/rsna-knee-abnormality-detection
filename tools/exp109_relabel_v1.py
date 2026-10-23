"""H-B-28: build the definition-aligned label set relabel_v1 from the LLM labeller outputs, and run the one-time Gold58 diagnostic.

  python tools/exp109_relabel_v1.py build       # all 25 batches b00..b24 -> storage/labels/derived_relabel/relabel_v1.csv + provenance.json
  python tools/exp109_relabel_v1.py gold-diag   # ONCE, after build: per-target AUC on Gold58 (relabel_v1 vs Dread vs flight); refuses to run twice

Frozen conversion (docs/runbooks/relabel-prompt-v1_1.md): p clipped to [0.02, 0.98]; s == 'na' -> 0.05, except Synovitis 'na' -> 0.6 * Effusion p
of the same report (then clipped). Gold rows are removed from the training file. The pilot (prompt v1) is not used. No report text is written.
"""
import argparse, hashlib, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UID = 'StudyInstanceUID'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
S = {'pos', 'neg', 'unc', 'na'}
BATCHES = [f'b{i:02d}' for i in range(25)]
PROMPT = 'docs/runbooks/relabel-prompt-v1_1.md'
IN_DIR, OUT_DIR = '.local/relabel/batches', '.local/relabel/out'
DEST = 'storage/labels/derived_relabel'
GOLD_SOURCE = 'storage/competition/train.csv'
DIAG = 'artifacts/experiments/EXP109_RELABEL_V1_GOLD_DIAG/diag.json'
REFS = {'dread': 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv',
        'flight': 'storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clip(p):
    return min(max(float(p), 0.02), 0.98)


def convert(f):
    """One labeller record's 'f' dict -> 12 training labels (frozen rule)."""
    if set(f) != set(T):
        raise ValueError(f'keys differ: {sorted(set(f) ^ set(T))}')
    out = {}
    for t in T:
        s, p = f[t]['s'], float(f[t]['p'])
        if s not in S or not 0.0 <= p <= 1.0:
            raise ValueError(f'invalid {t}: {f[t]}')
        if s == 'na':
            p = 0.6 * float(f['Effusion']['p']) if t == 'Synovitis' else 0.05
        out[t] = clip(p)
    return out


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def gold_uids(root):
    import pandas as pd
    df = pd.read_csv(Path(root) / GOLD_SOURCE, dtype={UID: str}, usecols=[UID, 'ACL'])
    return set(df.loc[df['ACL'].notna(), UID].astype(str))


def collect(root, batches=BATCHES):
    """uid -> raw 'f' dict over all batches; every output must equal its input uids in order."""
    root, recs, files = Path(root), {}, {}
    for b in batches:
        inp, out = root / IN_DIR / f'{b}.jsonl', root / OUT_DIR / f'{b}.jsonl'
        if not out.exists():
            raise SystemExit(f'missing labeller output {out}')
        want = [r['uid'] for r in read_jsonl(inp)]
        got = read_jsonl(out)
        if [r['uid'] for r in got] != want:
            raise SystemExit(f'{b}: output uids differ from the input (count {len(got)} vs {len(want)})')
        for r in got:
            if r['uid'] in recs:
                raise SystemExit(f'duplicate uid across batches: {r["uid"]}')
            recs[r['uid']] = r['f']
        files[b] = dict(input_sha256=sha256(inp), output_sha256=sha256(out), rows=len(got))
    return recs, files


def build(root=ROOT, batches=BATCHES):
    import pandas as pd
    root = Path(root)
    dest = root / DEST
    csv, prov = dest / 'relabel_v1.csv', dest / 'provenance.json'
    if csv.exists():
        raise SystemExit(f'refusing to overwrite {csv}')
    recs, files = collect(root, batches)
    gold = gold_uids(root)
    rows = [{UID: u, **convert(f)} for u, f in recs.items() if u not in gold]
    df = pd.DataFrame(rows, columns=[UID] + T).sort_values(UID)
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False, float_format='%.4f', lineterminator='\n')
    states = {t: {s: sum(1 for f in recs.values() if f[t]['s'] == s) for s in sorted(S)} for t in T}
    meta = dict(name='relabel_v1', plan_item='H-B-28', built_at=time.strftime('%Y-%m-%d %H:%M %z'), prompt=PROMPT, prompt_sha256=sha256(root / PROMPT),
                labeller='Claude Code sub-agents (model opus), one per batch of <=177 reports',
                conversion='p clipped to [0.02, 0.98]; na -> 0.05; Synovitis na -> 0.6 x Effusion p (clipped)',
                labelled_reports=len(recs), gold_rows_removed=len(set(recs) & gold), rows=len(df), csv_sha256=sha256(csv), batches=files,
                state_counts_all_reports=states, report_text_included=False)
    prov.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: meta[k] for k in ('rows', 'labelled_reports', 'gold_rows_removed', 'csv_sha256', 'prompt_sha256')}, indent=1))
    return meta


def auc(y, p):
    import numpy as np
    y, p = np.asarray(y, float), np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def gold_diag(root=ROOT):
    """One-time extraction diagnostic on Gold58 (pre-registered: no prompt change is allowed because of it)."""
    import numpy as np
    import pandas as pd
    root = Path(root)
    out = root / DIAG
    if out.exists():
        raise SystemExit(f'the Gold58 diagnostic already ran: {out}')
    if not (root / DEST / 'relabel_v1.csv').exists():
        raise SystemExit('run build first')
    recs, _ = collect(root)
    g = pd.read_csv(root / GOLD_SOURCE, dtype={UID: str}).set_index(UID)
    g = g[g['ACL'].notna()][T]
    ids = [u for u in g.index if u in recs]
    new = pd.DataFrame({u: convert(recs[u]) for u in ids}).T
    refs = {k: pd.read_csv(root / v, dtype={UID: str}).set_index(UID) for k, v in REFS.items()}
    res = {}
    for t in T:
        y = g.loc[ids, t].values
        row = dict(positives=int(y.sum()), relabel_v1=auc(y, new.loc[ids, t].values))
        for k, r in refs.items():
            col = t if t in r.columns else t + '_soft'
            have = [u for u in ids if u in r.index]
            row[k] = auc(g.loc[have, t].values, r.loc[have, col].values) if have and col in r.columns else None
            row[k + '_n'] = len(have)
        res[t] = row
    macro = {k: (float(np.mean(v)) if (v := [r[k] for r in res.values() if r[k] is not None]) else None) for k in ('relabel_v1', *REFS)}
    rec = dict(ran_at=time.strftime('%Y-%m-%d %H:%M %z'), n_gold_labelled=len(ids), per_target=res, macro=macro,
               note='Gold58 is exposed (anchor epoch choice, legacy gate) and tiny (9 MCL positives); diagnostic only, never a selection gate')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(n=len(ids), macro=macro), indent=1))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['build', 'gold-diag'])
    a = ap.parse_args(argv)
    build() if a.cmd == 'build' else gold_diag()


if __name__ == '__main__':
    main()
