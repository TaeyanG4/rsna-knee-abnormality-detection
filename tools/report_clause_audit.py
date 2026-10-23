"""Prepare a frozen, local-only blinded report audit. No training or label edits.

Usage: .venv/Scripts/python.exe tools/report_clause_audit.py prepare --out <new-dir>
Only prepare is provided: adjudication must precede access to the existing labels.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
RULES = {
    'mcl_partial': {
        'structure': r'\bmcl\b|\blcm\b|collateral.{0,30}medial|medial.{0,30}collateral|colateral.{0,30}(?:medial|intern)|colateral medial|ligament.{0,30}intern|innenband|медиальн.{0,40}(?:коллатеральн|боков)|내측.{0,10}측부|内侧副',
        'clause': r'partial|parcial|partiel|partiell|par[țt]ial|частич|부분|部分|high.grade|hochgrad|grade\s*[23]|grado\s*[23]|grau\s*[23]|grad(?:ul)?\s*[23]|rupt|tear|riss|lesion|entorse|sprain',
        'window': 200,
        'targets': ['MCL'],
    },
    'meniscus_morphology': {
        'structure': r'menisc|menisk|мениск|반월|半月',
        'clause': r'truncat|diminut|displac|fragment|bucket.{0,12}handle|flap|anse.{0,12}seau|asa.{0,12}cubo|toart|korbhenkel|luxiert|amput|desplaz|desloc|смещ|фрагмент|ручк.{0,15}лейк|절단|전위|移位|桶柄',
        'window': 300,
        'targets': ['Medial Meniscus', 'Lateral Meniscus'],
    },
    'fracture_line': {
        'structure': r'fractur|fraktur|fractuur|перелом|골절|骨折',
        'clause': r'.',
        'window': 300,
        'targets': ['Fracture'],
    },
}
OFFICIAL = {
    'MCL': 'A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative.',
    'Medial Meniscus': 'Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.',
    'Lateral Meniscus': 'The same criteria applied to the lateral meniscus.',
    'Fracture': 'An acute cortical break or fracture line.',
}

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def normalise(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s.casefold()) if not unicodedata.combining(c))

def matched(report: str, rule: dict) -> bool:
    text = normalise(report)
    for m in re.finditer(rule['structure'], text):
        if re.search(rule['clause'], text[max(0,m.start()-rule['window']):m.end()+rule['window']], re.S):
            return True
    return False

def write_json(path: Path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def prepare(dest: Path, seed: int = 20261007):
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    # Register the retrieval/sampling protocol before reading labels or reports.
    protocol = {
        'schema': 'report-clause-audit-v1', 'plan': 'H-C-01', 'seed': seed,
        'recorded_at': datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'rules': RULES, 'sample_per_clause': 24, 'random_controls': 24,
        'selection': 'Sort eligible UIDs by sha256(seed|stratum|UID); first24 per clause, overlapping clause samples allowed; controls from remaining non-Gold UIDs. Report-only retrieval, not label-based retrieval.',
        'official_source': 'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343',
        'official_definitions': OFFICIAL,
        'limits': 'English/Latin/Russian/East-Asian keyword retrieval is incomplete; targeted subset is not a prevalence estimate or a clinical validation set. Report is not image truth. Do not assume unknown grade/acuity/laterality or postoperative morphology is a qualifying tear.',
        'review_protocol': 'Two independent passes read only the blinded packet and official criteria, never current labels, prompt text or Gold/Public scores. Record exact local evidence spans, status pos/neg/unc/na, negation, severity, laterality/compartment, ambiguity and reasoning. Compare to raw v1.1 only after both outputs are frozen and disagreements adjudicated.',
        'decision': 'Confirmed evidence-supported extraction contradictions justify a separate prompt draft/counterexample suite, not automatic relabelling/training/submission or efficacy claims. Current frozen v1.1 remains unchanged.',
        'tool_sha256': sha(Path(__file__).read_bytes()),
    }
    write_json(dest/'protocol.json', protocol)
    rows=list(csv.DictReader((ROOT/'storage/competition/train.csv').open(encoding='utf-8',newline='')))
    assert len(rows)==4407
    gold={r['StudyInstanceUID'] for r in rows if any(r[t].strip() not in ('','NaN','nan') for t in TARGETS)}
    assert len(gold)==58
    source={r['StudyInstanceUID']:r['Report'] for r in rows}
    batch_records={}
    batch_sha={}
    for p in sorted((ROOT/'.local/relabel/batches').glob('b[0-9][0-9].jsonl')):
        batch_sha[p.name]=sha(p.read_bytes())
        for lineno,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
            r=json.loads(line);uid=r['uid'];assert uid not in batch_records
            assert r['report']==source[uid], f'Original/batch report mismatch in {p.name}:{lineno}'
            batch_records[uid]={'batch':p.name,'batch_line':lineno,'input_record_sha256':sha(line.encode()),'report_sha256':sha(r['report'].encode())}
    assert set(batch_records)==set(source)
    eligible={uid:s for uid,s in source.items() if uid not in gold}
    def ordered(ids, stratum):
        return sorted(ids,key=lambda u:sha(f'{seed}|{stratum}|{u}'.encode()))
    groups={k:[u for u,s in eligible.items() if matched(s,v)] for k,v in RULES.items()}
    selected={k:ordered(ids,k)[:24] for k,ids in groups.items()}
    selected_union=set().union(*(set(v) for v in selected.values()))
    selected['random_control']=ordered(set(eligible)-selected_union,'random_control')[:24]
    selected_union.update(selected['random_control'])
    packets=[];bindings=[]
    for n,uid in enumerate(ordered(selected_union,'packet_order'),1):
        strata=[k for k,v in selected.items() if uid in v]
        targets=list(OFFICIAL) if 'random_control' in strata else list(dict.fromkeys(t for k in strata for t in RULES[k]['targets']))
        case=f'C{n:03d}'
        packets.append({'case_id':case,'report':source[uid],'targets':targets})
        bindings.append({'case_id':case,'uid':uid,'strata':strata,'targets':targets,**batch_records[uid]})
    for name,records in [('blinded_packet.jsonl',packets),('bindings.jsonl',bindings)]:
        with (dest/name).open('x',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_json(dest/'review_instructions.json',{'definitions':OFFICIAL,'review_protocol':protocol['review_protocol'],'limits':protocol['limits'],'schema_per_case':{'case_id':'C001','findings':{'TARGET':{'state':'pos|neg|unc|na','evidence':['exact report substring, or [] if unmentioned'],'reason':'brief report-bound reason, including grade/negation/laterality/uncertainty as applicable'}}}})
    manifest={'protocol_sha256':sha((dest/'protocol.json').read_bytes()),'train_sha256':sha((ROOT/'storage/competition/train.csv').read_bytes()),'packet_sha256':sha((dest/'blinded_packet.jsonl').read_bytes()),'bindings_sha256':sha((dest/'bindings.jsonl').read_bytes()),'gold_excluded':len(gold),'eligible_reports':len(eligible),'retrieved_counts':{k:len(v) for k,v in groups.items()},'sampled_counts':{k:len(v) for k,v in selected.items()},'unique_reports':len(packets),'judgments':sum(len(r['targets']) for r in packets),'input_batches_sha256':batch_sha,'frozen_output_batches_sha256':{p.name:sha(p.read_bytes()) for p in sorted((ROOT/'.local/relabel/out').glob('b[0-9][0-9].jsonl'))},'no_output_values_read':True}
    assert all(r['uid'] not in gold for r in bindings)
    assert len(set(r['uid'] for r in bindings))==len(bindings)
    write_json(dest/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['protocol_sha256','packet_sha256','gold_excluded','eligible_reports','retrieved_counts','sampled_counts','unique_reports','judgments']},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare'])
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();prepare(a.out)
