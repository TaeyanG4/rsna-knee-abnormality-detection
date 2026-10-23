"""v1.0.0: lossless b3 column projection to the strict training CSV contract."""
import argparse
import csv
import io
from pathlib import Path
from tools import l1_rco_k1_failure_v1 as producer


def convert(source, output):
    k=producer.k
    with Path(source).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        k.need(reader.fieldnames==['StudyInstanceUID','source_oof_fold','target','weight','agreement','primary_error'],
               'expected exact b3 schema')
        rows=list(reader)
    k.need(all(None not in r and all(v is not None for v in r.values())
               and r['source_oof_fold'] in ('1','2','3') for r in rows), 'b3 training folds only')
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=['study_uid','target','weight'])
    writer.writeheader()
    writer.writerows(dict(study_uid=r['StudyInstanceUID'],target=r['target'],weight=r['weight']) for r in rows)
    output=Path(output);raw=stream.getvalue().encode()
    k.need(not output.exists() or output.read_bytes()==raw,'canonical weights already differ')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(raw)
    parent=k.read(k.PARENT)
    values=producer.weights.load_weights(output,parent['train'],parent['holdout'],k.s.LABELS)
    receipt=dict(status='PASS',version='1.0.0',source_sha256=k.sha(source),canonical_sha256=k.sha(output),
                 rows=len(values),studies=len({u for u,t in values}),holdout_rows=0,
                 transform='rename StudyInstanceUID to study_uid; project target/weight unchanged; preserve row order',
                 clean_oof=False,exposure='EXPOSED_EXPLORATORY')
    k.write(output.with_suffix('.provenance.json'),receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();import json;print(json.dumps(convert(a.source,a.output)))
