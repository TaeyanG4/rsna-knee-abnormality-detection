"""R2 v1.0.0 adapter: frozen HB116 native workers, authenticated R+ journals.

2026-10-14: replace only C85-R7's R slot; never relabel R+ as original R weights.
"""
from pathlib import Path
from tools import hb116_vnext_release as h

DATASET='taeyangg4/knee-hb113-rplus-weights-v2'
def registration(runtime=None):
    path=Path(runtime)/'sealed-registration.json' if runtime else Path(__file__).resolve().parents[1]/'sealed-registration.json'
    if not path.exists() and runtime is None:
        path=h.REG/'consults/astra3-registration-draft.json'
    h.pin(path,{'sha256':h.DRAFT_SHA})
    return h.read(path)

def identity(reg):
    return dict(version=h.VERSION,members={f"R2-s{e['seed']}":e['sha256'] for e in reg['P']['Rplus2']['members']})

def preflight(runtime):
    reg=registration(runtime)
    h.pin(h.mount(DATASET,'manifest.json'),reg['P']['Rplus2']['manifest'])
    entries=[dict(e,id=f"R2-s{e['seed']}",resolved=str(h.mount(DATASET,Path(e['path']).name))) for e in reg['P']['Rplus2']['members']]
    h.need({e['id'] for e in entries}=={'R2-s43','R2-s44'},'sealed R2 members')
    for e in entries: h.pin(e['resolved'],e)
    return reg,entries

def run(runtime, competition, output, device='cuda'):
    runtime,output=Path(runtime),Path(output)
    reg,entries=preflight(runtime)
    paths=dict(legacy=str(runtime/'legacy'),public_source=str(runtime/'public-serving.py'),
               own_manifest=str(runtime/'r2-own-manifest.json'))
    # Identical native worker/FP32 inference as vNext; only the selected members differ.
    return h.run_phases(reg,entries,paths,Path(competition),output,device=device)

def recover(output, ids, reg=None):
    import numpy as np
    reg=registration() if reg is None else reg
    expected=identity(reg)
    values={k:np.full((len(ids),12),np.nan,np.float32) for k in expected['members']}
    assigned=set()
    for worker in sorted((Path(output)/'NATIVE').glob('worker*')):
        if not (worker/'journal-identity.json').exists(): continue
        part=h.c.v5q.validate_ids(h.read(worker/'ids.json'))
        h.need(set(part)<=set(ids) and not assigned.intersection(part),'R2 journal cohort')
        assigned.update(part)
        raw,fails=h.c.read_journal(worker,part,list(values),expected)
        for k in values:
            for j,uid in enumerate(part):
                if uid not in fails[k]: values[k][ids.index(uid)]=raw[k][j]
    result=h.mean_survivors([values['R2-s43'],values['R2-s44']],dtype='float32')
    failed=~np.isfinite(result).all(axis=1)
    failures={uid:'MissingPrediction: both R2 seeds' for uid,bad in zip(ids,failed) if bad}
    result[failed]=.5
    return result,failures,dict(expected,exposure='EXPOSED',clean_oof=False)
