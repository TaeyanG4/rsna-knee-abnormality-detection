"""Synthetic data boundary for the real H-B-146 run/prediction/AUC readout.

No decision/schema/evaluator is mocked. All files and numeric values are test
inputs; these fixtures are never scientific evidence or production artifacts.
"""
import csv
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tools import hb146_readout_v111 as r
from tools import hb146_screen_v111 as screen
from tools import oai_moaks_panel as panel


def make_readout(work, winner='global', ids=None, engineering=False):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    ids = list(ids) if ids is not None else [f'synthetic_{i}' for i in range(24)]
    gold = [f'gold_{i}' for i in range(24)]
    oai = [f'oai_{i}' for i in range(24)]
    rosters = dict(val=ids, gold=gold, oai=oai)
    def labels(n):
        return ((np.arange(n)[:, None] + np.arange(12)[None, :]) % 2).astype(np.float32)
    def table(path, roster, values):
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([r.pl.UID, *r.pl.TARGETS])
            writer.writerows([u, *row] for u, row in zip(roster, values))
    registration = dict(val=ids, gold=gold, synthetic_fixture=True)
    r.pl.write(work/'registration.json', registration)
    r.pl.write(work/'prepare.json', dict(synthetic_fixture=True))
    components = [dict(component_id=u, members=[u]) for u in ids+gold]
    r.pl.write(work/'components.json', components)
    r.pl.write(work/'manifest.json', dict(synthetic_fixture=True))
    teacher = work/'synthetic-teacher.csv'
    table(teacher, ids, np.full((len(ids),12), .5))
    table(work/'synthetic-gold.csv', gold, labels(len(gold)))
    admission = dict(eligible=True, binding_sha256=r.pl.sha(work/'manifest.json'))
    for side in ('control', 'global', 'fill'):
        for seed in (42,43):
            folder = work/'fits'/f'{side}-s{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'final.pt').write_bytes(f'SYNTHETIC checkpoint {side} {seed}'.encode())
            identity = dict(arm=side, seed=seed, epochs=20, runtime={'fixture':'CPU'},
                prepare_sha256=r.pl.sha(work/'prepare.json'),
                registration_sha256=r.pl.sha(work/'registration.json'))
            ck_hash = r.pl.sha(folder/'final.pt')
            r.pl.write(folder/'summary.json',dict(identity=identity,smoke=False,checkpoint_sha256=ck_hash))
            for name, roster in rosters.items():
                values = .1 + .8*labels(len(roster)) if side == winner else np.full((len(roster),12), .5)
                path = folder/(name+'.csv')
                table(path, roster, values)
                receipt = dict(identity=identity, engineering_only=False, k=36, tta=True,
                               checkpoint_sha256=ck_hash, csv_sha256=r.pl.sha(path))
                r.pl.write(path.with_suffix('.receipt.json'),receipt)
                if name == 'oai':
                    r.pl.write(folder/'oai-admission.json',dict(admission,csv_sha256=receipt['csv_sha256'],
                               cache_sha256=r.pl.sha(work/'manifest.json')))
    y = labels(len(ids))
    ctx = r.lens._base.Context(ids,{name:(y,np.ones_like(y,dtype=bool),1) for name in r.lens.RULERS},components)
    ctx.bindings = {'synthetic-rulers':r.pl.sha(work/'components.json')}
    provenance = dict(status=r.pl.STATUS, limitations=r.pl.LIMITATIONS, exposed_exploratory=True,
                      teacher_csv=str(teacher))
    with patch.object(screen,'prepared'), patch.object(r.pl,'validate_labels',return_value=(None,None,provenance)), \
         patch.object(r.pl,'registration',return_value=registration), \
         patch.object(r.lens._base,'make_context',return_value=ctx), \
         patch.object(r.lens._base,'COMPONENTS',work/'components.json'), \
         patch.object(r.pl,'GOLD',work/'synthetic-gold.csv'), \
         patch.object(screen,'panel_admit',return_value=(oai,admission)), \
         patch.object(panel,'outcomes',return_value=labels(len(oai))), \
         patch.object(panel,'CACHES',{'64':work}):
        result = r.run(work,engineering=engineering,exposed_exploratory=True)
    return Path(result['readout'])
