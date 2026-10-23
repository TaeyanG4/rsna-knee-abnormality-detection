"""Explicit historical contrast adapters for multi_lens_readout 1.0.0 (CPU only)."""
from pathlib import Path
import json
import re
import time
from tools.multi_lens_readout import ROOT, VERSION, DRAWS, SEED, sha, write, run_spec


def old(d):
    for key in ('registered_gate', 'registered_exp181_gate', 'verdict', 'decision', 'status'):
        if key in d:
            v = d[key]
            if isinstance(v, str): return v
            if isinstance(v, dict):
                for k in ('status', 'verdict', 'decision'): 
                    if k in v: return str(v[k])
                if 'pass' in v: return 'PASS' if v['pass'] else 'HOLD'
                if 'pass_' in v: return 'PASS' if v['pass_'] else 'HOLD'
    if 'tiers' in d:
        return '; '.join(k + ': ' + old(v) for k,v in d['tiers'].items())
    if 'pass' in d: return 'PASS' if d['pass'] else 'HOLD'
    return 'UNRECORDED'


def blend(*parts):
    return {'blend': [{'files': f, 'weight': w} for f, w in parts]}


def held_verdict(value):
    # THRESHOLDS_MET must never match the substring HOLD.
    return bool(re.search(r'(?<![A-Z])(?:HOLD|STOP|CLOSE|EXCLUDE|REVIEW|KEEP_BASELINE|FAIL|THRESHOLDS_NOT_MET|INCONCLUSIVE)(?![A-Z])', value.upper()))


class Inventory:
    def __init__(self, root, screens):
        self.root, self.hashes = root, {}
        paths = set(root.glob('artifacts/experiments/**/oof*.csv'))
        paths.update(root.glob('artifacts/experiments/EXP161_K_SWEEP2/outputs/exp161-pooled/*.csv'))
        paths.update(root.glob('artifacts/experiments/EXP161_K_SWEEP2/screen/h1*csv'))
        def hints(x):
            if isinstance(x, dict):
                for k,v in x.items(): hints(k); hints(v)
            elif isinstance(x, list):
                for v in x: hints(v)
            elif isinstance(x, str) and x.endswith('.csv') and 'oof' in x.lower():
                p = Path(x.replace('\\','/')); p = p if p.is_absolute() else root / p
                if p.is_file(): paths.add(p)
        for _,d in screens: hints(d)
        for p in sorted(paths): self.hashes.setdefault(sha(p), p)
    def get(self, x, expected=None):
        if isinstance(x, dict): return self.get(x['path'], x.get('sha256', expected))
        if re.fullmatch('[0-9a-f]{64}', str(x)):
            if x not in self.hashes: return {'path': 'MISSING_SHA256/' + x, 'sha256': x}
            return {'path': str(self.hashes[x]), 'sha256': x}
        p = Path(str(x).replace('\\','/')); p = p if p.is_absolute() else self.root / p
        return {'path': str(p), **({'sha256': expected} if expected else {})}


def build_specs(root, screens, inv):
    specs, skipped = [], []
    bynum = {re.match(r'EXP(\d+B?)', p.parent.parent.name)[1]: (p,d) for p,d in screens}
    p179,d179 = bynum['179']; fixed = d179['input_files']
    rad = blend((inv.get(fixed['R42'], d179['input_sha256']['R42']),.5), (inv.get(fixed['R43'], d179['input_sha256']['R43']),.5))
    partners = {'r50': rad, 'dino': inv.get(fixed['D172'], d179['input_sha256']['D172']),
                'convnext': inv.get(fixed['C42'], d179['input_sha256']['C42'])}
    anchor = inv.get(bynum['181'][1]['inputs']['control_primary'])
    for path,d in screens:
        exp = path.parent.parent.name; n = re.match(r'EXP(\d+B?)',exp)[1]
        def add(name, pairs, family='convnext', label=False, prior=None, fold=False):
            spec = dict(name=exp+'/'+name, seeds=pairs, family=family, label_change=label,
                        old_verdict=old(d) if prior is None else prior, fold_rank=fold,
                        source=str(path), source_sha256=sha(path),
                        old_screen_verdict=old(d), old_verdict_scope='contrast' if prior is not None else 'screen')
            if family != 'bundle': spec.update(partners=partners, anchor=anchor)
            specs.append(spec)
        def pair(a,c): return {'42': {'arm':a,'control':c}}
        def files(prefix, count=5): return [get(f'{prefix}_f{f}') for f in range(count)]
        raw = d.get('input_files',{}); hashes = d.get('input_sha256',{})
        def get(k): return inv.get(raw[k], hashes.get(k))
        if n == '150':
            h=d['predictions_sha256']
            def find(a,s):
                k=next(k for k in h if f'EXP150_{a}{s}_F0' in k); return inv.get(k,h[k])
            add('W-vs-C', {str(s):{'arm':find('W',s),'control':find('C',s)} for s in (42,43)}, 'r50', True)
        elif n in ('149','157'):
            h=d['oof_k36_sha256']
            if n=='149':
                for arm in ('L1','L2','L3','H1'):
                    ac=[inv.get(h[f'EXP149_{arm}_S42_F{f}']) for f in range(2)]
                    cc=[inv.get(h[(f'EXP138_BUNDLE_AUG1_MH_S42_F{f}' if arm=='H1' else (f'EXP127_AUG1_S42_F0' if f==0 else 'EXP134_AUG1_S42_F1'))]) for f in range(2)]
                    add(arm,pair(ac,cc),'r50',True,prior=old(d['metrics'][arm]),fold=True)
                    ga=[];gc=[]
                    for f in range(2):
                        ga.append((inv.get(str(path.parent/'gold58'/f'EXP149_{arm}_S42_F{f}.csv')),.5))
                        tag=f'EXP138_BUNDLE_AUG1_MH_S42_F{f}' if arm=='H1' else ('EXP127_AUG1_S42_F0' if f==0 else 'EXP134_AUG1_S42_F1')
                        gc.append((inv.get(str(path.parent/'gold58'/f'{tag}.csv')),.5))
                    specs[-1]['image_reads']={'gold58':{'arm':dict(blend(*ga),rank=False),'control':dict(blend(*gc),rank=False),
                        'labels':'storage/competition/train.csv','components':'gold58_singletons','eligible':False}}
            else:
                ps={}
                for s in (42,43):
                    ac=[];cc=[]
                    for f in range(5):
                        tag=f'EXP149_L2_S42_F{f}' if s==42 and f<2 else f'EXP157_L2_S{s}_F{f}'
                        ac.append(inv.get(h[tag])); cc.append(inv.get(h[f'EXP127_AUG1_S{s}_F0' if f==0 else f'EXP134_AUG1_S{s}_F{f}']))
                    ps[str(s)]={'arm':ac,'control':cc}
                add('L2-vs-AUG1',ps,'r50',True,fold=True)
        elif n in ('156','161'):
            h=dict(d['prediction_sha256'])
            if n=='161': h.update({f'h1_k{k}':v for k,v in d['h1_served_oof_sha256'].items()})
            for fam in (('s42',) if n=='156' else ('aug1','h1')):
                for a,c in ((62,36),) if n=='156' else ((62,36),(94,62)):
                    prior=d['verdict'][fam][f'{a}_vs_{c}'] if n=='161' else None
                    add(f'{fam}-K{a}-vs-K{c}',pair(inv.get(h[f'{fam}_k{a}']),inv.get(h[f'{fam}_k{c}'])),'r50',prior=prior)
        elif n in ('164','166','167'):
            if n=='164':
                h=d['input_sha256']; source=[inv.get(h[f'dino_f{f}']) for f in range(5)]
                for fam, ws in d['partners'].items():
                    c=[inv.get(h[f'{fam}_f{f}']) for f in range(5)]
                    for w,rec in ws.items():
                        if not re.fullmatch(r'0\.\d+',w):continue
                        add(f'{fam}+DINO-w{w}',pair(blend((c,1-float(w)),(source,float(w))),c),'bundle',fold=True)
            elif n=='166':
                h=d['inputs_sha256']; source=inv.get(h['ConvNeXt-T'])
                for fam,ws in d['blends'].items():
                    c=inv.get(h[fam])
                    for w,rec in ws.items():
                        add(f'{fam}+ConvNeXt-w{w}',pair(blend((c,1-float(w)),(source,float(w))),c),'bundle')
            else:
                h=d['oof_files']
                def pool(k):return [inv.get(h[k][str(f)],d['input_sha256'][k][str(f)]) for f in range(5)]
                for name,rec in d['blends'].items():
                    fam='h1' if name.startswith('h1') else 'radr50'; c=pool(fam)
                    if '080' in name: a=blend((c,.8),(pool('dino'),.1),(pool('convnext'),.1))
                    elif '070' in name:a=blend((c,.7),(pool('dino'),.15),(pool('convnext'),.15))
                    else:a=blend((c,.9),(pool('convnext'),.1))
                    add(name,pair(a,c),'bundle',prior='PASS' if rec.get('pass') else 'HOLD',fold=True)
        elif n=='169':
            r=blend((files('l2_s42'),.5),(files('l2_s43'),.5)); di=files('dino'); co=files('convnext')
            q=dict(blend((r,1/3),(di,1/3),(co,1/3)),rank='pooled')
            for name,c in {'l2':r,'dino':di,'convnext':co,'h1':files('h1')}.items():
                add('bundle-vs-'+name,pair(q,c),'bundle',prior=old(d['comparisons'][name]),fold=True)
        elif n in ('170','172','183','186'):
            a,c = {'170':('convnext','control_flip'),'172':('dino_l2','dino_dread'),'183':('mixed','control'),'186':('candidate','control')}[n]
            add('primary',pair(files(a),files(c)), 'dino' if n=='172' else 'convnext', n in ('170','172'),fold=True)
            if n=='170':add('plain',pair(files('convnext_plain'),files('control_plain')),'convnext',True,fold=True)
        elif n=='171':
            for mode in ('flip','plain'):add(mode,pair(get('small_'+mode),get('tiny_'+mode)))
        elif n=='179':
            for prefix in ('','plain_k36_','plain_k24_'):
                add(prefix or 'flip_tta',{str(s):{'arm':get(prefix+'M'+str(s)),'control':get(prefix+'C'+str(s))} for s in (42,43)})
        elif n=='181':
            for mode in ('primary','plain'):add('AUG1-'+mode,pair(inv.get(d['inputs']['candidate_'+mode]),inv.get(d['inputs']['control_'+mode])),'coatnet')
        elif n in ('182','184'):
            c='CONTROL' if n=='182' else 'A25'
            for a in (('FILL',) if n=='182' else ('A0','A50')):
                for mode in ('flip_tta','plain'):add(a+'-'+mode,pair(get(a+'_'+mode),get(c+'_'+mode)),'convnext',True,prior=old(d['results'][a][mode]))
        elif n=='185':
            add('MIL', {str(s):{'arm':inv.get(hashes[f'candidate_S{s}']),'control':inv.get(hashes[f'control_S{s}'])} for s in (42,43)})
        elif n in ('188','192B'):
            a='COVERAGE' if n=='188' else 'CROP'
            for mode in ('oof_k94.csv','oof_k94_mirror_tta.csv'):add(a+'-'+mode,pair(inv.get(d['inputs'][a][mode]),inv.get(d['inputs']['CONTROL'][mode])),'coatnet')
        elif n=='191':
            for a,rec in d['arms'].items():add(a,{str(s):{'arm':inv.get(rec['input_sha256'][str(s)]),'control':inv.get(d['control_sha256'][str(s)])} for s in (42,43)},'convnext',True,prior=old(rec))
        elif n=='165':
            h150=bynum['150'][1]['predictions_sha256']
            ps={}
            for surface,rec in d['seeds'].items():
                s=rec['seed']; k=next(k for k in h150 if f'EXP150_C{s}_F0' in k)
                ps[str(s)]={'arm':inv.get(rec['prediction_sha256']),'control':inv.get(k,h150[k])}
            add('G4-vs-C',ps,'r50',True)
        elif n=='173' and path.parent.name=='screen-w4':
            for a,c in (('W4','C'),('G4','W4')):
                add(a+'-vs-'+c,{str(rec['seed']):{'arm':inv.get(rec['prediction_sha256'][a]),'control':inv.get(rec['prediction_sha256'][c])} for rec in d['seeds'].values()},'r50',True)
        elif n=='173':
            h150=bynum['150'][1]['predictions_sha256']
            ck=next(k for k in h150 if 'EXP150_C42_F0' in k); control=inv.get(ck,h150[ck])
            for half in (1,2):
                runs={}; receipts={}
                for arm in ('G4','J4'):
                    folder=path.parent.parent/'outputs'/f'{arm}-H{half}'
                    receipt=json.loads((folder/'exp173-arm.json').read_text())
                    receipts[arm]=receipt
                    runs[arm]=inv.get(str(folder/f'EXP173_{arm}_H{half}_F0/oof_k36.csv'),receipt['runner_oof_sha256'])
                for arm,base in (('G4','C'),('J4','C'),('G4','J4')):
                    add(f'{arm}-vs-{base}-H{half}',pair(runs[arm],control if base=='C' else runs[base]),'r50',True)
                    folder=path.parent.parent/'outputs'/f'{arm}-H{half}'
                    image_c=folder/'control-heldout-k36.csv' if base=='C' else path.parent.parent/'outputs'/f'{base}-H{half}'/'candidate-heldout-k36.csv'
                    control_hash=receipts[arm]['heldout_sha256']['control'] if base=='C' else receipts[base]['heldout_sha256']['candidate']
                    specs[-1]['image_reads']={'gold58':{'arm':inv.get(str(folder/'candidate-heldout-k36.csv'),receipts[arm]['heldout_sha256']['candidate']),'control':inv.get(str(image_c),control_hash),
                        'labels':'storage/competition/train.csv','components':'gold58_singletons','eligible':False}}
        elif n=='138':
            h={k.replace('\\','/'):v for k,v in d['input_bindings'].items()}
            for contrast,rec in d['contrasts'].items():
                mode,base=contrast.split('_vs_'); kfile='oof_best.csv' if 'k24' in mode else 'oof_k36.csv'
                ps={}
                for s in (42,43):
                    ac=[];cc=[]
                    for f in range(5):
                        ak=f'artifacts/experiments/EXP138_BUNDLE_AUG1_MH_S{s}_F{f}/{kfile}'
                        if base.startswith('aug1'): ct=f'EXP127_AUG1_S{s}_F0' if f==0 else f'EXP134_AUG1_S{s}_F{f}'
                        else:ct=f'EXP113_RADR50_LOCAL_F{f}' if s==42 else f'EXP118_RADR50_LOCAL_S43_F{f}'
                        ck=f'artifacts/experiments/{ct}/{kfile}'
                        ac.append(inv.get(ak,h[ak]));cc.append(inv.get(ck,h[ck]))
                    if mode=='tta':
                        ak=f'artifacts/experiments/EXP138_BUNDLE_AUG1_MH/tta/s{s}/exp117/oof_tta_k36.csv'
                        ac={'files':inv.get(ak,h[ak]),'rank_folds_like':ac}
                    ps[str(s)]={'arm':ac,'control':cc}
                add(contrast,ps,'r50',True,prior=old(rec),fold=True)
        elif n=='142':
            h={k.replace('\\','/'):v for k,v in d['input_bindings'].items()}
            for file in ('oof_best.csv','oof_k36.csv'):
                ps={}
                for s in (42,43):
                    ak=next(k for k in h if f'EXP142_MRN_RESCUE_S{s}_F0/{file}' in k)
                    ck=next(k for k in h if f'EXP127_AUG1_S{s}_F0/{file}' in k)
                    ps[str(s)]={'arm':inv.get(ak,h[ak]),'control':inv.get(ck,h[ck])}
                add(file,ps,'r50')
        else:
            skipped.append({'screen':str(path),'pair':exp,'reason':'Unrecognized historical schema; explicit contrast adapter required.'})
    return specs, skipped


def add_plain(specs, inv):
    """Recover paired plain siblings only when every primary input has that sibling."""
    extra=[]
    def fingerprint(s):
        def key(x):
            if isinstance(x,list):return tuple(key(v) for v in x)
            if isinstance(x,dict):
                if 'path' in x:return x.get('sha256') or sha(x['path'])
                return tuple((k,key(v)) for k,v in sorted(x.items()))
            return x
        return key(s['seeds'])
    known=set()
    for s in specs:
        try: known.add(fingerprint(s))
        except FileNotFoundError: pass
    for s in specs:
        if any(x in s['name'] for x in ('plain','K','oof_k94','AUG1-','138','142','149','150','156','157','161','164','165','166','167','169','173')):continue
        pairs={}; okay=True
        for seed,pair in s['seeds'].items():
            pairs[seed]={}
            for side, entries in pair.items():
                entries=entries if isinstance(entries,list) else [entries]
                paths=[]
                for entry in entries:
                    if not isinstance(entry,dict) or 'path' not in entry:okay=False;break
                    p=Path(entry['path'])
                    if '_flip_tta' not in p.name:okay=False;break
                    pp=p.with_name(p.name.replace('_flip_tta',''))
                    if not pp.is_file():okay=False;break
                    paths.append(inv.get(str(pp),sha(pp)))
                pairs[seed][side]=paths
        if okay:
            # Preserve scalar-vs-fold-list representation for fingerprinting.
            for seed in pairs:
                for side in pairs[seed]:
                    if not isinstance(s['seeds'][seed][side],list): pairs[seed][side]=pairs[seed][side][0]
            other=dict(s);other.update(name=s['name']+'-plain-sibling',seeds=pairs,
                old_verdict='UNREGISTERED PLAIN DIAGNOSTIC', old_verdict_scope='unregistered sibling', parent_old_verdict=s['old_verdict'])
            fp=fingerprint(other)
            if fp not in known:extra.append(other);known.add(fp)
    return specs+extra


def run(root=ROOT, out=None):
    root=Path(root);out=Path(out or root/'artifacts/experiments/MULTI_LENS_RETRO');start=time.time()
    screens=[(p,json.loads(p.read_text(encoding='utf-8'))) for p in sorted(root.glob('artifacts/experiments/*/screen*/result*.json'))]
    print(f'Indexing OOF bindings for {len(screens)} screens',flush=True)
    inv=Inventory(root,screens); specs,skipped=build_specs(root,screens,inv);specs=add_plain(specs,inv)
    write(out/'inputs.json',{'version':VERSION,'specs':specs,'skipped':skipped})
    rows=[];cache={}
    for s in specs:
        try:
            row=run_spec(s,root,cache)
            row.update({k:s[k] for k in ('source','source_sha256','old_screen_verdict','old_verdict_scope')})
            held=held_verdict(row['old_verdict'])
            row['flip']=row['verdict']=='ADVANCE' and held
            row['overlooked_targets']=row['lenses']['L4']['selected_targets'] if held else []
            row['parent_old_verdict']=s.get('parent_old_verdict')
            rows.append(row);print(f'{len(rows)} {s["name"]}: {row["decision_label"]}',flush=True)
        except (FileNotFoundError,ValueError,KeyError) as e:
            skipped.append({'screen':s['source'],'pair':s['name'],'reason':str(e)});print('SKIP '+s['name']+': '+str(e),flush=True)
    result={'version':VERSION,'bootstrap':{'draws':DRAWS,'seed':SEED,'unit':'validation connected component','ci':'paired percentile 95%; undefined draws excluded, fixed target support'},
            'decision_policy':'Mechanical user-dispatch gate: guard AND any eligible lens. B holistic decision remains separate.',
            'screen_count':len(screens),'pair_count':len(rows),'flips':[r['name'] for r in rows if r['flip']],
            'code_sha256':{p:sha(root/p) for p in ('tools/multi_lens_readout.py','tools/multi_lens_retro.py')},
            'results':rows,'skipped':skipped,'elapsed_seconds':round(time.time()-start,2)}
    write(out/'result.json',result)
    lines=['# Multi-lens retrospective v'+VERSION,'',f'{len(rows)} evaluated contrasts; {len(skipped)} skipped; {len(result["flips"])} old held/rejected → ADVANCE.',
           '', 'Mechanical dispatch verdict: harm guard AND any eligible lens. B retains the later holistic release judgment. No release/submission was made.',
           '', '300 paired component draws, seed 20261003; point thresholds (CI need not exclude zero). Fixed 12-target support; undefined bootstrap draws omitted and counted. All target/ruler CIs are in result.json.',
           '', 'L1–L4 aggregate paired seed AUC deltas; L5 instead scores arithmetic seed-averaged predictions and requires every seed positive on the same passing weak/macro statistic. Folds are percentile-ranked before pooling.',
           '', 'L6 uses equal-family rank Q, removing the replaced family. Final proxy is 50:50 rank(Q)/rank(EXP181 CONTROL), on common fold-0 studies only; CoAtNet replaces the anchor itself. Weights are exploratory fixed proxies, not fitted. Existing composed-bundle contrasts have L6 unavailable to avoid double counting.',
           '', 'L7 uses saved EXP149 Gold58 predictions (two-fold probability-mean diagnostic, not the old mean-fold AUC) and EXP173 held-out half predictions. Gold58 has no validation component mapping: singleton study bootstrap is explicitly diagnostic. Other image reads are unavailable without paired predictions; OAI roster/labels alone are not a delivered model readout. Synovitis/PF OA need image-read review. Aquila label-change readouts are flagged self-referential.',
           '', 'Old verdict scope is recorded in JSON: screen-level gates may concern a registered bundle or runtime, while this table shows the direct arm contrast. Plain siblings without an explicit old gate are unregistered diagnostics, never counted as gate flips. L3/L4 maxima and multi-lens selection are exploratory, without multiplicity correction.',
           '', '| Arm | Old verdict | L1 | L2 | L3 | L4 | L5 | L6 | L7 | Guard | New verdict |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    def cell(x):
        if x['value'] is None:return 'N/A'
        if 'sign_agreement' in x:
            return ('**PASS**' if x['PASS'] else 'fail')+'; weak '+cell(x['weak_four'])+'; macro '+cell(x['macro12'])+'; seed signs '+str(x['sign_agreement'])
        if 'final_proxy' in x:
            base=dict(x);base.pop('final_proxy');base['PASS']=base['value'] >= base['threshold']-1e-12
            return 'Q '+cell(base)+'; final '+cell(x['final_proxy'])
        if 'entries' in x:
            return '; '.join(k+(' (diagnostic) ' if not v['eligible'] else ' ')+cell(v) for k,v in x['entries'].items())
        v=x['value'];v=f'{v:+.5f}' if isinstance(v,(float,int)) else 'see JSON'
        ci=x.get('ci95');ci=f' [{ci[0]:+.5f},{ci[1]:+.5f}]' if isinstance(ci,list) and len(ci)==2 else ''
        return ('**PASS** ' if x['PASS'] else 'fail ')+v+ci
    for r in rows:
        name=('**'+r['name']+'**') if r['flip'] else r['name']
        lines.append('| '+' | '.join([name,r['old_verdict'].replace('|','/'),*[cell(r['lenses'][f'L{i}']) for i in range(1,8)],str(r['harm_guard']['PASS']),r['decision_label']])+' |')
    lines+=['','## Held/rejected target gains (including guard-blocked arms)','']
    for r in rows:
        if r['overlooked_targets']:lines.append('- '+r['name']+': '+', '.join(r['overlooked_targets'])+'; guard '+str(r['harm_guard']['PASS'])+'; '+r['decision_label'])
    lines+=['','## Newly recovered plain diagnostics (not registered-gate flips)','']
    for r in rows:
        if r['parent_old_verdict'] and r['verdict']=='ADVANCE':lines.append('- '+r['name']+': '+r['decision_label']+'; primary old gate '+r['parent_old_verdict'])
    lines+=['','## Skipped inputs / non-selection diagnostics','']
    for x in skipped:lines.append('- '+x['pair']+': '+x['reason'])
    if not skipped:lines.append(f'None: all {len(screens)} discovered screens have local evaluable pairs. EXP186B is stored as EXP186_FINDING_MIL_5FOLD/screen-b.')
    lines+=['','## Reproduce','','```powershell', "$env:PYTHONUTF8='1'", 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe tools/multi_lens_readout.py --retro','```','', 'SHA256 source bindings and exact recipes: inputs.json. Per-prediction hashes: result.json. No labels or study rows are printed.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'pairs':len(rows),'skipped':len(skipped),'flips':result['flips'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
