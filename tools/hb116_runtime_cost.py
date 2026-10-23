"""Offline cost accounting for the named, immutable HB116/C85 receipts."""
import json
from pathlib import Path
from tools import hb116_vnext_release as h

FAMILIES = ('R50', 'C4', 'S', 'OWN', 'Rx', 'S6')


def build(out=h.OUT):
    runs = {}
    for name in ('a', 'b', 'stress-n32'):
        root = h.ROOT/f'.local/scratch/slot-b/vnext-{name}-visible'
        receipt = h.read(root/'hb116-receipt.json')
        n = receipt['studies']
        h.need(n == (32 if name == 'stress-n32' else 3), 'receipt cohort size')
        phase_file = root/'hb116-run/phases.json'
        h.need(h.read(phase_file) == receipt['phases'], 'phase receipt differs from journal')
        telemetry_hashes = {'hb116-run/phases.json': h.sha(phase_file)}
        phases = {}
        for family in FAMILIES:
            wall = receipt['phases'][family]['seconds']
            if family == 'S6':
                source = root/'hb116-run/public/p956-run/s6/controller/submission_receipt.json'
                telemetry_hashes[str(source.relative_to(root))] = h.sha(source)
                control = h.read(source)
                workers = h.c.s6._worker_receipts(control)
                rows = [r for w in workers for r in w['per_study']]
                h.need(len(rows) == n, 'S6 incomplete service rows')
                decode, render, anchor, inference = [sum(float(r[i]) for r in rows) for i in range(2, 6)]
                service = decode + render + anchor + inference
                # Worker elapsed residual is not necessarily model load alone.
                setup = sum(w['seconds'] for w in workers) - service
                prep = decode + render + anchor
                extra = dict(decode=decode/n,render=render/n,anatomy_anchor=anchor/n,
                             setup_scope='worker non-service residual; setup vs idle unidentifiable')
            else:
                sources = [root/f'hb116-run/{family}/worker{i}/timing.json' for i in range(2)]
                telemetry_hashes.update({str(p.relative_to(root)):h.sha(p) for p in sources})
                workers = [h.read(p) for p in sources]
                h.need(sum(len(w['study_seconds']) for w in workers) == n, 'native incomplete service rows')
                service = sum(sum(w['study_seconds'].values()) for w in workers)
                setup = sum(w['setup_seconds'] for w in workers)
                def member_of(k):
                    return k.startswith(family+'-') if family in ('R50','C4','S') else k in (
                        ('EXP143v2','EXP155','S43') if family=='OWN' else ('R','R2-s43','R2-s44'))
                inference = sum(v['seconds'] for k,v in receipt['timing']['members'].items() if member_of(k))
                prep = service - inference
                extra = dict(decode=None,render=None,anatomy_anchor=None,
                             setup_scope='measured child setup; decode/prepare/journal inseparable')
            idle = 2*wall - setup - service
            phases[family] = dict(wall=wall,setup_equivalent=setup/2,
                decode_prepare_equivalent=prep/2,inference_equivalent=inference/2,idle_tail_equivalent=idle/2,
                per_study_equivalents=dict(setup=setup/(2*n),decode_prepare=prep/(2*n),
                    inference=inference/(2*n),idle_tail=idle/(2*n)),
                service_decode_prepare_per_study=prep/n,service_inference_per_study=inference/n,
                wall_per_study=wall/n,**extra)
            assert abs(sum(phases[family][k] for k in ('setup_equivalent','decode_prepare_equivalent',
                'inference_equivalent','idle_tail_equivalent'))-wall)<1e-8
        runs[name]=dict(studies=n,fixed=receipt['fixed_seconds'],total=receipt['total_seconds'],phases=phases,
                        receipt_sha256=h.sha(root/'hb116-receipt.json'),telemetry_sha256=telemetry_hashes)
    stress=runs['stress-n32']; marginals={}
    for family in FAMILIES:
        values=[(stress['phases'][family]['wall']-runs[v]['phases'][family]['wall'])/29 for v in ('a','b')]
        marginals[family]=dict(range=[min(values),max(values)],mean=sum(values)/2,
            intercepts=[runs[v]['phases'][family]['wall']-3*x for v,x in zip(('a','b'),values)])
    rec=h.read(h.ROOT/'.local/scratch/slot-b/vnext-stress-n32-visible/hb116-receipt.json')
    mem={k:v['seconds_per_study'] for k,v in rec['timing']['members'].items()}
    native_infer=sum(v for k,v in mem.items() if k!='S6')
    # Preserve warmup in this estimate; no invented target engineering measurement.
    # Shared Q and anchor preparation are overlapped with inference. Budget the
    # entire old OWN non-inference residual as unhidden device preparation.
    unhidden=stress['phases']['OWN']['service_decode_prepare_per_study']/2
    base=sum(marginals[f]['mean'] for f in FAMILIES)
    startup=300.  # planning allowance, separately disclosed, not a measurement
    variants={}
    for rx in ('R2','R'):
        for s43 in (True,False):
            key=rx.lower()+('' if s43 else '-no-s43')
            cut=(0 if s43 else mem['S43'])+(0 if rx=='R2' else mem['R2-s44'])
            infer=(native_infer-cut)/2
            # R speed proxy is same architecture R2-s43. Spread uses warm/cold
            # same-architecture service difference, not a claimed R measurement.
            r_uncertainty=0 if rx=='R2' else abs(mem['R2-s43']-mem['R2-s44'])/2
            point=infer+unhidden+marginals['S6']['mean']
            low=infer-r_uncertainty+13.3
            high=infer+unhidden+r_uncertainty+max(marginals['S6']['range'])
            variants[key]=dict(old_marginal=base-cut/2,old_projection=300+1300*(base-cut/2),
                engineered_slow_node_point=point,engineered_fast_node_point=infer+unhidden+13.3,
                engineering_floor_same_node=infer+min(marginals['S6']['range'])-r_uncertainty,
                engineered_range=[low,high],projection_point=startup+1300*point,
                projection_range=[startup+1300*low,startup+1300*high],
                native_inference_equivalent=infer,unhidden_prepare_allowance=unhidden)
    c85=h.read(h.ROOT/'.local/scratch/slot-b/c85-v104-stress/c85-receipt.json')
    result=dict(runs=runs,marginals=marginals,variants=variants,startup_allowance=startup,
        s6_node_comparison=dict(vnext_n32=stress['phases']['S6']['wall'],
            c85_n32=c85['p956']['phases']['S6']['seconds'],c85_marginal_user_supplied=13.3,
            c85_receipt_sha256=h.sha(h.ROOT/'.local/scratch/slot-b/c85-v104-stress/c85-receipt.json'),
            caution='Different node/trajectory; 13.3 is user-supplied, not derivable from this N32-only C85 receipt.'),
        accounting='Every phase wall equals half summed setup + half summed study non-inference + half summed inference + residual idle/tail. These are two-GPU equivalents, not a critical-worker trace.',
        limitations=['Native inference is member-service time, including warmup/transfer/synchronization; old telemetry cannot separate kernel-only time.',
            'Native decode/prepare/journal and exact idle vs startup are not individually identifiable from the downloaded telemetry.',
            'N3->N32 is a cohort-and-node comparison, not a paired scaling experiment or statistical confidence interval.',
            'Estimates assume full CPU preparation overlap, unchanged GPU service, no memory spills, and two balanced shards; real T4 qualification remains mandatory.',
            'R is estimated by same-architecture R2 member service; cut runtimes were not measured.',
            'Stress fixed cost includes construction of diagnostic cohort; do not charge its 465 seconds as hidden-test startup.',
            'Memory and storage/CPU throughput remain unmeasured on the new T4 worker; scenario ranges are not hard upper bounds.'])
    folder=Path(out)/'validation';folder.mkdir(parents=True,exist_ok=True)
    h.dump(folder/'cost-model.json',result)
    lines=['# HB116 v1.1.0 runtime cost model','',result['accounting'],'',
           '| Run/phase | Setup equiv s | Decode/prep equiv s | Inference equiv s | Idle/tail equiv s | Wall s | Wall s/study |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for run,data in runs.items():
        for f,p in data['phases'].items():
            lines.append('| '+run+'/'+f+' | '+' | '.join(f'{p[k]:.3f}' for k in ('setup_equivalent','decode_prepare_equivalent','inference_equivalent','idle_tail_equivalent','wall','wall_per_study'))+' |')
    lines += ['', 'The same observed allocation per study (not a hidden-set marginal):', '',
              '| Run/phase | Setup s/study | Decode/prep s/study | Inference s/study | Idle/tail s/study |',
              '|---|---:|---:|---:|---:|']
    for run,data in runs.items():
        for f,p in data['phases'].items():
            parts=p['per_study_equivalents']
            lines.append('| '+run+'/'+f+' | '+' | '.join(f'{parts[k]:.3f}' for k in ('setup','decode_prepare','inference','idle_tail'))+' |')
    lines += ['', 'S6 service detail (single-worker seconds per study; divide by two for ideal balanced wall contribution):', '',
              '| Run | Decode | Render | Anatomy/anchor | GPU inference |', '|---|---:|---:|---:|---:|']
    for run,data in runs.items():
        p=data['phases']['S6']
        lines.append('| '+run+' | '+' | '.join(f'{p[k]:.3f}' for k in ('decode','render','anatomy_anchor','service_inference_per_study'))+' |')
    lines += ['', 'S6 setup is the worker non-service residual, not separately measured model load. Native decode, prepare and journal time are inseparable in these receipts. Idle/tail includes process launch/exit and shard imbalance.', '',
              '| Run | Fixed outside phases s | Total receipt s |', '|---|---:|---:|']
    for run,data in runs.items():
        lines.append(f'| {run} | {data["fixed"]:.3f} | {data["total"]:.3f} |')
    lines+=['','N32 minus N3 marginal s/study: '+', '.join(f'{f} {m["mean"]:.3f} [{m["range"][0]:.3f}, {m["range"][1]:.3f}]' for f,m in marginals.items()),'',
            '| Cut (A and B identical cost) | v1.0.1 est s/study | v1.1.0 slow-node est s/study | Node/engineering scenario s/study | 1300 slow-node s | 1300 scenario s |',
            '|---|---:|---:|---:|---:|---:|']
    for v,p in variants.items():
        lines.append(f'| {v} | {p["old_marginal"]:.2f} | {p["engineered_slow_node_point"]:.2f} | {p["engineered_range"][0]:.2f} to {p["engineered_range"][1]:.2f} | {p["projection_point"]:.0f} | {p["projection_range"][0]:.0f} to {p["projection_range"][1]:.0f} |')
    lines+=['','The 1300-study projections include a planning allowance of 300 seconds (not a measurement); marginal seconds/study exclude startup. A/B and stress retain the same member path; stress only changes the cohort.',
            'S6 N32: vNext 616.243 s versus C85 509.686 s (106.557 s difference); no S6 engineering speedup is claimed.',
            result['s6_node_comparison']['caution'],
            f'The full native inference estimate is {native_infer/2:.6f} s/study, plus {unhidden:.6f} s/study unhidden preparation. Registered cuts subtract half the removed member service. The slow-node point then adds the measured {marginals["S6"]["mean"]:.6f} s/study S6 marginal; the fast-node scenario uses the supplied 13.3 s/study. The R proxy uncertainty is +/- {abs(mem["R2-s43"]-mem["R2-s44"])/2:.6f} s/study.',
            'On the measured slow-S6 regime, even both registered cuts exceed 28,800 s under ideal native overlap. A faster S6 node could fit, but cannot be relied on.',
            '',*['- '+s for s in result['limitations']]]
    (folder/'COST_MODEL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return variants


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
