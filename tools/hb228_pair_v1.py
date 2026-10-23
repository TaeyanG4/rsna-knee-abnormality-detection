"""H-B-228 v1.0.0: diagnostic R6 stride1/2, same N32; NEVER SUBMIT.

GPU entry points are for B's later reviewed Kaggle run only. CPU tests inject
execution boundaries. Paired inference retains parent code, codecs, flip, weights
and controller; the diagnostic removes only K1 injection.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

from tools import hb228_compare_v1 as comparison
from tools import k1sr_n32_v1 as n32


def write(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(runtime, output):
    import exp169_release_c85r7codec as release
    runtime,output = Path(runtime),Path(output)
    comparison.need(not (output/'submission.csv').exists(), 'root submission forbidden')
    work = output/'hb228-pair'
    comparison.need(not work.exists(), 'fresh diagnostic directory required')
    work.mkdir()
    write(work/'receipt.json',dict(status='STARTING', never_submit=True, qualified_for_submission=False))
    comp = release.competition_root()
    roster = work/'n32'
    n32.prepare(comp,roster)
    ids = json.loads((roster/'selection.json').read_bytes())['ids']
    comparison.need(len(ids) == 32, 'exact N32 required')
    code,anatomy = release.s6.prepare_code(
        release.v5q._mount(release.v5q.CODE_SLUG,'anatomy.pt'), work/'s-code', install_wheels=False)
    codec = release.install_codecs(runtime,output)
    _,tele = release.codec_dirs(output/release.RUN)
    write(work/'paths.json',dict(code=str(code),anatomy=str(anatomy),roster=str(roster)))
    # Install only the inherited pinned offline wheels; fresh children import them.
    release.s6.install_wheels(sorted((work/'s-code/wheels').glob('*.whl')),code)
    elapsed = {}
    for stride in (1,2):
        env = release.codecs.child_env(release.child_base_env(runtime),codec,
                                       branch='PR7',telemetry_dir=tele)
        started = time.perf_counter()
        release.v5q.run_process_group(
            [sys.executable,'-u',str(runtime/'tools/hb228_pair_v1.py'),'worker',
             '--runtime',str(runtime),'--output',str(output),'--stride',str(stride)],
            label='HB228 R6 stride '+str(stride),env=env)
        elapsed[str(stride)] = time.perf_counter()-started
        arm = work/('stride'+str(stride))
        a_ids,_ = comparison.predictions(arm/'s_r7/predictions.csv')
        comparison.need(set(a_ids) == set(ids) and len(a_ids) == 32, 'arm roster mismatch')
        shutil.copyfile(arm/'s_r7/predictions.csv',work/('r6-stride'+str(stride)+'.csv'))
        write(work/'progress.json',dict(completed_strides=list(elapsed),seconds=elapsed,never_submit=True))
    report = comparison.compare(work/'r6-stride1.csv',work/'r6-stride2.csv',expected_ids=ids)
    write(work/'comparison.json',report)
    arms = {str(s):json.loads((work/('stride'+str(s))/'timing.json').read_bytes()) for s in (1,2)}
    write(work/'receipt.json',dict(status='COMPLETE',never_submit=True,
        qualified_for_submission=False,submission_sha256=None,studies=32,
        selection_sha256=comparison.sha(roster/'selection.json'), order=[1,2],
        child_wall_seconds=elapsed,arms=arms,
        inference_seconds_saved=arms['1']['inference_seconds']-arms['2']['inference_seconds'],
        caveat='Sequential fixed order; OS/GPU cache warming confounds timing. R6 only, no K1 or other parent branches.',
        clean_oof=False,exposure='EXPOSED_EXPLORATORY'))
    comparison.need(not (output/'submission.csv').exists(), 'root submission forbidden')
    print(json.dumps(dict(status='COMPLETE_DIAGNOSTIC',studies=32,never_submit=True)))


def worker(runtime, output, stride):
    import exp169_release_c85r7codec as release
    import exp169_r7_port as r7
    comparison.need(type(stride) is int and stride in (1,2), 'invalid R6 stride')
    comparison.need(r7.VERSION == '1.0.2-hb228-pair', 'diagnostic port identity')
    r7.HB228_EVAL_STRIDE = stride
    work = Path(output)/'hb228-pair'
    arm = work/('stride'+str(stride))
    arm.mkdir()
    paths = json.loads((work/'paths.json').read_bytes())
    code,anatomy,roster = (Path(paths[k]) for k in ('code','anatomy','roster'))
    ids = json.loads((roster/'selection.json').read_bytes())['ids']
    run_dir = Path(output)/release.RUN
    gate = release.gate_s_r7(run_dir,release.codec_receipt(run_dir),code)
    # Same branch gate used by the parent; inactive codecs are recorded, not hidden.
    started = time.perf_counter()
    r7.infer_studies([roster/'test_series'/uid for uid in ids],
        bundle_dir=r7.resolve_bundle(),code_dir=code,anatomy_weights=anatomy,
        output_dir=arm/'s_r7',workers=2,amp='auto',device='cuda',
        max_fallback_fraction=0.0,source_dir=Path(runtime)/'exp169_r7_src',
        expected_manifest_sha256=r7.MANIFEST_SHA256)
    elapsed = time.perf_counter()-started
    prov = json.loads((arm/'s_r7/provenance.json').read_bytes())
    args = prov['worker_args']
    comparison.need(args[args.index('--eval-stride')+1] == str(stride)
                    and '--flip-tta' in args, 'actual worker contract differs')
    receipt = json.loads((arm/'s_r7/controller/submission_receipt.json').read_bytes())
    comparison.need(receipt['rows'] == 32 and receipt['fallback_studies'] == 0, 'incomplete/fallback arm')
    write(arm/'timing.json',dict(stride=stride,inference_seconds=elapsed,codec_gate=gate,
        controller_receipt_sha256=comparison.sha(arm/'s_r7/controller/submission_receipt.json'),
        provenance_sha256=comparison.sha(arm/'s_r7/provenance.json'),
        predictions_sha256=comparison.sha(arm/'s_r7/predictions.csv'),
        fallback_studies=receipt['fallback_studies'],never_submit=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('worker',))
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,choices=(1,2),required=True)
    a = p.parse_args()
    worker(a.runtime,a.output,a.stride)


if __name__ == '__main__':
    main()
