"""Prepare a pinned-source subset experiment; never starts GPU work.

The derived runner preserves the frozen full-control algorithm and replaces only
training membership/sampling, with extra identity, skip and wall-time guards.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='artifacts/experiments/EXP094_B8_VARIANT_PROTOCOL_V4/manifest.json'
SOURCE='training/runners/exp094_b8_variant.py'
PIN='daab9d7703924161495be90121e509675f94ac8a7460fbfda4d8fcb1457abfaf'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--subset',type=Path,required=True);a=ap.parse_args()
    assert sha(ROOT/SOURCE)==PIN
    subset=json.loads(a.subset.read_text());base=json.loads((ROOT/BASE).read_text())
    assert len(subset['full_train'])==3455 and len(subset['subset_train'])==2764 and len(subset['validation'])==888
    assert set(subset['subset_train']) < set(subset['full_train'])
    a.out.mkdir(parents=True,exist_ok=False)
    original=(ROOT/SOURCE).read_text(encoding='utf-8');source=original
    patches=[]
    def patch(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new,1);patches.append({'old':old,'new':new})
    patch('ROOT = Path(__file__).resolve().parents[2]',f'ROOT = Path({str(ROOT)!r})')
    patch('    a = ap.parse_args()', '    ap.add_argument("--hc05-protocol", required=True)\n    a = ap.parse_args()')
    patch('    protocol, ordered, arm = guard.validate_arguments(a, ROOT)', '''    protocol, ordered, arm = guard.validate_arguments(a, ROOT)
    hcpath = Path(a.hc05_protocol)
    hc = json.loads(hcpath.read_text(encoding="utf-8"))
    hc_run = hc["runs"][str(a.seed)]
    assert a.mode == "full" and a.variant == "seed"
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == hc["runner_sha256"]
    assert hashlib.sha256((ROOT / hc["subset_path"]).read_bytes()).hexdigest() == hc["subset_sha256"]
    assert hashlib.sha256((ROOT / hc["parity_path"]).read_bytes()).hexdigest() == hc["parity_sha256"]
    assert json.loads((ROOT / hc["parity_path"]).read_text())["status"] == "PASS"
    subset = json.loads((ROOT / hc["subset_path"]).read_text())
    assert subset["full_train"] == ordered["train"] and subset["validation"] == ordered["val"]
    assert len(subset["subset_train"]) == len(set(subset["subset_train"])) == 2764
    assert set(subset["subset_train"]) < set(ordered["train"])
    a.tag = hc_run["tag"]
    if (EXPERIMENTS / a.tag).exists():
        raise SystemExit("Refuse to overwrite HC05 run")
    hc_start = time.monotonic()
    hc_budget_path = hcpath.with_name("execution_budget.json")
    hc_budget = json.loads(hc_budget_path.read_text())
    assert hc_budget["completed_seeds"] == ([] if a.seed == 42 else [42])
    hc_remaining = 7200 - hc_budget["seconds_used"]
    if hc_remaining < 300:
        raise SystemExit("HC05 aggregate budget exhausted")''')
    patch('    if not provenance["environment_matches_baseline"] and not __import__("os").environ.get("KNEE_REMOTE_RELAX_ENV"):',
          '    if not provenance["environment_matches_baseline"]:')
    patch('    tds = VariantStudyWindows(', '''    full_train_ids = train_ids
    train_ids = subset["subset_train"]
    log(f"HC05 subset intervention: {len(train_ids)} unique studies; full-T pos_weight; 3455 sampler positions/epoch")
    tds = VariantStudyWindows(''')
    patch('    tl = DataLoader(tds, batch_size=a.bs, shuffle=True,', '''    from torch.utils.data import RandomSampler
    sampler = RandomSampler(tds, replacement=False, num_samples=len(full_train_ids), generator=shuffle_generator)
    tl = DataLoader(tds, batch_size=a.bs, sampler=sampler, shuffle=False,''')
    patch('    (out / "provenance.json").write_text(', '''    assert provenance["initial_state_sha256"] == hc_run["initial_state_sha256"]
    provenance.update(hc05_protocol_sha256=hashlib.sha256(hcpath.read_bytes()).hexdigest(),
                      original_runner_sha256=hc["original_runner_sha256"], subset_sha256=hc["subset_sha256"],
                      sampler="RandomSampler without replacement, num_samples=3455; cycles subset per virtual epoch",
                      full_train_pos_weight_fixed=True)
    (out / "provenance.json").write_text(''')
    patch('        for i, (x, y, batch_uids) in enumerate(tl):', '''        for i, (x, y, batch_uids) in enumerate(tl):
            if time.monotonic() - hc_start > hc_remaining - 120:
                raise RuntimeError("HC05 aggregate hard budget: stop before scoring")''')
    patch('    provenance["fp16_skipped_updates"] = skipped', '''    provenance["fp16_skipped_updates"] = skipped
    if optimizer_updates - skipped != hc_run["expected_successful_updates"]:
        (out / "invalid-contrast.json").write_text(json.dumps(dict(reason="successful-update mismatch",
            attempts=optimizer_updates, skips=skipped, expected=hc_run["expected_successful_updates"])))
        raise RuntimeError("HC05 successful-update mismatch; no efficacy score or automatic retry")''')
    patch('    write_predictions(out / "oof_best.csv", U, P)', '''    write_predictions(out / "oof_best.csv", U, P)
    if time.monotonic() - hc_start > hc_remaining - 90:
        raise RuntimeError("HC05 insufficient budget for registered K36 diagnostic")
    a.k_eval = 36
    P36, U36 = evaluate(oof_ids)
    assert U36 == oof_ids
    write_predictions(out / "oof_k36.csv", U36, P36)
    a.k_eval = 24''')
    patch('    log(f"done in {summary[\'total_seconds\']}s -> {out}"); log_file.close()', '''    hc_elapsed = time.monotonic() - hc_start
    if hc_elapsed > hc_remaining:
        raise RuntimeError("HC05 aggregate budget exceeded: invalid run")
    (out / "runner-complete.json").write_text(json.dumps(dict(successful_updates=optimizer_updates-skipped,
        expected_successful_updates=hc_run["expected_successful_updates"], scope="requires outer VALIDATED_SUCCESS receipt")))
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()''')
    patch('    names = ["summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv"]',
          '    names = ["summary.json", "provenance.json", "history.json", "model_best.pt", "oof_best.csv", "oof_k36.csv"]')
    # Environment guards run before importing numerical/model code.
    patch('import numpy as np','import os\nfor _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):\n    os.environ[_k]="1"\nos.environ["HF_HUB_OFFLINE"]="1"\nimport numpy as np')
    compile(source,str(a.out/'runner.py'),'exec')
    (a.out/'runner.py').write_text(source,encoding='utf-8')
    (a.out/'source_changes.json').write_text(json.dumps(patches,indent=2),encoding='utf-8')
    runs={}
    for seed in [42,43]:
        pv=json.loads((ROOT/f'artifacts/experiments/EXP094L_B8_SEED{seed}/provenance.json').read_text())
        runs[str(seed)]={'tag':f'HC05_UNIQUE2764_SEED{seed}','initial_state_sha256':pv['initial_state_sha256'],
                         'expected_attempts':3448,'expected_successful_updates':3448 if seed==42 else 3447}
    protocol={'schema':'hc05-unique-study-training-v1','status':'PREPARED_NOT_LAUNCHED','original_runner_sha256':PIN,
              'runner_sha256':sha(a.out/'runner.py'),'subset_path':str(a.subset.relative_to(ROOT) if a.subset.is_absolute() else a.subset),
              'subset_sha256':sha(a.subset),'parity_path':'.local/hc05-unique-study-v1/parity.json','parity_sha256':'REQUIRES_CPU_PARITY',
              'runs':runs,'hard_budget_seconds':7200,'review_required':True,
              'decision':'full-minus-subset K24 macro12 >0 at both seeds and mean>=.004 to prioritize data scale; fixed V888 weak reference only'}
    (a.out/'manifest.draft.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    commands=[];runner_commands={}
    for seed in [42,43]:
        cmd=[sys.executable,str((a.out/'runner.py').resolve()),'--protocol',str(ROOT/BASE),'--arm',f'seed{seed}','--mode','full','--tag',f'EXP094L_B8_SEED{seed}',
             '--variant','seed','--seed',str(seed),'--slot-dropout','0.0','--hc05-protocol',str((a.out/'manifest.json').resolve())]
        for k,v in {**base['common_args'],**{k:str(ROOT/v) for k,v in base['paths'].items() if k in ('corpus','labels','eval_labels')}}.items():
            if isinstance(v,bool):
                if v:cmd.append('--'+k.replace('_','-'))
            else:cmd+=['--'+k.replace('_','-'),str(v)]
        runner_commands[str(seed)]=cmd
        outer=[sys.executable,str(ROOT/'tools/unique_study_run.py'),'--protocol',str((a.out/'manifest.json').resolve()),'--seed',str(seed)]
        commands.append({'tag':runs[str(seed)]['tag'],'command':outer,**({'after':runs['42']['tag']} if seed==43 else {})})
    (a.out/'runner_commands.json').write_text(json.dumps(runner_commands,indent=2),encoding='utf-8')
    (a.out/'queue.draft.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(a.out),'runs':runs,'status':'DRAFT_REQUIRES_PARITY_AND_REVIEW'},indent=2))
if __name__=='__main__':main()
