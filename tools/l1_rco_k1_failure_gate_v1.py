"""v1.0.0: preregistered 977-row failure-weight gate; CPU only, no tuning.

Changelog: reuse foldgate roster/export/ruler validation; require >=3/4 wins
AND mean delta >= .002. Indirect fold0 ancestry remains exposed exploratory.
"""
import argparse
import json
import math
import os
from pathlib import Path

from tools import l1_rco_k1_failure_v1 as producer
from tools import l1_rco_k1_foldgate_v1 as gate

k = producer.k
POLICY = "at least 3/4 candidate>=seed0 rulers AND arithmetic mean delta>=0.002"


def decide(baseline, candidate):
    k.need(set(baseline) == set(candidate) == set(gate.RULERS)
           and all(math.isfinite(v) and 0 <= v <= 1 for d in (baseline,candidate) for v in d.values()),
           "four finite macro AUCs required; never drop a ruler")
    delta = {r: candidate[r]-baseline[r] for r in gate.RULERS}
    wins = [r for r in gate.RULERS if candidate[r] >= baseline[r]]
    mean = math.fsum(delta.values())/4
    # AUC floating arithmetic only; no rounded/displayed metrics in decisions.
    tool_pass = len(wins) >= 3
    passed = tool_pass and mean >= .002
    return dict(status="SCREEN_PASS_PENDING_REVIEW" if passed else "HOLD",
                tool_pass=tool_pass, wins=wins, delta=delta, mean_delta=mean,
                mean_delta_threshold=.002, policy=POLICY,
                clean_oof=False, exposure="EXPOSED_EXPLORATORY",
                independent_efficacy_evidence=False,
                limitation="fold1-3 weight-generator ancestors saw fold0; outer/nested regeneration required for independent efficacy")


def evaluate(export_root):
    from tools import exp156_k_sweep as rulers
    expected = producer.expected_registration()
    k.need(not expected["test_only"], "test fixture cannot qualify candidate")
    ids = expected["holdout"]
    base = k.parent.OUT
    baseline = dict(k1_csv=base/"holdout.csv",k1_result=base/"result.json")
    gate.validate_k1(baseline,ids,0)
    root = Path(export_root)
    paths = dict(k1_csv=root/"fit/holdout.csv",k1_result=root/"fit/result.json")
    result = k.read(paths["k1_result"])
    k.need(result["status"] == "COMPLETE" and result["cpu_test"] is False
           and result["model_fold"] == 0 and result["holdout_rows"] == 977
           and result["holdout_sha256"] == k.sha(paths["k1_csv"]), "complete production result required")
    binding = gate.validate_host_export(paths,result,expected,0,k)
    refs = rulers.references(ids,root=k.ROOT)
    metrics = {name:gate.macro_aucs(gate.predictions(p["k1_csv"],ids,0),refs)
               for name,p in (("seed0",baseline),("failure_weights",paths))}
    output = decide(metrics["seed0"],metrics["failure_weights"])
    output.update(metrics=metrics,holdout_rows=977,cohort_sha256=k.digest(ids),binding=binding,
                  baseline_receipt_sha256=k.sha(baseline["k1_result"]),
                  candidate_receipt_sha256=k.sha(paths["k1_result"]))
    return output


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    result = evaluate(a.export_root)
    k.write(a.output,result)
    print(json.dumps(result,allow_nan=False))


if __name__ == "__main__": main()
