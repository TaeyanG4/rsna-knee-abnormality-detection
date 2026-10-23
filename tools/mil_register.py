"""MIL registration CLI v1.1.0 (2026-10-11): CPU-only, create-only registration.

Changelog: --kind member binds a recipe and declared target without controls.

Staging/admission/receipts use tools.arm_lifecycle; execution uses
training.runners.mil_arms. No source generation or inherited guard chain.
"""
import argparse

from training.arms import lifecycle as life
from training.arms import mil_registration


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "arm", "attempt", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--kind", choices=("experiment", "member"), default="experiment")
    experiment = ("protocol", "protocol-sha256", "control", "summary-sha256", "provenance-sha256")
    member = ("recipe", "recipe-sha256", "target", "target-sha256")
    for name in experiment + member:
        ap.add_argument("--" + name)
    a = ap.parse_args(argv)
    try:
        required, forbidden = (member, experiment) if a.kind == "member" else (experiment, member)
        life._need(all(getattr(a, n.replace("-", "_")) is not None for n in required), "registration arguments missing")
        life._need(all(getattr(a, n.replace("-", "_")) is None for n in forbidden), "mixed registration kinds")
        if a.kind == "member":
            spec = mil_registration.register_member(a.root, recipe=a.recipe, recipe_sha256=a.recipe_sha256,
                target=a.target, target_sha256=a.target_sha256, arm=a.arm, attempt=a.attempt)
        else:
            spec = mil_registration.migrate(a.root, protocol=a.protocol, protocol_sha256=a.protocol_sha256,
                control=a.control, control_hashes={"summary.json": a.summary_sha256,
                                                   "provenance.json": a.provenance_sha256},
                arm=a.arm, attempt=a.attempt)
        life.register(spec.document, a.out)
        print(life.canonical_bytes(dict(status="REGISTERED_NO_EXECUTION", spec_sha256=spec.sha256)).decode())
        return 0
    except (life.ContractError, OSError, ValueError, KeyError, TypeError) as error:
        print(life.canonical_bytes(dict(status="REJECTED", error_type=type(error).__name__)).decode())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
