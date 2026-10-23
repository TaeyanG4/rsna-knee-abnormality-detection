"""EXP155 Gold58 CLI with the authenticated v1.2 release metadata guard.

The registered training driver and its protocol-v1_1 source list stay untouched.
Only an explicit predict-gold58 command can enter the baseline GPU predictor.
"""
from __future__ import annotations

import types

from tools import exp155_own0949_oai_v11 as driver
from tools.exp155_own0949_oai_release_v12 import own_meta, sha


def check_checkpoint(ck, registration, protocol, lane):
    own_meta(ck, sha(protocol), registration["run_dirs"][lane])


def predict_gold58(args):
    environment = dict(driver.base.predict.__globals__,
                       load_registration=driver.load_registration,
                       registered_run=driver.registered_run,
                       check_checkpoint=check_checkpoint,
                       guard_output=driver.scoped_path,
                       write_new_json=driver.write_new_json,
                       CODE=driver.CODE)
    return types.FunctionType(driver.base.predict.__code__, environment)(args, True)


def main(argv=None):
    args = driver.parser().parse_args(argv)
    if args.command != "predict-gold58":
        raise ValueError("This wrapper supports predict-gold58 only")
    return predict_gold58(args)


if __name__ == "__main__":
    main()
