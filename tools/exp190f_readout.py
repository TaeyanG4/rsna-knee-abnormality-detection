"""EXP190F readout adapter v1.0.0: byte-preserving UID header conversion.

Changelog: immutable paired screen specs; reject engineering smoke as OOF.
"""
import argparse
from pathlib import Path
from training.arms import exp190f as c, lifecycle as life
from tools import multi_lens_readout as readout


def convert(source, destination, ids):
    life.load_prediction_csv(source, ids, c.control.LAB)
    raw = Path(source).read_bytes()
    header, rest = raw.split(b'\n', 1)
    c.need(header.startswith(b'study_id,'), 'prediction ID header differs')
    life.create_new(destination, b'StudyInstanceUID,'+header[len(b'study_id,'):]+b'\n'+rest)
    return dict(path=Path(destination).resolve().as_posix(), sha256=life.file_sha256(destination))


def prepare(spec, root, out, arm_outputs=None, control_self_test=False):
    root, out = Path(root), Path(out)
    p = c.validate_inputs(spec, root)
    preset = c.read(root/c.PRESET)
    life._verify_files(root, preset['control_predictions'])
    c.need(bool(arm_outputs) != control_self_test, 'choose full arm outputs or explicit control self-test')
    if arm_outputs:
        arm_outputs = Path(arm_outputs)
        summary, prov = [c.read(arm_outputs/f) for f in ('summary.json','provenance.json')]
        c.need(summary['status']=='COMPLETE_TRAINING_AND_READOUT' and summary['mode']=='full'
               and summary['updates']==3456 and summary['skips']==0, 'full arm output required')
        c.need(prov['spec_sha256']==spec.sha256 and prov['scientific']==spec.document['config']
               and prov['verification_sha256'] is not None, 'arm provenance differs')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='exp190f-readout-adapter/v1', spec_sha256=spec.sha256,
                  scope='PLUMBING_CONTROL_SELF_TEST' if control_self_test else 'PAIRED_FOLD0_S43', files={})
    for mode in ('mean','plain','mirror'):
        filename = 'predictions-'+mode+'.csv'
        control = root/c.CONTROL/filename
        arm = control if control_self_test else arm_outputs/filename
        pair = dict(arm=[convert(arm,out/f'arm-{mode}.csv',p['val_ids'])],
                    control=[convert(control,out/f'control-{mode}.csv',p['val_ids'])])
        screen = dict(name='EXP190F-S43-vs-CONTROL43-'+mode, family='coatnet',label_change=True,
                      seeds={'43':pair})
        c.write(out/f'spec-{mode}.json',screen)
        result['files'][mode] = dict(arm_original_sha256=life.file_sha256(arm),
            control_original_sha256=life.file_sha256(control), pair=pair)
    c.write(out/'adapter-receipt.json',result)
    return dict(status='READOUT_SPEC_PREPARED',scope=result['scope'],rows=len(p['val_ids']))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('spec','root','out'): p.add_argument('--'+name,required=True)
    p.add_argument('--arm-outputs'); p.add_argument('--control-self-test',action='store_true')
    a=p.parse_args()
    import json
    print(json.dumps(prepare(c.Spec(Path(a.spec).read_bytes()),a.root,a.out,a.arm_outputs,a.control_self_test)))


if __name__=='__main__': main()
