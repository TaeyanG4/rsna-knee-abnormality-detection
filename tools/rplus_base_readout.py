"""R+ v1.1.0: C-L2 minus C-BASE label proxy and C-BASE minus parent harm screen."""
import argparse
import json
from pathlib import Path
from training.arms import rplus_base as c, rplus as original, lifecycle as life
from tools.rplus_readout import full
from tools.exp190f_readout import convert


def require_pair(spec, l2):
    c.need(l2 is not None and l2.sha256==original.Spec.from_document(spec.document['l2_basis']).sha256,
           'C-L2 must match embedded continuation control exactly')


def prepare(spec,root,out,arm_outputs=None,l2_spec=None,l2_outputs=None,control_self_test=False):
    root,out=Path(root),Path(out)
    p=c.validate_inputs(spec,root)
    life._verify_files(root,c.read(root/c.o.PRESET)['control_predictions'])
    c.need(bool(arm_outputs)!=control_self_test, 'choose full outputs or explicit control self-test')
    if control_self_test:
        c.need(l2_spec is None and l2_outputs is None,'self-test cannot supply continuation results')
        l2=baseline=root/c.o.CONTROL
    else:
        require_pair(spec,l2_spec)
        c.need(l2_outputs is not None, 'C-L2 outputs required')
        original.validate_inputs(l2_spec,root)
        baseline=full(spec,arm_outputs)
        l2=full(l2_spec,l2_outputs)
    out.mkdir(parents=True,exist_ok=False)
    scope='PLUMBING_CONTROL_SELF_TEST' if control_self_test else 'PAIRED_FOLD0_S43_LABEL_CONTINUATION'
    receipt=dict(schema='rplus-base-readout/v1',scope=scope,spec_sha256=spec.sha256,
                 l2_spec_sha256=l2_spec.sha256 if l2_spec else None,files={})
    for label,arm,control,directory in (
        ('C-L2-minus-C-BASE',l2,baseline,out),
        ('C-BASE-minus-P',baseline,root/c.o.CONTROL,out/'base-minus-parent')):
        directory.mkdir(exist_ok=True)
        for mode in ('mean','plain','mirror'):
            filename=f'predictions-{mode}.csv'
            pair=dict(arm=[convert(arm/filename,directory/f'arm-{mode}.csv',p['val_ids'])],
                      control=[convert(control/filename,directory/f'control-{mode}.csv',p['val_ids'])])
            c.write(directory/f'spec-{mode}.json',dict(name=label+'-'+mode,
                family='coatnet',label_change=True,seeds={'43':pair}))
            receipt['files'][label+'-'+mode]=pair
    c.write(out/'adapter-receipt.json',receipt)
    return dict(status='READOUT_SPEC_PREPARED',scope=scope,rows=len(p['val_ids']))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('spec','root','out'): p.add_argument('--'+name,required=True)
    for name in ('arm-outputs','l2-spec','l2-outputs'): p.add_argument('--'+name)
    p.add_argument('--control-self-test',action='store_true');a=p.parse_args()
    spec=c.Spec(Path(a.spec).read_bytes())
    l2=original.Spec(Path(a.l2_spec).read_bytes()) if a.l2_spec else None
    print(json.dumps(prepare(spec,a.root,a.out,a.arm_outputs,l2,a.l2_outputs,a.control_self_test)))


if __name__=='__main__': main()
