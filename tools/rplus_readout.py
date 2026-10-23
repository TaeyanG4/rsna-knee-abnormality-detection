"""R+ v1.0.0: paired continuation readouts; full results only, or explicit self-test."""
import argparse
import json
from pathlib import Path
from training.arms import rplus as c, lifecycle as life
from tools.exp190f_readout import convert


def full(spec, outputs):
    outputs=Path(outputs)
    summary,prov=[c.read(outputs/f) for f in ('summary.json','provenance.json')]
    c.need(summary['status']=='COMPLETE_TRAINING_AND_READOUT' and summary['mode']=='full'
           and summary['updates']==648 and summary['skips']==0, 'full continuation output required')
    c.need(prov['spec_sha256']==spec.sha256 and prov['scientific']==spec.document['config']
           and prov['verification_sha256'] is not None, 'continuation provenance differs')
    return outputs


def prepare(spec,root,out,arm_outputs=None,control_spec=None,control_outputs=None,control_self_test=False):
    root,out=Path(root),Path(out)
    p=c.validate_inputs(spec,root)
    life._verify_files(root,c.read(root/c.o.PRESET)['control_predictions'])
    c.need(bool(arm_outputs)!=control_self_test, 'choose full outputs or explicit control self-test')
    if control_self_test:
        c.need(control_spec is None and control_outputs is None,'self-test cannot supply continuation control')
        arm=control=root/c.o.CONTROL
    else:
        arm=full(spec,arm_outputs)
        if spec.document['arm']=='C-OAI':
            c.need(control_spec is not None and control_outputs is not None,'C-OAI requires C-L2 control')
            c.validate_inputs(control_spec,root)
            c.need(control_spec.document['arm']=='C-L2','wrong continuation control')
            c.need(control_spec.document['preset']['parent_sha256']==spec.document['preset']['parent_sha256'],'different parents')
            control=full(control_spec,control_outputs)
        else:
            c.need(control_spec is None and control_outputs is None,'C-L2 compares to parent')
            control=root/c.o.CONTROL
    out.mkdir(parents=True,exist_ok=False)
    scope='PLUMBING_CONTROL_SELF_TEST' if control_self_test else 'PAIRED_FOLD0_S43_CONTINUATION'
    receipt=dict(schema='rplus-readout/v1',scope=scope,spec_sha256=spec.sha256,
                 control_spec_sha256=control_spec.sha256 if control_spec else None,files={})
    for mode in ('mean','plain','mirror'):
        filename=f'predictions-{mode}.csv'
        pair=dict(arm=[convert(arm/filename,out/f'arm-{mode}.csv',p['val_ids'])],
                  control=[convert(control/filename,out/f'control-{mode}.csv',p['val_ids'])])
        c.write(out/f'spec-{mode}.json',dict(name=spec.document['arm']+'-continuation-'+mode,
            family='coatnet',label_change=spec.document['arm']=='C-OAI',seeds={'43':pair}))
        receipt['files'][mode]=pair
    c.write(out/'adapter-receipt.json',receipt)
    return dict(status='READOUT_SPEC_PREPARED',scope=scope,rows=len(p['val_ids']))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('spec','root','out'): p.add_argument('--'+name,required=True)
    for name in ('arm-outputs','control-spec','control-outputs'): p.add_argument('--'+name)
    p.add_argument('--control-self-test',action='store_true');a=p.parse_args()
    spec=c.Spec(Path(a.spec).read_bytes())
    control=c.Spec(Path(a.control_spec).read_bytes()) if a.control_spec else None
    print(json.dumps(prepare(spec,a.root,a.out,a.arm_outputs,control,a.control_outputs,a.control_self_test)))


if __name__=='__main__': main()
