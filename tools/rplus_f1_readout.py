"""R+ v1.2.1: three contrasts x three views, fold1 and pooled fold0+1.

Pooled AUC uses within-fold ranks, with the unchanged multi-lens rulers and
component bootstrap. Self-test repeats one real smoke prediction; never OOF.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from training.arms import rplus_f1 as c, rplus, rplus_base, lifecycle as life
from tools.exp190f_readout import convert
from tools.rplus_readout import full as old_full


def require_pair(l2, baseline):
    left,right=copy.deepcopy(l2.document),copy.deepcopy(baseline.document)
    c.need(left['arm']=='C-L2-f1' and right['arm']=='C-BASE-f1','paired arms differ')
    for doc in (left,right):
        doc.pop('arm');doc.pop('attempt')
        doc['config']['recipe']['labels']='LABEL_VARIABLE'
        doc['config']['supervision']['kind']='LABEL_VARIABLE'
    c.need(left==right,'non-label pair drift')


def full(spec,outputs):
    c.need(not spec.document['engineering'],'engineering parent is not scientific evidence')
    outputs=old_full(spec,outputs)
    summary=c.read(outputs/'summary.json')
    c.need(summary['bn']['studies']==3455 and summary['bn']['unique_studies']==3455
           and summary['results']['validation_batches']==864,'continuation population differs')
    return outputs


def fold0(root):
    pin=c.read(root/c.BASE/'fold0-evidence.json');life._verify_files(root,pin['files'])
    l2=rplus.Spec((root/pin['specs']['L2']).read_bytes())
    baseline=rplus_base.Spec((root/pin['specs']['BASE']).read_bytes())
    c.need(rplus.Spec.from_document(baseline.document['l2_basis']).sha256==l2.sha256,'fold0 label pair differs')
    old_full(l2,root/pin['paths']['L2']);old_full(baseline,root/pin['paths']['BASE'])
    p=c.read(root/rplus.control.FOLD)
    return p['val_ids'],{k:root/v for k,v in pin['paths'].items()}


def build_screens(out,ids0,ids1,paths0,paths1,self_test=False):
    """Strict disjointness, explicit direction, and identical lenses in both scopes."""
    c.need(len(ids0)==888 and len(ids1)==864 and len(set(ids0+ids1))==1752,'pooled roster is not disjoint 1752')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    scope='PLUMBING_SELF_TEST_NOT_OOF' if self_test else 'PAIRED_FOLD1_AND_POOLED_01_S43'
    receipt=dict(schema='rplus-f1-readout/v1',scope=scope,fold1_rows=864,pooled_rows=1752,files={})
    for population in ('fold1','pooled'):
        for contrast,arm,control in (('L2-minus-P','L2','P'),('BASE-minus-P','BASE','P'),('L2-minus-BASE','L2','BASE')):
            directory=out/population/contrast;directory.mkdir(parents=True)
            for view in ('mean','plain','mirror'):
                pair={'arm':[],'control':[]}
                parts=[(1,ids1,paths1)] if population=='fold1' else [(0,ids0,paths0),(1,ids1,paths1)]
                for number,ids,paths in parts:
                    for side,key in (('arm',arm),('control',control)):
                        pair[side].append(convert(paths[key]/f'predictions-{view}.csv',directory/f'{side}-f{number}-{view}.csv',ids))
                screen=dict(name=f'{scope}-{population}-{contrast}-{view}',family='coatnet',
                    label_change=True,fold_rank=population=='pooled',seeds={'43':pair})
                c.write(directory/f'spec-{view}.json',screen)
                receipt['files'][f'{population}/{contrast}/{view}']=pair
    c.write(out/'adapter-receipt.json',receipt)
    return dict(status='READOUT_SPEC_PREPARED',scope=scope,rows=1752,screens=18)


def prepare(root,binding_root,out,l2_spec,base_spec,l2_outputs=None,base_outputs=None,self_test=False):
    root,binding_root,out=Path(root),Path(binding_root),Path(out)
    require_pair(l2_spec,base_spec)
    p=c.validate_inputs(l2_spec,binding_root);c.validate_inputs(base_spec,binding_root)
    ids0,paths0=fold0(root)
    if self_test:
        c.need(l2_outputs is None and base_outputs is None,'self-test cannot accept fit outputs')
        c.need(l2_spec.document['engineering'],'self-test requires engineering binding')
        fixture=out.parent/(out.name+'-prediction-fixture');fixture.mkdir(exist_ok=False)
        # Expand an actual CPU model output onto the real held-out roster solely
        # to exercise numeric-label joins and pooled screen plumbing.
        import pandas as pd
        for view in ('mean','plain','mirror'):
            frame=pd.read_csv(binding_root/f'parent/outputs/predictions-{view}.csv')
            values=np.repeat(frame[l2_spec.document['config']['targets']].to_numpy('float32')[:1],864,axis=0)
            life.create_new(fixture/f'predictions-{view}.csv',life.prediction_csv_bytes(p['val_ids'],values,l2_spec.document['config']['targets']))
        paths1=dict.fromkeys(('P','L2','BASE'),fixture)
        paths0=dict.fromkeys(('P','L2','BASE'),paths0['P'])
    else:
        c.need(l2_outputs is not None and base_outputs is not None,'both full continuation outputs required')
        paths1={'P':binding_root/'parent/outputs','L2':full(l2_spec,l2_outputs),'BASE':full(base_spec,base_outputs)}
    return build_screens(out,ids0,p['val_ids'],paths0,paths1,self_test)


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('root','binding-root','out','l2-spec','base-spec'):p.add_argument('--'+name,required=True)
    for name in ('l2-outputs','base-outputs'):p.add_argument('--'+name)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    print(json.dumps(prepare(a.root,a.binding_root,a.out,c.Spec(Path(a.l2_spec).read_bytes()),
        c.Spec(Path(a.base_spec).read_bytes()),a.l2_outputs,a.base_outputs,a.self_test)))


if __name__=='__main__':main()
