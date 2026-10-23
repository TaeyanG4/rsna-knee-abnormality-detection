"""Focused INF2 tests: static release, original science, arithmetic and journals.

No model construction, model/renderer forwards, labels, GPU, network or admission.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import rankdata

from tools import c85r7_r6w958inf2_v1148 as b


def functions(source):
    tree = ast.parse(source)
    return {node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}


def function_source(source, name):
    return ast.get_source_segment(source, functions(source)[name])+'\n'


def run_functions(source, names, scope=None):
    scope = dict(scope or {})
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes}==set(names)
    exec(compile(ast.Module(nodes,type_ignores=[]),'<selected-static-functions>','exec'),scope)
    return scope


def structural(node): return ast.dump(node,include_attributes=False)


@pytest.fixture(scope='module')
def candidate():
    original=b.parent()[1]
    actual=b.unpack(b.read(b.OUT/'kernel/release.ipynb'))[0]
    return original,actual


def rank(value):
    a=np.asarray(value,np.float64)
    b.need(a.ndim==2 and a.shape[1]==12 and len(a)>0 and np.isfinite(a).all(),'rank shape/value')
    return (rankdata(a,axis=0,method='average')-1)/max(1,len(a)-1)


@pytest.fixture(scope='module')
def aggregators(candidate):
    return [run_functions(files['exp169_partner_bundle_release.py'].decode(),['aggregate'],dict(need=b.need,rank=rank))['aggregate'] for files in candidate]


def inputs(n=17):
    rng=np.random.default_rng(91018)
    return {name:((rng.integers(0,32,(count,n,12)))/31).astype(np.float32)
            for name,count in [('l2',10),('dino',5),('convnext',5),('small',10)]}


def test_exact_stage_and_all_embedded(candidate):
    assert b.validate()['status']=='BUILD_BYTES_VALID'
    original,actual=candidate
    assert len(original)==len(actual)==31 and original.keys()==actual.keys()
    changed={name for name in original if original[name]!=actual[name]}
    assert changed=={'a172-cell.py.txt','exp169_partner_bundle_release.py','exp169_release_c85.py',
        'exp169_release_c85r7.py','exp169_release_c85r7codec.py','exp169_release_p956.py',
        'exp169_release_v2.py','exp169_release_v5q.py','q-a5q-manifest.json'}
    inventory=b.read(b.OUT/'embedded-diff.json')
    assert all(inventory[n]['parent']==b.digest(original[n]) and inventory[n]['candidate']==b.digest(actual[n]) for n in original)
    assert sum(not row['changed'] for row in inventory.values())==22


def test_no_launch_import_or_timing_change(candidate):
    original,actual=candidate
    # All top-level imports and every existing function outside the narrow
    # scientific/reader/receipt changes are AST-identical (hash constants aside).
    allowed={'exp169_partner_bundle_release.py':{'aggregate','_load_models','partner_scores'},
             'exp169_release_c85.py':{'manifest','authenticated_a172'},
             'exp169_release_c85r7codec.py':{'candidate_receipt'}}
    for name in original:
        if not name.endswith('.py'): continue
        old,new=original[name].decode(),actual[name].decode()
        before,after=functions(old),functions(new)
        for func in before:
            if func not in allowed.get(name,set()):
                assert structural(before[func])==structural(after[func]),(name,func)
        def imports(src): return [structural(n) for n in ast.parse(src).body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Try))]
        assert imports(old)==imports(new),name
    # The producer/consumer's queue, threading, joins, CUDA entrypoint, backend,
    # device checks and timers are byte-for-byte identical AST call sites.
    srcs=[functions(f['exp169_partner_bundle_release.py'])['partner_scores'] for f in candidate]
    names={'Thread','Queue','Barrier','Lock','join','put','get','task_done','wait','abort',
           'set_device','set_num_threads','use_deterministic_algorithms','device_count',
           'get_device_name','reset_peak_memory_stats','max_memory_allocated','monotonic'}
    calls=lambda node:[structural(n) for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in names]
    assert calls(srcs[0])==calls(srcs[1])


def test_original_model_and_renderer_ast():
    helper=ast.parse(b.HELPER.read_text(encoding='utf-8'))
    nodes={node.name:node for node in ast.walk(helper) if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    for path,names in [('training/arms/mil.py',['FindingMIL','strict_load','scale_windows']),
                       ('training/src/raptor_exp185.py',['valid_windows','spatial_tokens','finding_attention'])]:
        originals=functions((b.ROOT/path).read_text(encoding='utf-8'))
        for name in names: assert structural(nodes[name])==structural(originals[name]),name
    # Exact original normalization expression; no renderer call is executed.
    original=functions((b.ROOT/'tools/exp169_release_v6.py').read_text())['mil_windows']
    assert structural(nodes['inf2_small_windows'].body[-1])==structural(original.body[-1])


def test_assets_original_ten_and_metadata(candidate):
    original,actual=candidate
    assets=b.asset_manifest(); manifest=json.loads(actual['q-a5q-manifest.json'])
    assert assets['checkpoint_bytes']==1987754780
    assert manifest['entries']==json.loads(original['q-a5q-manifest.json'])['entries']
    assert manifest['inf2_entries']==assets['entries']
    assert [(e['seed'],e['fold']) for e in assets['entries']]==[(s,f) for s in (42,43) for f in range(5)]
    assert len({e['sha256'] for e in assets['entries']})==10
    for entry in assets['entries']:
        identity=entry['source_identity']
        assert entry['sha256']==identity['sha256']==entry['source_sha256']
        assert entry['source']==identity['path'] and entry['spec_sha256']==identity['spec_sha256']
        assert manifest['files'][entry['path']]=={k:entry[k] for k in ('bytes','sha256')}
    for name in ('exp169_partner_bundle_release.py','exp169_release_v2.py'):
        assert manifest['files']['code/'+name]==dict(bytes=len(actual[name]),sha256=b.digest(actual[name]))
    nb=b.read(b.OUT/'kernel/release.ipynb'); parent=b.read(b.BASE/'kernel/release.ipynb')
    assert nb['cells'][0]['source']==['FAST = False\n']
    assert nb['cells'][1]==parent['cells'][1] and len(nb['cells'])==len(parent['cells'])
    for key in set(nb)-{'cells'}: assert nb[key]==parent[key]
    meta=b.read(b.OUT/'kernel/kernel-metadata.json'); pm=b.read(b.BASE/'kernel/kernel-metadata.json')
    assert meta['id']==b.SLUG and meta['title']==b.SLUG.split('/')[1]
    assert meta['dataset_sources']==pm['dataset_sources']+[b.DATASET]
    for key in set(meta)-{'id','title','dataset_sources'}: assert meta[key]==pm[key]


def test_complete_manifest_delta_and_original_source_pins(candidate):
    original,actual=candidate
    old,new=[json.loads(files['q-a5q-manifest.json']) for files in candidate]
    assert set(new)-set(old)=={'inf2_entries','inf2_serving','inf2_asset_manifest_sha256'}
    for key in set(old)-{'files','source_code'}: assert new[key]==old[key],key
    assert set(new['files'])-set(old['files'])=={entry['path'] for entry in new['inf2_entries']}
    for key in old['files']:
        if key not in ('code/exp169_partner_bundle_release.py','code/exp169_release_v2.py'):
            assert new['files'][key]==old['files'][key],key
    assert new['source_code'].keys()==old['source_code'].keys()
    for key in old['source_code']:
        if key not in ('exp169_partner_bundle_release.py','exp169_release_v2.py'):
            assert new['source_code'][key]==old['source_code'][key],key
    for name,pin in b.asset_manifest()['original_source_pins'].items():
        path=b.ROOT/name
        assert path.stat().st_size==pin['bytes'] and b.sha(path)==pin['sha256'],name


def test_complete_outer_notebook_code_delta():
    parent=b.read(b.BASE/'kernel/release.ipynb'); child=b.read(b.OUT/'kernel/release.ipynb')
    _,old_blob,old=b.unpack(parent); _,new_blob,new=b.unpack(child)
    source=''.join(child['cells'][2]['source'])
    source=b.replace(source,repr(new['BLOB']),repr(old['BLOB']))
    source=b.replace(source,'PINS = '+repr(new['PINS']),'PINS = '+repr(old['PINS']))
    source=b.replace(source,repr(b.digest(new_blob)),repr(b.digest(old_blob)))
    assert source==''.join(parent['cells'][2]['source'])
    for index in (0,2):
        for key in set(parent['cells'][index])-{'source'}:
            assert parent['cells'][index][key]==child['cells'][index][key]


@pytest.mark.parametrize('n',[1,2,9,37])
def test_registered_q2_arithmetic_protected_columns(aggregators,n):
    old,new=aggregators; raw=inputs(n)
    parent=old({k:v for k,v in raw.items() if k!='small'})[0]
    actual,families=new(raw)
    r=rank(((raw['l2'][:5]+raw['l2'][5:])/2).mean(0))
    d=rank(raw['dino'].mean(0)); c=rank(raw['convnext'].mean(0))
    s=rank(((raw['small'][:5]+raw['small'][5:])/2).mean(0)); s[:,8:]=d[:,8:]
    expected=rank((r+s+c)/3)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual[:,8:],parent[:,8:])
    np.testing.assert_array_equal(families['l2'],r)
    np.testing.assert_array_equal(families['convnext'],c)
    if n>2: assert np.any(actual[:,:8]!=parent[:,:8])


@pytest.mark.parametrize('failure',['missing','nan','infinite','negative','above_one','wrong_count','wrong_columns','wrong_rows','float64'])
def test_small_invalid_never_omitted(aggregators,failure):
    raw=inputs()
    if failure=='missing': raw.pop('small')
    elif failure in ('nan','infinite','negative','above_one'):
        raw['small'][3,2,5]={'nan':np.nan,'infinite':np.inf,'negative':-.01,'above_one':1.01}[failure]
    elif failure=='wrong_count': raw['small']=raw['small'][:-1]
    elif failure=='wrong_columns': raw['small']=raw['small'][:,:,:11]
    elif failure=='wrong_rows': raw['small']=raw['small'][:,:-1]
    else: raw['small']=raw['small'].astype(np.float64)
    with pytest.raises(ValueError): aggregators[1](raw)


def meta_scope():
    labels=b.read(b.BRIDGE/'missing-serving-inputs.json')['gold_roster']['targets']
    scope=run_functions(b.HELPER.read_text(),['inf2_small_meta'],dict(TARGETS=labels,need=b.need))
    entry=b.asset_manifest()['entries'][0]
    ck=dict(arch=entry['arch'],res=224,lab=labels,contract='raptor64_336_c140_s02',norm='imagenet',
        window_policy='within-slot-valid',fold=0,epoch=19,model_kind='finding_spatial_tokens_v1',
        input_policy='center3',k_eval=24,flip_tta='coronal_axial_horizontal_mean_probability',
        provenance=dict(spec_sha256=entry['spec_sha256']))
    return scope['inf2_small_meta'],entry,ck


@pytest.mark.parametrize('field',['arch','res','lab','contract','norm','window_policy','fold','epoch','model_kind','input_policy','k_eval','flip_tta','provenance'])
def test_metadata_rejects_variant_mismatch(field):
    check,entry,ck=meta_scope(); check(ck,entry)
    ck[field]={} if field=='provenance' else 'incorrect'
    with pytest.raises(ValueError): check(ck,entry)


def test_compiles_actual_dynamic_q_and_a172(candidate):
    actual=candidate[1]
    v=run_functions(actual['exp169_release_v5q.py'].decode(),['_replace_once','a172_function_source','a172_runtime_source'],
        dict(require=b.need,hashlib=hashlib,A172_CELL_SHA=b.digest(actual['a172-cell.py.txt'])))
    c=run_functions(actual['exp169_release_c85.py'].decode(),['q_runtime_source','a172_source'],dict(v5q=SimpleNamespace(**v),need=b.need))
    src=function_source(actual['exp169_partner_bundle_release.py'].decode(),'partner_scores')
    generated=c['q_runtime_source'](v['a172_function_source']('partner_scores',src))
    assert "manifest['inf2_entries']" in generated and "journal.record(ids[i], str(m), value)" in generated
    assert 'deadline exceeded' not in generated and 'timeout=600' not in generated
    compile(generated,'<inf2-actual-q>','exec')
    a=c['a172_source'](actual['a172-cell.py.txt'].decode(),Path('/kaggle/working/c85r7-run/a172'))
    assert a.count('_c85.overlay_inf2(_partner)')==1
    compile(a,'<inf2-actual-a172>','exec')
    # The inspect-based original loader also compiles with the new helper branch.
    loader=function_source(actual['exp169_partner_bundle_release.py'].decode(),'_load_models')
    loader=b.replace(loader,"need(ck.get('source_sha256') == entry['source_sha256'],",
        "need((entry.get('format') == 'authenticated-original' and sha(path) == entry['source_sha256']) or ck.get('source_sha256') == entry['source_sha256'],")
    compile(loader,'<inf2-actual-v2-loader>','exec')


def authenticated_fixture(candidate,aggregators,tmp_path,failed=None):
    actual=candidate[1]; source=actual['exp169_release_c85.py'].decode()
    manifest=json.loads(actual['q-a5q-manifest.json']); msha=b.digest(actual['q-a5q-manifest.json'])
    targets=b.asset_manifest()['entries']; raw=inputs(7); ids=[f'synthetic-{i}' for i in range(7)]
    (tmp_path/'q-a5q-manifest.json').write_bytes(actual['q-a5q-manifest.json'])
    def prediction(values):
        a=np.asarray(values,np.float64); b.need(a.shape==(12,) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),'invalid prediction'); return a
    def validate_ids(values):
        b.need(len(values)==len(set(values)),'duplicate IDs'); return values
    def check_pin(path,pin): b.need(b.sha(path)==pin,'pin mismatch')
    rname,rsha=next(iter(manifest['anchor_binding']['checkpoint_sha256'].items()))
    scope=run_functions(source,['manifest','optional_json','read_journal','clean_member','array_sha','telemetry','authenticated_a172'],
        dict(Path=Path,json=json,hashlib=hashlib,need=b.need,read_json=b.read,MANIFEST_SHA=msha,
             p=SimpleNamespace(check_pin=check_pin,R_NAME=rname,R_SHA=rsha,RBinding=SimpleNamespace(prediction=prediction)),
             v5q=SimpleNamespace(validate_ids=validate_ids,_read_csv=lambda p: (_ for _ in ()).throw(ValueError('synthetic: no saved CSV'))),
             a172=SimpleNamespace(aggregate=aggregators[1])))
    for folder,identity in [('exp169-trio',manifest['anchor_binding']),('q-journal',{'manifest_sha256':msha})]:
        path=tmp_path/folder; path.mkdir(); (path/'ids.json').write_bytes(b.encoded(ids)); (path/'journal-identity.json').write_bytes(b.encoded(identity))
    own=[dict(uid=uid,member=name,values=[.4]*12) for uid in reversed(ids) for name in ('EXP143v2','EXP155')]
    (tmp_path/'exp169-trio/predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in own))
    rows=[]
    for member,(family,index) in enumerate([(f,k) for f,n in [('l2',10),('dino',5),('convnext',5),('small',10)] for k in range(n)]):
        for i,uid in reversed(list(enumerate(ids))):
            row=dict(uid=uid,member=str(member),values=raw[family][index,i].tolist())
            if failed and failed==(member,i): row=dict(uid=uid,member=str(member),error='synthetic failed Small member')
            rows.append(row)
    (tmp_path/'q-journal/predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return scope['authenticated_a172'](tmp_path),raw,ids


def test_actual_journal_reader_all30_uid_alignment(candidate,aggregators,tmp_path):
    result,raw,ids=authenticated_fixture(candidate,aggregators,tmp_path)
    assert result[0]==ids and result[4]['failures']['Q_A5q']=={}
    np.testing.assert_array_equal(result[3],aggregators[1](raw)[0])


def test_small_failure_uses_existing_q_branch_semantics(candidate,aggregators,tmp_path):
    result,raw,ids=authenticated_fixture(candidate,aggregators,tmp_path,(26,3))
    assert set(result[4]['failures']['Q_A5q'])=={ids[3]}
    good=np.arange(7)!=3
    np.testing.assert_array_equal(result[3][good],aggregators[1]({k:v[:,good] for k,v in raw.items()})[0])
    np.testing.assert_array_equal(result[3][3],np.full(12,.5))
    assert result[4]['failures']['EXP143v2']==result[4]['failures']['EXP155']=={}


def test_no_model_import_or_adapter_dependency(candidate):
    text='\n'.join(data.decode() for data in candidate[1].values())
    assert 'hb199_inf2_serving_v1' not in text
    # CPU test suite never imports torch/timm or executes load/render functions.
    assert 'torch' not in sys.modules and 'timm' not in sys.modules
