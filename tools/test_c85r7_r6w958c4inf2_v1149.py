"""Focused H207 joint tests: static release, original science, arithmetic and journals.

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

from tools import c85r7_r6w958c4inf2_v1149 as b


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
    original=b.components()[1]
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
    allowed={'exp169_partner_bundle_release.py':{'aggregate','_load_models','partner_scores','model_meta'},
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
    helper=ast.parse(b.small.HELPER.read_text(encoding='utf-8'))
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
    assets=b.small.asset_manifest(); manifest=json.loads(actual['q-a5q-manifest.json'])
    assert assets['checkpoint_bytes']==1987754780
    assert manifest['entries'][:15]==json.loads(original['q-a5q-manifest.json'])['entries'][:15]
    assert manifest['entries'][15:]==b.read(b.c4.C4_DIR/'release-manifest.json')['entries']
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
    assert meta['dataset_sources']==pm['dataset_sources']+[b.c4.DATASET,b.small.DATASET]
    for key in set(meta)-{'id','title','dataset_sources'}: assert meta[key]==pm[key]


def test_complete_manifest_delta_and_original_source_pins(candidate):
    original,actual=candidate
    old,new=[json.loads(files['q-a5q-manifest.json']) for files in candidate]
    assert set(new)-set(old)=={'inf2_entries','inf2_serving','inf2_asset_manifest_sha256','c4_manifest_sha256'}
    for key in set(old)-{'files','source_code','entries'}: assert new[key]==old[key],key
    assert set(new['files'])-set(old['files'])=={entry['path'] for entry in new['inf2_entries']}
    for key in old['files']:
        if key not in ('code/exp169_partner_bundle_release.py','code/exp169_release_v2.py') and key not in {e['path'] for e in new['entries'][15:]}:
            assert new['files'][key]==old['files'][key],key
    assert new['source_code'].keys()==old['source_code'].keys()
    for key in old['source_code']:
        if key not in ('exp169_partner_bundle_release.py','exp169_release_v2.py'):
            assert new['source_code'][key]==old['source_code'][key],key
    for name,pin in b.small.asset_manifest()['original_source_pins'].items():
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
    labels=b.read(b.small.BRIDGE/'missing-serving-inputs.json')['gold_roster']['targets']
    scope=run_functions(b.small.HELPER.read_text(),['inf2_small_meta'],dict(TARGETS=labels,need=b.need))
    entry=b.small.asset_manifest()['entries'][0]
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
    assert a.count('_c85.overlay_c4inf2(_partner)')==1
    compile(a,'<inf2-actual-a172>','exec')
    # The inspect-based original loader also compiles with the new helper branch.
    loader=function_source(actual['exp169_partner_bundle_release.py'].decode(),'_load_models')
    loader=b.replace(loader,"need(ck.get('source_sha256') == entry['source_sha256'],",
        "need((entry.get('format') == 'authenticated-original' and sha(path) == entry['source_sha256']) or ck.get('source_sha256') == entry['source_sha256'],")
    compile(loader,'<inf2-actual-v2-loader>','exec')


def authenticated_fixture(candidate,aggregators,tmp_path,failed=None):
    actual=candidate[1]; source=actual['exp169_release_c85.py'].decode()
    manifest=json.loads(actual['q-a5q-manifest.json']); msha=b.digest(actual['q-a5q-manifest.json'])
    targets=b.small.asset_manifest()['entries']; raw=inputs(7); ids=[f'synthetic-{i}' for i in range(7)]
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


def test_fixed_four_arms_joint_reductions_precision_and_permutation(candidate):
    _,original,_,_,_,_,s,c=b.components()
    joint=candidate[1]
    aggs=[run_functions(f['exp169_partner_bundle_release.py'].decode(),['rank','aggregate'],
                       dict(need=b.need))['aggregate'] for f in (original,c,s,joint)]
    raw=inputs(58); c4raw=np.random.default_rng(71).random((5,58,12),dtype=np.float32)
    base={k:v for k,v in raw.items() if k!='small'}
    four=[aggs[0](base)[0],aggs[1]({**base,'convnext':c4raw})[0],
          aggs[2](raw)[0],aggs[3]({**raw,'convnext':c4raw})[0]]
    np.testing.assert_array_equal(four[3][:,8:],four[1][:,8:])
    # C4=C reduces joint exactly to INF2; S=D reduces exactly to C4.
    np.testing.assert_array_equal(aggs[3](raw)[0],four[2])
    same_small=np.concatenate([raw['dino'],raw['dino']])
    np.testing.assert_array_equal(aggs[3]({**raw,'small':same_small,'convnext':c4raw})[0],four[1])
    np.testing.assert_array_equal(aggs[3]({**raw,'small':same_small})[0],four[0])
    permutation=np.random.default_rng(2).permutation(58)
    perm={k:v[:,permutation] for k,v in {**raw,'convnext':c4raw}.items()}
    np.testing.assert_array_equal(aggs[3](perm)[0],four[3][permutation])
    adversarial=np.random.default_rng(72).random((10,58,12),dtype=np.float32)
    assert not np.array_equal(((adversarial[:5]+adversarial[5:])/2).mean(0),adversarial.mean(0))
    _,family=aggs[3]({**raw,'small':adversarial})
    np.testing.assert_array_equal(family['dino'][:,:8],rank(((adversarial[:5]+adversarial[5:])/2).mean(0))[:,:8])


@pytest.mark.parametrize('member',[15,19,20,29])
def test_any_c4_or_small_failure_preserves_existing_whole_q_failure(candidate,aggregators,tmp_path,member):
    result,raw,ids=authenticated_fixture(candidate,aggregators,tmp_path,(member,2))
    assert set(result[4]['failures']['Q_A5q'])=={ids[2]}
    good=np.arange(7)!=2
    np.testing.assert_array_equal(result[3][good],aggregators[1]({k:v[:,good] for k,v in raw.items()})[0])
    np.testing.assert_array_equal(result[3][2],np.full(12,.5))


@pytest.mark.parametrize('mutation',['notebook','metadata','receipt'])
def test_exact_stage_validator_rejects_unregistered_mutation(tmp_path,monkeypatch,mutation):
    target=tmp_path/'stage';(target/'kernel').mkdir(parents=True)
    for name in ('kernel/release.ipynb','kernel/kernel-metadata.json','build-receipt.json','embedded-diff.json'):
        (target/name).write_bytes((b.OUT/name).read_bytes())
    if mutation=='notebook':
        path=target/'kernel/release.ipynb';value=b.read(path);value['cells'][1]['source'].append('# unregistered launch change\n')
    elif mutation=='metadata':
        path=target/'kernel/kernel-metadata.json';value=b.read(path);value['title']='wrong/slug'
    else:
        path=target/'embedded-diff.json';value=b.read(path);value['exp169_r224.py']['changed']=True
    path.write_bytes(b.encoded(value))
    monkeypatch.setattr(b,'OUT',target)
    with pytest.raises(ValueError):b.validate()


def test_pre_metric_registration_and_exact_prospective_rule():
    r=b.read(b.WORK/'JOINT-REGISTRATION.json')
    assert r['metrics_executed'] is False and r['roster']['rows']==4343
    assert [x['rows'] for x in r['roster']['folds']]==[888,864,864,864,863]
    assert r['bootstrap']['draws']==300 and r['bootstrap']['seed']==20261003
    assert r['targets']==b.TARGETS and r['small_columns']==list(range(8))
    assert r['overlap']['novel_mechanism'] is False and len(r['lenses'])==7
    points={k:np.full((4,2),.8) for k in ('Q0','Q4','Q2','Q24')}
    contrast={k:np.zeros((4,12)) for k in ('Q24-Q0','Q24-Q4','Q24-Q2')}
    assert b.decision_readout(points,contrast)=='REPORT_FOR_B_HOLISTIC_JUDGMENT'
    points['Q24'][0,0]=.79
    assert b.decision_readout(points,contrast)=='NO_DEMONSTRATED_INCREMENTAL_ROLE'
    points['Q24'][1,1]=.81
    assert b.decision_readout(points,contrast)=='REPORT_FOR_B_HOLISTIC_JUDGMENT'
    assert b.decision_readout(points,{k:v for k,v in contrast.items() if k!='Q24-Q2'}).startswith('HOLD')
    assert b.decision_readout(points,contrast,valid=False).startswith('HOLD')
    contrast['Q24-Q4'][0,0]=np.nan
    assert b.decision_readout(points,contrast).startswith('HOLD')


@pytest.mark.parametrize('broken',[None,'c4_missing','small_missing','c4_hash'])
def test_coherent_overlay_authenticates_once_and_never_restores_other_manifest(tmp_path,monkeypatch,broken):
    import zipfile
    root=tmp_path/'runtime';root.mkdir();partner=tmp_path/'partner';(partner/'models').mkdir(parents=True);(partner/'code').mkdir()
    def entries(prefix,n,family):
        out=[]
        for i in range(n):
            data=(prefix+str(i)).encode();out.append(dict(family=family,seed=42,fold=i,
                path='models/'+prefix+str(i)+'.pt',bytes=len(data),sha256=b.digest(data),source_sha256=b.digest(data)))
        return out
    retained=entries('retained',15,'retained');old=entries('plain',5,'convnext')
    c4e=entries('c4',5,'convnext');se=entries('small',10,'small')
    original={'entries':retained+old};raw_original=b.encoded(original)
    (partner/'release-manifest.json').write_bytes(raw_original)
    cm=dict(entries=c4e,crop_policy='mixed',k_eval=36,tta='plane-aware-coronal-axial-flip')
    assets=dict(entries=se)
    candidate=dict(entries=retained+c4e,inf2_entries=se,files={})
    for name in ('exp169_partner_bundle_release.py','exp169_release_v2.py'):
        data=('joint-'+name).encode();(root/name).write_bytes(data)
        candidate['files']['code/'+name]=dict(bytes=len(data),sha256=b.digest(data))
    (root/'q-a5q-manifest.json').write_bytes(b.encoded(candidate))
    for name,manifest,es in [('exp169-c4-v4.zip.bin',cm,c4e),('inf2-small10.zip.bin',assets,se)]:
        manifest_name='release-manifest.json' if name.startswith('exp169') else 'inf2-small-manifest.json'
        with zipfile.ZipFile(tmp_path/name,'w') as z:
            z.writestr(manifest_name,b.encoded(manifest))
            for i,e in enumerate(es):
                if i==0 and ((broken=='c4_missing' and es is c4e) or (broken=='small_missing' and es is se)):continue
                data=e['path'].split('/')[-1].removesuffix('.pt').encode()
                if i==0 and broken=='c4_hash' and es is c4e:data=b'corrupt'
                z.writestr(e['path'],data)
    actual_sha=b.sha
    monkeypatch.setattr(b.small,'Q_SHA',b.digest(raw_original))
    monkeypatch.setattr(b.c4,'ARCHIVE_SHA',actual_sha(tmp_path/'exp169-c4-v4.zip.bin'))
    monkeypatch.setattr(b,'SMALL_ARCHIVE_SHA',actual_sha(tmp_path/'inf2-small10.zip.bin'))
    monkeypatch.setattr(b,'sha',lambda p:b.digest(b.encoded(cm)) if Path(p)==b.c4.C4_DIR/'release-manifest.json' else actual_sha(p))
    checks=[]
    def checked(path,expected):b.need(actual_sha(path)==expected,'fixture pin')
    def first(path,value):checks.append(('original',copy.deepcopy(value)))
    def final(path,value):checks.append(('joint',copy.deepcopy(value)))
    scope=run_functions(b.overlay_source(assets,cm),['overlay_c4inf2'],dict(Path=Path,json=json,hashlib=hashlib,
        zipfile=zipfile,os=__import__('os'),need=b.need,read_json=b.read,__file__=str(root/'exp169_release_c85.py'),
        p=SimpleNamespace(check_pin=checked),v5q=SimpleNamespace(_mount=lambda *a:tmp_path),
        manifest=b.read,a172=SimpleNamespace(base=SimpleNamespace(verify_assets=first),verify_assets=final)))
    if broken:
        with pytest.raises(ValueError):scope['overlay_c4inf2'](partner)
        assert b.read(partner/'release-manifest.json')==original
        assert [x[0] for x in checks]==['original']
    else:
        scope['overlay_c4inf2'](partner)
        assert checks==[('original',original),('joint',candidate)]
        assert b.read(partner/'release-manifest.json')==candidate
        for e in c4e+se:assert actual_sha(partner/e['path'])==e['sha256']
