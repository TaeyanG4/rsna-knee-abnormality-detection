"""HB105 production member contracts and actual executed CPU evidence."""
import ast
import inspect
from pathlib import Path
import numpy as np
import pytest
from training.arms import lifecycle as life, mil, mil_small, mil_small_members as small
from tools import hb105_members as ops, hb105_members_readout as ro

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT/'storage/cache/derived/RAPTOR64_336_C140_S02'


@pytest.mark.parametrize('seed,fold',ops.FITS)
def test_real_member_recipe(seed,fold):
    spec=ops.spec_for(ROOT,seed,fold)
    old=life.ArmSpec((ROOT/f'training/configs/hb105-small-s{seed}-f0-v1.json').read_bytes())
    tiny=life.ArmSpec((ROOT/f'training/configs/mil-member-s43-f{fold}-v1.json').read_bytes())
    assert spec.run_kind=='member' and not spec.document['inputs'].get('controls')
    for k in ('model','windows','sampler','readout','targets','bn','backend_policy'):
        assert spec.document['config'][k]==old.document['config'][k]
    for k in ('roots','updates','labels'):
        assert spec.document['config'][k]==tiny.document['config'][k]
    _,train,val,_,_=small.validate_data(spec,ROOT,CORPUS)
    assert not set(train)&set(val) and len(train)+len(val)==4343


def test_run_scientific_ast_parity():
    """Exact H105 fit AST after stripping only resume/retention/docstring."""
    old=ast.parse(inspect.getsource(mil_small.run))
    new=ast.parse(inspect.getsource(small.run))
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            pairs=list(zip(node.args.kwonlyargs,node.args.kw_defaults))
            node.args.kwonlyargs=[a for a,d in pairs if a.arg!='resume']
            node.args.kw_defaults=[d for a,d in pairs if a.arg!='resume']
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str): return None
            if isinstance(node.value,ast.Call):
                fun=node.value.func
                if isinstance(fun,ast.Name) and fun.id=='save_latest_epoch': return None
                if isinstance(fun,ast.Attribute) and isinstance(fun.value,ast.Name) and fun.value.id=='work': return None
            return self.generic_visit(node)
        def visit_If(self,node):
            if 'resume' in ast.unparse(node.test): return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0])=='(out, work)':
                return ast.parse('out = destination / "outputs"').body[0]
            return node
    assert ast.dump(Strip().visit(old),include_attributes=False)==ast.dump(Strip().visit(new),include_attributes=False)
    for name in ('build_model','FindingMIL','MILWindows','optimizer_stack','train_epoch','evaluate','record_smoke'):
        assert getattr(small,name) is getattr(mil,name)


@pytest.mark.parametrize('key,value',[('fold',0),('fold',5),('seed',44),('run_kind','experiment')])
def test_bad_identity(key,value):
    doc=ops.spec_for(ROOT,42,1).document;doc[key]=value
    with pytest.raises(life.ContractError): small.validate_recipe(life.ArmSpec.from_document(doc))


@pytest.mark.parametrize('section,key,value',[('windows','train_k',24),('model','arch',mil.ARCH),
    ('updates','epochs',19),('readout','selection','best'),('sampler','batch',4)])
def test_recipe_rejects(section,key,value):
    doc=ops.spec_for(ROOT,42,1).document;doc['config'][section][key]=value
    with pytest.raises(life.ContractError): small.validate_recipe(life.ArmSpec.from_document(doc))


@pytest.mark.parametrize('seed,fold',ops.FITS)
def test_executed_cpu_card_and_final_only(seed,fold):
    path=ROOT/'.local/hb105m-validation-v1'/f's{seed}-f{fold}'
    spec=ops.spec_for(ROOT,seed,fold)
    summary=ops.check_outputs(spec,path/'collected/outputs',smoke=True)
    assert summary['cpu_cuda_initialized'] is False
    assert summary['results']['optimizer_steps']==1 and summary['results']['validation_batches']==3
    note=ops.read(path/'cpu/verification.json')
    life.verify_evidence(spec,note,path/'cpu')
    assert note['verdict']=='HOLD' and note['smokes']['target'] is None
    assert not (path/'cpu/epochs').exists()
    assert ops.sha(path/'cpu/outputs/model_best.pt')==ops.sha(path/'collected/outputs/model_best.pt')


def test_no_cpu_training():
    with pytest.raises(life.ContractError,match='full CPU'):
        small.run(ops.spec_for(ROOT,42,1),ROOT,CORPUS,'unused',mode='full',device='cpu')


def test_cpu_cannot_issue_pass_or_scientific_export(tmp_path):
    spec=ops.spec_for(ROOT,42,1);p=ROOT/'.local/hb105m-validation-v1/s42-f1/cpu'
    with pytest.raises(life.ContractError): ops.combine(spec,p,p,tmp_path/'pass','B','Claude Code')
    with pytest.raises(life.ContractError): ops.finish(spec,p/'outputs',p/'verification.json',tmp_path/'a.tar',tmp_path/'r.json')


def test_collect_tampering_rejected(tmp_path):
    p=ROOT/'.local/hb105m-validation-v1/s42-f1'
    fake=tmp_path/'bad.tar';fake.write_bytes(b'bad')
    with pytest.raises(life.ContractError,match='archive differs'):
        ops.collect(ops.spec_for(ROOT,42,1),fake,p/'export.json',tmp_path/'out')


def test_readout_smoke_not_science():
    p=ROOT/'.local/hb105m-validation-v1/s42-f1/collected'
    with pytest.raises(life.ContractError,match='cannot use smoke'):
        ro.collected(ROOT,p,ops.spec_for(ROOT,42,1),[],{})


def test_rank_equal_seed_family_policy():
    rng=np.random.default_rng(1)
    raw={f:{k:rng.random((10,12)) for k in ('R42','R43','C4','DINO','M42','M43','S42','S43')} for f in range(2)}
    vv=ro.variants(raw,(0,1)); rank=ro.ea.rank
    smallp=rank(np.concatenate([.5*(rank(raw[f]['S42'])+rank(raw[f]['S43'])) for f in range(2)]))
    np.testing.assert_array_equal(vv['MIL_S'],smallp)
    m=ro.ea.members(raw,(0,1),mil_seeds=('M42','M43'))
    np.testing.assert_array_equal(vv['Q7_add_S'],rank((m['R50']+m['C4']+m['MIL']+smallp)/4))


def test_missing_production_fits_fail(tmp_path):
    with pytest.raises(FileNotFoundError): ro.analyze(ROOT,tmp_path)
