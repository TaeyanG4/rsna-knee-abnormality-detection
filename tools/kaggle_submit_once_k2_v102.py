"""K2 v1.0.2 offline admission adapter. No sender, CLI, lock or ledger.

Canonical alone owns transport. This fixed exact-family API authenticates local
validator bytes before imports, including cached imports, and returns every
consumed hash for canonical's two qualifications and late recheck.
"""
from __future__ import annotations
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
VERSION='1.0.2'
CANDIDATE='K2-RCOF0-V1'
KERNEL='taeyangg4/knee-l1-rcof0-v1'
STAGE='artifacts/experiments/L1_RCO_K2/v1.0.2-rcof0'
PARENT_SHA='62df1fb0bc062c3a7f9a763e37c9a06e4261656ecab720c9c9c1c8d2387c3a86'
EXECUTABLE_SOURCES=(
    'tools/l1_rco_k2_v102.py','tools/l1_rco_k2_worker_v101.py','tools/strict_variant_gate_k2_v102.py',
    'tools/l1_rco_k2_admission_v102.py','tools/kaggle_submit_once_k2_v102.py',
    '.local/run-scripts/b-preflight-k2-v102.py','tools/test_l1_rco_k2_v102.py',
    'tools/strict_variant_gate.py','tools/strict_variant_gate_standalone.py','tools/strict_variant_gate_hb177.py',
    'tools/hb177_release_admission_v1.py','tools/strict_variant_gate_r224_v145.py','tools/strict_variant_gate_hb151.py',
    'tools/native_codex_dispatch_record_v1.py','tools/s6plus.py','tools/kaggle_submit_once_v139.py','.local/run-scripts/b-preflight-v107.py',
    '.local/run-scripts/kaggle-visible-wait-v3.sh','tools/l1_rco_k1_v2.py','tools/l1_rco_stage0_v1.py',
    'tools/l1_rco_stage0_v120.py','tools/l1_rco_runtime_v120.py','tools/exp189_s6_port.py')


def need(ok,message):
    if not ok:raise ValueError(message)


def validate_identity(sub_id,kernel):
    need(isinstance(sub_id,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',sub_id),'valid explicit candidate ID required')
    need(sub_id==CANDIDATE and kernel==KERNEL,'exact reserved K2 ID/slug pair required')


def _digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _read(path):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'duplicate key');out[k]=v
        return out
    return json.loads(path.read_bytes(),object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite JSON'))


def _authenticate(root):
    """No local validator has been imported at this point."""
    root=Path(root).resolve();regpath=root/STAGE/'registration.json';sourcepath=regpath.parent/'source-manifest.json'
    reg=_read(regpath);source=_read(sourcepath)
    need(reg['schema']=='k2-registration/v1.0.2' and reg['source_manifest_sha256']==_digest(sourcepath),'BUILD source bootstrap binding')
    need(set(EXECUTABLE_SOURCES)<=set(source) and 'tools/kaggle_submit_once.py' not in source,'fixed adapter executable closure')
    pins={}
    for name in EXECUTABLE_SOURCES:
        p=root/name;need(_digest(p)==source[name],'validator source changed before import: '+name);pins[str(p)]=source[name]
    need(source['tools/kaggle_submit_once_v139.py']==PARENT_SHA,'immutable offline schema validator')
    return pins


def _qualify(sub_id,kernel,version,receipt,review,root):
    validate_identity(sub_id,kernel);root=Path(root).resolve();source_pins=_authenticate(root)
    # Imports may be cached, but authentication above and the final recheck are
    # performed on every invocation. Canonical independently pins this tuple.
    sys.path.insert(0,str(ROOT))
    base=importlib.import_module('tools.strict_variant_gate')
    gate=importlib.import_module('tools.strict_variant_gate_k2_v102')
    admission=importlib.import_module('tools.l1_rco_k2_admission_v102')
    need(set(EXECUTABLE_SOURCES)==admission.transport_source_names()-{'tools/kaggle_submit_once.py'},'adapter/final fixed closure disagreement')
    source=root/'tools/kaggle_submit_once_v139.py'
    spec=importlib.util.spec_from_file_location('_k2_offline_schema_v139',source)
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator);validator.ROOT=root
    inputs=base.Inputs(root);path=base.local_path(receipt,root);review_path=base.local_path(review,root)
    record=inputs.json(path);binding_path=path.parent/'submission-binding.json'
    binding=inputs.json(binding_path) if binding_path.exists() else record
    need(binding.get('phase')=='FINAL_READY' and binding.get('qualified_for_submission') is True,'mandatory FINAL_READY even non-FAST')
    gatepath=root/'.local/state/gates'/('knee-l1-rcof0-v1.json')
    observed=gate.verify_receipt(gatepath,inputs,'FINAL_READY')
    need(binding.get('gate_sha256')==inputs.digest(gatepath),'mandatory exact final gate binding')
    need(str(observed['kernel_version'])==str(version) and base.local_path(observed['final_review'],root)==review_path,'exact final review/remote version')
    bp=gate.checked_pin(observed['pin'],inputs);checked=admission.replay_final(observed,inputs,bp)
    need(base.local_path(checked['visible_receipt']['path'],root)==path and checked['visible_receipt']['sha256']==inputs.digest(path),'exact qualified visible receipt')
    def strict(k,binding_value,hashes):
        need(k==KERNEL and binding_value.get('gate_sha256')==inputs.digest(gatepath),'offline validator exact K2 gate')
        hashes.update({str(root/n):h for n,h in inputs.hashes().items()});return observed
    validator._strict_gate=strict
    result=validator.qualify(kernel,version,path,review_path)
    result['input_sha256'].update(source_pins)
    result['input_sha256'].update({str(root/n):h for n,h in inputs.hashes().items()})
    result.update(phase='FINAL_READY',qualified_for_submission=True,gate_version=gate.VERSION,
        provenance_contract='k2-build-final-transport/v1',adapter_version=VERSION,candidate_id=sub_id,
        transport_sources=checked['transport_sources'])
    inputs.recheck();validator._recheck(result)
    _authenticate(root)
    return result


def qualify(sub_id,kernel,version,receipt,review):
    return _qualify(sub_id,kernel,version,receipt,review,ROOT)
