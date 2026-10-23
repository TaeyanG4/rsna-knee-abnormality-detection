"""CoAtNet paired historical readout v1.0.0.

Changelog: explicit records replace patched module globals; fresh immutable
outputs retain the registered equations alongside all seven holistic lenses.
Record authentication is performed by the caller's family-specific verifier.
"""
from __future__ import annotations

import json
from pathlib import Path

from training.arms import collected_screen as cs
from training.arms import coatnet_history as history

ROOT = Path(__file__).resolve().parents[2]
VERSION = '1.0.0'
WEAK = ('Lateral Meniscus', 'PF OA', 'Lateral OA', 'Synovitis')
SOURCES = ('training/arms/coatnet_screen.py', 'training/arms/coatnet_history.py',
           'tools/exp181_coatnet_f0_bridge.py', 'tools/exp179_screen.py',
           'tools/coatnet_config.py', 'tools/coatnet_history_cli.py',
           'tools/exp188_coatnet_coverage_f0.py', 'tools/exp192_coatnet_crop_f0.py')


def write_new(out, value, root):
    out = Path(out).resolve()
    cs.portable_path(str(out), root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def paired_bindings(ctrl, cand, candidate, root):
    if cand['p']['arm'] != candidate or ctrl['p']['arm'] != 'CONTROL':
        raise ValueError('paired arm identity differs')
    if ctrl['p']['bridge_contract_sha256'] != cand['p']['bridge_contract_sha256']:
        raise ValueError('paired bridge contracts differ')
    if ctrl['p']['val_ids'] != cand['p']['val_ids']:
        raise ValueError('paired validation roster differs')
    for key in ('device_name', 'torch'):
        if ctrl['receipt'][key] != cand['receipt'][key]:
            raise ValueError('paired runtime differs: ' + key)
    bindings, oof = {}, {}
    for name in (*SOURCES, *cs.SOURCES, *cs.EVALUATION):
        cs.pin(root, name, bindings)
    for name, rec in (('CONTROL', ctrl), (candidate, cand)):
        for file in ('summary.json', 'provenance.json', 'model_best.pt', history.INVENTORY,
                     history.OOF_RECEIPT, history.OOF_PRIMARY, history.OOF_PLAIN):
            cs.pin(root, Path(rec['run']) / file, bindings)
        oof[name] = {f: history.sha256(Path(rec['run']) / f)
                     for f in (history.OOF_PRIMARY, history.OOF_PLAIN)}
    return bindings, oof


def collect_records(ctrl, cand, *, candidate, out, verification_note, root=ROOT):
    bindings, oof = paired_bindings(ctrl, cand, candidate, root)
    result = dict(schema='coatnet-paired-collection/v1', version=VERSION, status='COLLECTED',
                  candidate=candidate, control_protocol_sha256=ctrl['protocol_sha256'],
                  candidate_protocol_sha256=cand['protocol_sha256'], oof=oof,
                  input_sha256=bindings, verification_note=verification_note,
                  launch_authorized=False)
    cs.recheck(root, bindings)
    write_new(out, result, root)
    return result


def score_records(ctrl, cand, *, root=ROOT):
    from tools import exp181_coatnet_f0_bridge as bridge
    from tools import exp115_vlm_eval as evaluator
    from tools import exp179_screen as helper
    ids, refs, draws, context = helper.load_context(root)
    if set(ids) != set(ctrl['p']['val_ids']) or len(ids) != 888:
        raise ValueError('screen cohort differs')
    values = {name: {key: bridge.load_oof(Path(rec['run']) / file, ids)
                     for key, file in (('primary', bridge.OOF_PRIMARY), ('plain', bridge.OOF_PLAIN))}
              for name, rec in (('candidate', cand), ('control', ctrl))}
    paired = bridge.score(values['candidate'], values['control'], refs, draws, evaluator)
    primary = paired['primary_k94_mirror_tta']
    if any(primary['references'][r]['per_target_delta'][t] is None
           for r in ('jev', 'consensus') for t in bridge.TARGETS):
        raise ValueError('undefined primary target delta')
    delta = {t: .5 * sum(primary['references'][r]['per_target_delta'][t] for r in ('jev', 'consensus'))
             for t in bridge.TARGETS}
    weak_mean = sum(delta[t] for t in WEAK) / len(WEAK)
    macro = primary['primary_mean_delta']
    dread = primary['references']['dread']['delta']
    checks = dict(macro12=macro >= -.001,
                  other_targets=all(delta[t] >= -.010 for t in bridge.TARGETS if t not in WEAK),
                  dread=dread is not None and dread >= -.003)
    return dict(studies=len(ids), seed=42, bootstrap='300 component draws, seed 20261003',
        weak_targets=list(WEAK), weak_target_mean_delta=weak_mean, per_target_mean_delta=delta,
        historical_weak_gate='HOLD' if not all(checks.values()) or weak_mean <= 0 else 'REVIEW',
        harm_guard=dict(checks=checks, macro12_delta=macro, dread_macro_delta=dread,
                        other_target_floor=-.010, macro12_floor=-.001, dread_floor=-.003),
        synovitis_image_read='REQUIRED: OAI MOAKS; Gold58 report-only',
        registered_exp181_gate=paired['registered_gate'], ruler_advisory=paired['ruler_advisory'],
        primary_k94_mirror_tta=primary, diagnostic_k94_plain=paired['diagnostic_k94_plain'], context=context)


def screen_records(ctrl, cand, *, candidate, collection, out, verification_note, readout=None, root=ROOT,
                   score_fn=None):
    if Path(out).exists():
        raise FileExistsError('fresh screen output required')
    bindings, oof = paired_bindings(ctrl, cand, candidate, root)
    receipt = cs.read(collection)
    if (receipt.get('schema') != 'coatnet-paired-collection/v1'
            or receipt.get('status') != 'COLLECTED' or receipt.get('candidate') != candidate
            or receipt.get('control_protocol_sha256') != ctrl['protocol_sha256']
            or receipt.get('candidate_protocol_sha256') != cand['protocol_sha256']
            or receipt.get('oof') != oof or receipt.get('input_sha256') != bindings):
        raise ValueError('collection no longer matches verified pair')
    cs.pin(root, collection, bindings)
    optional = {} if readout is None else cs.read(readout)
    if not isinstance(optional, dict) or set(optional) - {'partners', 'anchor', 'image_reads'}:
        raise ValueError('readout may only add partner/anchor/image evidence')
    if readout is not None:
        cs.pin(root, readout, bindings)
    spec = dict(name=candidate+'-vs-CONTROL', family='coatnet', label_change=candidate == 'OAI',
                seeds={'42': {'arm': str(Path(cand['run']) / history.OOF_PRIMARY),
                              'control': str(Path(ctrl['run']) / history.OOF_PRIMARY)}}, **optional)
    scores = (score_fn or score_records)(ctrl, cand, root=root)
    if set(scores) & {'schema', 'version', 'status', 'candidate', 'control', 'fold',
                      'holistic', 'inputs', 'input_sha256', 'verification_note', 'launch_authorized'}:
        raise ValueError('score callback may not override collection identity or authority')
    result = dict(schema='coatnet-paired-screen/v1', version=VERSION, status='B_DECISION_PENDING',
                  candidate=candidate, control='EXP181 CONTROL', fold=0,
                  **scores, holistic=cs.readout(spec, root=root),
                  inputs=oof, input_sha256=bindings, verification_note=verification_note,
                  launch_authorized=False)
    cs.recheck(root, bindings)
    write_new(out, result, root)
    return result
