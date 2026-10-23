"""vNext-R7-PL 1.1.1: one pinned H-B-146 readout adapter for all consumers.

Changelog 1.1.1: consume the producer envelope and replay its imported gate and
selection schema. Never turn a legacy four-field decision into a scientific GO.
This checks provenance/decision consistency, not scientific reproducibility.
"""
from tools import hb146_readout_v111 as source

VERSION = '1.1.1'
SOURCES = ['tools/vnext_r7_pl_go_v111.py', 'tools/hb146_readout_v111.py',
           'tools/hb146_pl_labels_v111.py', 'tools/multi_lens_readout_l4_v2.py',
           'tools/multi_lens_readout.py', 'tools/exp156_k_sweep.py',
           'tools/exp125_radr50_k16_local.py', 'tools/exp118_radr50_local_s43.py']


def validate(readout_pin, arm):
    """Return a source/evidence seal; do not load any label or prediction rows."""
    from tools import vnext_r7_pl_v1 as p
    try:
        doc = p.read(p.pin(readout_pin))
        p.need(doc['version'] == source.pl.VERSION and doc['engineering_only'] is False
               and doc['status'] == source.pl.STATUS, 'readout version/exposure/engineering')
        gates = {}
        for side in ('global', 'fill'):
            result = doc['arms'][side]
            gate = result['registered_go']
            images = result['lenses']['L7']['entries']
            macro = [gate['report_macro_seed_mean_deltas'][r] for r in source.lens.RULERS]
            weak = [gate['report_weak4_seed_mean_deltas'][r] for r in source.lens.RULERS]
            p.need(macro == [result['report_seed_mean_macros'][r]['value'] for r in source.lens.RULERS],
                   'report gate evidence')
            gates[side] = source.gate(macro, weak, images['gold58']['value'],
                                      images['oai_moaks']['value'], doc['admission']['eligible'])
            p.need(gate == gates[side], 'arm gate differs from H-B-146 evidence')
        decision = source.choose_arm(gates)
        p.need(doc['registered_go'] == decision and decision['GO'] is True
               and decision['selected_arm'] == arm, 'selected arm/NO_GO/decision differs')

        # Every non-engineering run emits 18 bound prediction receipts. Bind all
        # their evidence to the source seal, and reject mixed screen identities.
        bindings = doc['prediction_bindings']
        expected = {f'{name}-{side}-{seed}' for name in ('val', 'gold', 'oai')
                    for side in ('control', 'global', 'fill') for seed in (42, 43)}
        p.need(set(bindings) == expected, 'prediction evidence roster')
        def hashed(value):
            p.need(isinstance(value, str) and len(value) == 64
                   and all(c in '0123456789abcdef' for c in value), 'evidence hash')
        identities, checkpoints = [], {}
        for key, receipt in bindings.items():
            _, side, seed = key.split('-')
            identity = receipt['identity']
            p.need(receipt['engineering_only'] is False and receipt['k'] == 36 and receipt['tta'] is True
                   and identity['arm'] == side and identity['seed'] == int(seed)
                   and identity['epochs'] == 20, 'prediction identity')
            for value in (receipt['csv_sha256'], receipt['checkpoint_sha256'],
                          identity['prepare_sha256'], identity['registration_sha256']):
                hashed(value)
            identities.append((identity['prepare_sha256'], identity['registration_sha256'], identity['runtime']))
            fit = (side, seed)
            p.need(checkpoints.setdefault(fit, receipt['checkpoint_sha256']) == receipt['checkpoint_sha256'],
                   'prediction checkpoint identity')
        p.need(all(i == identities[0] for i in identities), 'paired screen identity')
        p.need(bool(doc['ruler_input_hashes']) and bool(doc['teacher_f0_binding']['path']), 'readout evidence absent')
        for value in [*doc['ruler_input_hashes'].values(), doc['teacher_f0_binding']['sha256']]:
            hashed(value)
        evidence = {k: doc[k] for k in ('arms', 'prediction_bindings', 'ruler_input_hashes',
                                       'admission', 'teacher_f0_binding')}
        return dict(adapter_version=VERSION, source_version=doc['version'],
                    source_sha256=readout_pin['sha256'], evidence_sha256=p.digest(evidence),
                    registered_go=decision)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError('scientific GO: ' + str(exc)) from exc


def admission(record, arm, spec_sha256):
    """Revalidate an admission at export/binding, including its evidence seal."""
    from tools import vnext_r7_pl_v1 as p
    p.need(record and record.get('status') == 'ADMITTED'
           and record.get('spec_sha256') == spec_sha256, 'production export requires matching admission')
    result = validate(record['go'], arm)
    p.need(record.get('decision') == result, 'scientific GO admission evidence changed')
    return result
