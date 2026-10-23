"""Small CPU regression suite for the additive coverage lifecycle adapter."""
import copy
import csv

import numpy as np
import pytest

from tools import coverage43_readout_v1 as adapter


def prediction(path, ids=('u1', 'u2'), value=.5):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['study_id', *adapter.multi.TARGETS])
        writer.writerows([[uid, *([value]*12)] for uid in ids])


def test_conversion_preserves_prediction_bytes(tmp_path):
    source, dest = tmp_path/'in.csv', tmp_path/'out.csv'
    prediction(source)
    adapter.validate_csv(source, ['u1', 'u2'])
    binding = adapter.convert(source, dest, ['u1', 'u2'])
    assert dest.read_bytes() == b'StudyInstanceUID,'+source.read_bytes()[len(b'study_id,'):]
    ids, values, _ = adapter.multi.load_oof(binding)
    assert ids == ['u1', 'u2'] and np.all(values == .5)
    with pytest.raises(FileExistsError):
        adapter.convert(source, dest, ['u1', 'u2'])


@pytest.mark.parametrize('ids,value', [(('u2', 'u1'), .5), (('u1', 'u1'), .5),
                                       (('u1', 'training-uid'), .5),
                                       (('u1', 'u2'), float('nan')),
                                       (('u1', 'u2'), float('inf')),
                                       (('u1', 'u2'), 1.1)])
def test_prediction_rejection(tmp_path, ids, value):
    source = tmp_path/'in.csv'
    prediction(source, ids, value)
    with pytest.raises(ValueError):
        adapter.validate_csv(source, ['u1', 'u2'])


def test_substitution_drift(tmp_path):
    source = tmp_path/'source'
    source.write_bytes(b'original')
    bindings = {}
    digest = adapter.pin(tmp_path, 'source', bindings)
    source.write_bytes(b'substituted')
    with pytest.raises(ValueError, match='hash differs'):
        adapter.pin(tmp_path, 'source', bindings, digest)


def fit():
    config = dict(variant='coverage', train_sha256='a'*64, val_sha256='b'*64)
    summary = dict(status='COMPLETE_TRAINING_AND_READOUT', mode='full', updates=3456, skips=0)
    provenance = dict(scientific=copy.deepcopy(config), population_sha256='c'*64,
        spec_sha256='d'*64, verification_sha256='e'*64, readout='held-out-fold0',
        identity=dict(mode='full', spec_sha256='d'*64, train_sha256='a'*64, readout_sha256='b'*64))
    return summary, provenance, config


@pytest.mark.parametrize('mutation', ['science', 'train', 'val', 'smoke', 'updates'])
def test_fit_pair_rejects_drift(mutation):
    summary, provenance, config = fit()
    adapter.validate_fit(summary, provenance, config, 'c'*64)
    if mutation == 'science': provenance['scientific']['variant'] = 'control'
    if mutation == 'train': provenance['identity']['train_sha256'] = '0'*64
    if mutation == 'val': provenance['identity']['readout_sha256'] = '0'*64
    if mutation == 'smoke': summary['mode'] = 'smoke'
    if mutation == 'updates': summary['updates'] = 1
    with pytest.raises(ValueError):
        adapter.validate_fit(summary, provenance, config, 'c'*64)


def test_view_wiring_and_fixed_method():
    for view in adapter.VIEWS:
        pair = dict(arm=[dict(path='arm-'+view+'.csv')], control=[dict(path='control-'+view+'.csv')])
        screen = adapter.view_spec(view, pair)
        assert screen['seeds'] == {'43': pair}
        assert screen['label_change'] is False and screen['family'] == 'coatnet'
        assert 'partners' not in screen and 'fold_rank' not in screen
    assert (adapter.multi.DRAWS, adapter.multi.SEED) == (300, 20261003)
    assert adapter.multi.WEAK == (3, 5, 6, 8)


def test_cuda_tripwire():
    import torch
    adapter.cpu_only()
    assert torch.get_num_threads() == 1 and not torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match='CUDA forbidden'):
        torch.cuda.init()
    with pytest.raises(RuntimeError, match='CUDA forbidden'):
        torch.cuda._lazy_init()


def test_existing_result_is_never_overwritten(tmp_path):
    out = tmp_path/adapter.BASE/'screen-full'
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match='immutable'):
        adapter.run(tmp_path)
