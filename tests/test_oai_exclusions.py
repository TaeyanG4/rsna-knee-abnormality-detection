"""OAI holdout v1.1.0: private registration retains source and participant checks."""
import copy
from types import SimpleNamespace

import pytest

from training.arms import oai_exclusions as holdout


def policy(tmp_path, contents='study_key\nOAI_9000001_R_V00\n'):
    path = tmp_path / 'panel.csv'
    path.write_text(contents, encoding='utf-8')
    return path, holdout.bind(path)


def test_default_excludes_both_knees_and_every_visit(tmp_path, monkeypatch):
    path, bound = policy(tmp_path)
    monkeypatch.setattr(holdout, 'DEFAULT_LIST', path)
    assert holdout.bind() == bound
    ids = ['OAI_9000001_R_V00', 'OAI_9000001_L_V00', 'OAI_9000001_L_V12',
           'OAI_9000002_R_V00', 'OAI_9000002_L_V24']
    assert holdout.filter_ids(ids, bound) == ids[3:]
    assert bound['participant_count'] == 1


def test_explicit_participant_list_and_duplicates(tmp_path):
    path, bound = policy(tmp_path, 'participant_id\n9000001\n9000001\n9000002\n')
    assert bound['participants'] == ['9000001', '9000002']
    text = tmp_path / 'exclude.txt'
    text.write_text('9000001\nOAI_9000002_L_V12\n', encoding='utf-8')
    assert holdout.bind(text)['participants'] == bound['participants']


@pytest.mark.parametrize('contents', [
    'study_key\n', 'participant_id\n', 'wrong\n9000001\n',
    'study_key,label\nOAI_9000001_R_V00,1\n', 'study_key\ninvalid\n',
    'study_key\nOAI_9000001_R_V00,extra\n', 'participant_id\n90000.1\n'])
def test_empty_malformed_or_non_id_only_list_rejected(tmp_path, contents):
    with pytest.raises(ValueError):
        policy(tmp_path, contents)


def test_missing_default_is_not_empty_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, 'DEFAULT_LIST', tmp_path/'missing.csv')
    with pytest.raises(FileNotFoundError):
        holdout.bind()


def test_same_participants_changed_file_requires_reregistration(tmp_path):
    path, bound = policy(tmp_path)
    path.write_text('study_key\nOAI_9000001_L_V12\n', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        holdout.verify(bound)


def test_registered_population_must_exclude_subject_before_training_or_bn(tmp_path):
    path, bound = policy(tmp_path)
    p = dict(oai_exclusion=bound, rsna_train_ids=['R'],
             oai_train_ids=['OAI_9000002_R_V00'], train_ids=['R','OAI_9000002_R_V00'])
    assert holdout.validate_training(p, path) == bound
    leaked = copy.deepcopy(p)
    leaked['oai_train_ids'] = ['OAI_9000001_L_V24']
    leaked['train_ids'] = ['R', *leaked['oai_train_ids']]
    with pytest.raises(ValueError, match='excluded participant'):
        holdout.validate_training(leaked)
    with pytest.raises(ValueError, match='requires a registered'):
        holdout.validate_training({k:v for k,v in p.items() if k != 'oai_exclusion'})
    reordered = copy.deepcopy(p)
    reordered['train_ids'].reverse()
    with pytest.raises(ValueError, match='population/order differs'):
        holdout.validate_training(reordered)


def test_registration_accepts_default_or_explicit_list_and_rejects_missing_before_writes(tmp_path, monkeypatch):
    from tools import exp190_coatnet_oai_f0 as arm
    args = arm.parser().parse_args(['register'])
    assert args.oai_exclusion_list is None
    explicit = tmp_path/'not-created.csv'
    args = arm.parser().parse_args(['register', '--oai-exclusion-list', str(explicit)])
    assert args.oai_exclusion_list == str(explicit)
    def forbidden(*args, **kwargs):
        raise AssertionError('labels must not be created before exclusion validation')
    monkeypatch.setattr(arm, 'build_label_artifacts', forbidden)
    with pytest.raises(FileNotFoundError):
        arm.register(args)


def test_explicit_override_cannot_change_bound_participants(tmp_path):
    path, bound = policy(tmp_path)
    alternate = tmp_path/'different.txt'
    alternate.write_text('9000002\n', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        holdout.verify(bound, alternate)


def test_new_trainer_requires_registered_holdout_before_cuda(monkeypatch):
    from tools import exp190_coatnet_oai_f0 as arm
    import exp155_own0949_oai_v11 as fit
    guard = SimpleNamespace(load_registration=lambda *a: {}, require_validation=lambda *a: None)
    args = SimpleNamespace(protocol='unused', corpus='unused', init_weights='unused', oai_corpus='unused')
    with pytest.raises(ValueError, match='requires a registered'):
        fit.train(args, guard=guard)


def test_real_registered_holdout_covers_all_cache_visits_and_bn():
    from tools import exp190_coatnet_oai_f0 as arm
    from oai_registration_fixture import registration
    p = registration()
    excluded = holdout.verify(p['oai_exclusion'])
    cache_ids = arm.cache_binding(arm.CACHE)['ids']
    removed = [uid for uid in cache_ids if holdout.participant(uid) in excluded]
    assert removed and not set(removed) & set(p['train_ids'])
    assert not any(holdout.participant(uid) in excluded for uid in p['oai_train_ids'])
    assert holdout.validate_training(p, holdout.DEFAULT_LIST) == p['oai_exclusion']
    args = SimpleNamespace(corpus=arm.bridge.CORPUS, oai_corpus=arm.CACHE,
                           oai_exclusion_list=holdout.DEFAULT_LIST)
    for augment in (True, False):
        dataset = arm.dataset_factory(args, p, 42, augment=augment)
        assert dataset.ids == p['train_ids']
        assert not set(removed) & set(dataset.ids)
    assert len(p['train_ids']) < 5630
