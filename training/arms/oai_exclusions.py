"""OAI participant holdout v1.0.0: bind ID-only lists before any new fit.

The default list is the MOAKS panel. A knee/visit entry excludes every study
of that participant. Missing, empty, malformed or changed lists fail closed.
"""
from pathlib import Path
import csv
import hashlib
import io
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIST = ROOT / 'artifacts/experiments/OAI_MOAKS_PANEL_V1/panel.csv'


def participant(value):
    value = str(value).strip()
    match = re.fullmatch(r'OAI_(\d{7})_[LR]_V\d{2}', value)
    if match:
        return match[1]
    if re.fullmatch(r'\d{7}', value):
        return value
    raise ValueError('invalid OAI participant/study identifier in exclusion policy')


def bind(path=None):
    path = Path(path or DEFAULT_LIST).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = path.read_bytes()
    text = raw.decode('utf-8-sig')
    if path.suffix.lower() == '.csv':
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames not in (['study_key'], ['participant_id']):
            raise ValueError('OAI exclusion CSV must contain only study_key or participant_id')
        values = []
        for row in reader:
            if set(row) != set(reader.fieldnames) or None in row.values():
                raise ValueError('malformed OAI exclusion CSV')
            values.append(row[reader.fieldnames[0]])
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    participants = sorted({participant(value) for value in values})
    if not participants:
        raise ValueError('OAI exclusion list must not be empty')
    return dict(schema='oai-participant-exclusion-v1', path=relative,
                sha256=hashlib.sha256(raw).hexdigest(), participants=participants,
                scope='all-knees-all-visits', participant_count=len(participants))


def verify(policy, path=None):
    if not isinstance(policy, dict):
        raise ValueError('new OAI supervision requires a registered participant exclusion list')
    if bind(path or ROOT / policy['path']) != policy:
        raise ValueError('OAI exclusion list changed; create a new registration')
    return set(policy['participants'])


def filter_ids(ids, policy):
    excluded = verify(policy)
    return [uid for uid in ids if participant(uid) not in excluded]


def validate_training(p, path=None):
    excluded = verify(p.get('oai_exclusion'), path)
    if any(participant(uid) in excluded for uid in p['oai_train_ids']):
        raise ValueError('OAI training/BN population contains an excluded participant')
    if p['train_ids'] != p['rsna_train_ids'] + p['oai_train_ids']:
        raise ValueError('OAI exclusion registration population/order differs')
    return p['oai_exclusion']
