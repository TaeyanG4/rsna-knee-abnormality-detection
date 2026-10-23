"""HB123 OAI557 header-only side registration v1.0.0.

Changelog 2026-10-13: bind selected cache groups to raw DICOM headers; retain
unknowns, audit description/ID agreement without using either to assign side.
No clinical outcomes, report text, pixel decoding or model inference here.
"""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np
from training.arms import laterality_v1 as lat
from tools.hb123_laterality_v1 import header_dict
from tools import hb123_build_v2 as build

ROOT = build.ROOT
WORK = ROOT / '.local/scratch/sol-5/hb123/screen-v1'
MAP = WORK / 'oai557-side-map-v1.json'
CACHE = ROOT / 'storage/cache/derived/EXT_OAI_RAPTOR64_336_C140_v2'
GEN = ROOT / 'storage/cache/external/OAI_HB89_V2/storage/cache/external/OAI/gen'
AUDIT = ROOT / 'artifacts/experiments/HB88_OAI_HEADER_AUDIT_V2/run-v1/archives.jsonl'
PANEL = ROOT / 'artifacts/experiments/OAI_MOAKS_PANEL_V1/panel.csv'
need = build.mil.need
read = build.mil.read
VERSION = '1.0.0'


def pins():
    paths = [PANEL, CACHE/'manifest.json', CACHE/'picks.jsonl', GEN/'index.csv',
             GEN/'registration.json', GEN/'receipt.json', AUDIT,
             ROOT/'tools/hb123_oai_side_v1.py', ROOT/'training/arms/laterality_v1.py',
             ROOT/'tools/hb123_laterality_v1.py']
    return {p.relative_to(ROOT).as_posix(): lat.sha(p) for p in paths}


def roster():
    with PANEL.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == ['study_key'], 'panel ID-only schema differs')
        ids = [r['study_key'] for r in reader]
    need(len(ids) == 557 and len(set(ids)) == 557 and ids == sorted(ids), 'OAI557 roster differs')
    return ids


def selected_groups(ids):
    manifest, receipt, reg = read(CACHE/'manifest.json'), read(GEN/'receipt.json'), read(GEN/'registration.json')
    need(lat.sha(GEN/'index.csv') == manifest['gen_index_sha256'] == receipt['index_sha256']
         and lat.sha(GEN/'registration.json') == receipt['registration_sha256']
         and lat.sha(CACHE/'picks.jsonl') == manifest['picks_sha256']
         and lat.sha(AUDIT) == reg['audit']['output_sha256']['archives.jsonl'], 'cache/audit ancestry differs')
    with (GEN/'index.csv').open(encoding='utf-8-sig', newline='') as stream:
        index = {r['group_key']: r for r in csv.DictReader(stream)}
    wanted, by_uid = {}, {}
    with (CACHE/'picks.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['uid'] not in ids:
                continue
            keys = []
            for slot, digest in zip(row['slots'], row['gen_sha256'], strict=True):
                if not slot.get('group_key'):
                    continue
                key = slot['group_key']
                need(key in index and index[key]['npz_sha256'] == digest
                     and index[key]['study_key'] == row['uid']
                     and index[key]['plane'] == slot['slot'][0], 'selected group identity differs')
                wanted[key] = index[key]
                keys.append(key)
            # Sag0 copied from Sag1 is one independent series, never two votes.
            by_uid[row['uid']] = sorted(set(keys))
    need(set(by_uid) == set(ids) and all(by_uid.values()), 'missing cache selection')
    groups = {}
    with AUDIT.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            for group in row['groups']:
                key = row['series_key'] + '::' + group['group_id']
                if key not in wanted:
                    continue
                need(key not in groups and row['status'] == 'READ_OK', 'duplicate/unreadable audited group')
                archive = Path(row['relocated_archive'])
                need(archive.resolve().is_relative_to(ROOT.resolve()), 'archive outside project')
                need(archive.resolve() == Path(wanted[key]['archive']).resolve(), 'archive link differs')
                groups[key] = dict(archive=archive.relative_to(ROOT).as_posix(),
                    size=row['raw_size'], mtime_ns=row['raw_mtime_ns'], group=group)
    need(set(groups) == set(wanted), 'selected group lacks audited frames')
    return by_uid, groups


def sampled_headers(record):
    """Read only first/middle/last selected DICOM headers, not audit side joins."""
    import pydicom
    path, group = ROOT/record['archive'], record['group']
    stat = path.stat()
    need((stat.st_size, stat.st_mtime_ns) == (record['size'], record['mtime_ns']), 'raw archive changed')
    refs = sorted(group['frame_refs'])
    need(bool(refs), 'empty selected series')
    selected = [refs[i] for i in sorted({0, len(refs)//2, len(refs)-1})]
    # The retained OAI panel uses conventional single-frame MR objects.
    need(all(r[1] == 1 for r in selected), 'enhanced DICOM requires new registered reader')
    expected = {r[0]: r[2] for r in selected}
    headers, descriptions = {}, set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = member.name.removeprefix('./')
            if name not in expected:
                continue
            with archive.extractfile(member) as stream:
                ds = pydicom.dcmread(io.BytesIO(stream.read()), stop_before_pixels=True,
                    specific_tags=lat.TAGS+['SeriesDescription', 'SOPInstanceUID', 'SeriesInstanceUID'])
            need(str(ds.SOPInstanceUID) == expected[name]
                 and str(ds.SeriesInstanceUID) == group['series_instance_uid'], 'DICOM selected identity differs')
            header = header_dict(ds)
            need(header.get('Rows') == group['rows'] and header.get('Columns') == group['columns']
                 and np.allclose(header['ImageOrientationPatient'], group['iop'], atol=1e-5, rtol=0)
                 and np.allclose(header['PixelSpacing'], group['pixel_spacing'], atol=1e-5, rtol=0),
                 'selected geometry differs from rendered series')
            need(name not in headers, 'duplicate selected DICOM member')
            headers[name] = header
            words = set(re.findall('[A-Z]+', str(ds.get('SeriesDescription', '')).upper()))
            descriptions.update(s for word, s in [('LEFT', 'L'), ('RIGHT', 'R')] if word in words)
            if len(headers) == len(expected):
                break
    need(set(headers) == set(expected), 'selected raw DICOM missing')
    return dict(summary=lat.summarize_series([headers[r[0]] for r in selected], group['plane']),
                description_sides=sorted(descriptions), selected_refs=selected)


def resolve(uid, evidence):
    result = {lat.UID: uid, **lat.resolve_study([e['summary'] for e in evidence])}
    descriptions = set().union(*(set(e['description_sides']) for e in evidence))
    # QA-only comparisons: neither identifier nor description supplies a vote.
    declared = uid.split('_')[2]
    comparable = result['side'] != 'U'
    result.update(id_comparable=comparable, id_agree=comparable and declared == result['side'],
                  id_disagree=comparable and declared != result['side'],
                  description_comparable=comparable and len(descriptions) == 1,
                  description_agree=comparable and descriptions == {result['side']},
                  description_disagree=comparable and bool(descriptions - {result['side']}))
    return result


def build_map():
    if MAP.exists():
        return validate()
    ids, inputs = roster(), pins()
    by_uid, groups = selected_groups(ids)
    evidence = {}
    for i, (key, record) in enumerate(sorted(groups.items())):
        cache = WORK/'header-evidence'/f'{build.life.content_sha256(key)}.json'
        binding = build.life.content_sha256(dict(record=record, sources=inputs))
        if cache.exists():
            doc = read(cache)
            need(doc['binding'] == binding, 'cached header evidence drift')
            stat = (ROOT/record['archive']).stat()
            need((stat.st_size, stat.st_mtime_ns) == (record['size'], record['mtime_ns']), 'raw archive changed')
        else:
            doc = dict(binding=binding, evidence=sampled_headers(record))
            build.same_or_new(cache, doc)
        evidence[key] = doc['evidence']
        if (i+1) % 100 == 0:
            print(json.dumps(dict(header_groups=i+1, total=len(groups))), flush=True)
    studies = [resolve(uid, [evidence[k] for k in by_uid[uid]]) for uid in ids]
    counts = Counter(studies=len(ids), unique_series=len(groups), headers=sum(e['summary']['headers'] for e in evidence.values()))
    for row in studies:
        counts['side_'+row['side']] += 1
        counts['reason_'+row['reason']] += 1
        for k in ('id_comparable','id_agree','id_disagree','description_comparable','description_agree',
                  'description_disagree','tag_geometry_comparable','tag_geometry_agree','tag_geometry_disagree',
                  'direct_tag_conflict','geometry_conflict'):
            counts[k] += row[k]
    need(not counts['id_disagree'] and not counts['description_disagree'], 'OAI side QA disagreement; investigate before registration')
    need(pins() == inputs, 'inputs changed while reading headers')
    document = dict(schema='hb123-side-map/v1', version=VERSION, policy=lat.POLICY,
        scope='OAI557 selected cache series only', inputs=inputs, studies=studies, counts=dict(counts),
        series_evidence=evidence, selected_groups=by_uid,
        limitations=['Sampled first/middle/last selected frames; geometry is a header proxy',
                    'Descriptions and panel identifiers are QA only; unknowns retained unchanged',
                    'No independent ancestor exclusion or clinical validation claim'])
    build.same_or_new(MAP, document)
    return validate()


def validate():
    doc = read(MAP)
    need(doc['version'] == VERSION and doc['inputs'] == pins(), 'OAI side-map inputs changed')
    ids = roster()
    sides = lat.SideMap(MAP)
    need(list(sides.sides) == ids and doc['counts']['studies'] == len(ids), 'OAI map roster differs')
    for row in doc['studies']:
        need(row == resolve(row[lat.UID], [doc['series_evidence'][k] for k in doc['selected_groups'][row[lat.UID]]]),
             'OAI map resolution differs')
    need(not doc['counts']['id_disagree'] and not doc['counts']['description_disagree'], 'OAI agreement gate failed')
    return dict(status='OAI557_SIDE_MAP_VALIDATED', path=MAP.relative_to(ROOT).as_posix(),
                sha256=sides.sha256, counts=doc['counts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'validate'])
    args = parser.parse_args()
    print(json.dumps(build_map() if args.command == 'build' else validate(), sort_keys=True))
