import copy
import json
import pytest
from tools import hb155_cache_v2 as c
from tools import hb155_transfer_v1 as t


@pytest.mark.parametrize('engineering,sample_pass,expected',[(True,False,2),(True,True,0),(False,True,2)])
def test_engineering_cli_cannot_waive_failed_sample(monkeypatch,engineering,sample_pass,expected):
    import sys
    monkeypatch.setattr(sys,'argv',['cache','qa']+(['--engineering'] if engineering else []))
    monkeypatch.setattr(c,'qa',lambda *a:dict(qa_pass=False,sample_qa_pass=sample_pass))
    assert c.main()==expected


def test_registration_rejects_geometry_drift(tmp_path,monkeypatch):
    monkeypatch.setattr(c.c,'QA_WORK',tmp_path)
    c.register()
    monkeypatch.setattr(c.c,'GEOMETRY_POLICY',dict(c.c.GEOMETRY_POLICY,padding='changed'))
    with pytest.raises(ValueError,match='registration changed'): c.register()


@pytest.mark.parametrize('workers',[0,5,100])
def test_worker_cap_before_source_access(workers):
    with pytest.raises(ValueError,match='workers'): c.build(workers=workers)


@pytest.mark.parametrize('counts,padding,key',[
    ({'pixel_decode_error':1},0,'decode_failures'),
    ({'header_decode_error':1},0,'decode_failures'),
    ({'laterality_conflict':1},0,'laterality_conflicts'),
    ({'anchor_outside_field':1},0,'geometry_failures'),
    ({'padded_crop':65},0,'padded_slice_fraction'),
    ({},.25001,'padding_fraction'),
    ({'clamped_stack_target':641},0,'clamped_slice_fraction'),
    ({'unrecognized':1},0,'unrecognized_failure:unrecognized'),
])
def test_registered_qa_fail_closed(counts,padding,key):
    _,fail=c.limits(counts,100,6400,padding)
    assert key in fail


def test_missing_slices_and_unknown_side_have_explicit_denominators():
    metrics,fail=c.limits({'unknown_laterality':100,'missing_series':4},100,6080,0)
    assert metrics['unknown_laterality_fraction']==1 and not fail
    assert 'empty_slice_fraction' in c.limits({},100,6079,0)[1]


def chunk(out):
    for name,data in [('vol.bin',b'pixels'),('qa.json',b'[]'),('mask.bin',b'mask'),('overlay.png',b'png')]:
        (out/name).write_bytes(data)
    return dict(identity={'id':'a'},file='vol.bin',bytes=6,sha256=c.c.sha(out/'vol.bin'),
        qa_file='qa.json',qa_sha256=c.c.sha(out/'qa.json'),mask_file='mask.bin',mask_sha256=c.c.sha(out/'mask.bin'),
        overlays={'overlay.png':c.c.sha(out/'overlay.png')})


@pytest.mark.parametrize('filename',['vol.bin','qa.json','mask.bin','overlay.png'])
def test_resume_rejects_changed_committed_bytes(tmp_path,filename):
    r=chunk(tmp_path); c.verify_chunk(tmp_path,r,{'id':'a'})
    (tmp_path/filename).write_bytes(b'broken')
    with pytest.raises(ValueError): c.verify_chunk(tmp_path,r,{'id':'a'})


def test_resume_rejects_changed_identity(tmp_path):
    r=chunk(tmp_path)
    with pytest.raises(ValueError,match='identity'): c.verify_chunk(tmp_path,r,{'id':'b'})


def test_parallel_transfer_lists_disjoint_exact_and_draft_rejected(tmp_path):
    files={f'file{i}.bin':dict(bytes=i+1,sha256='sealed') for i in range(17)}
    manifest=tmp_path/'inventory.json'; manifest.write_text(json.dumps(dict(status='READY',files=files)))
    result=t.streams(manifest,tmp_path)
    listed=[name for i in range(4) for name in (tmp_path/f'transfer-stream-{i}.txt').read_text().splitlines()]
    assert len(listed)==len(set(listed))==17 and set(listed)==set(files)
    assert sum(result['bytes'])==sum(f['bytes'] for f in files.values())
    manifest.write_text(json.dumps(dict(status='DRAFT_NOT_READY',files=files)))
    with pytest.raises(ValueError,match='draft'): t.streams(manifest,tmp_path)


def test_transfer_exact_byte_count_even_when_digest_matches(tmp_path):
    p=tmp_path/'x'; p.write_bytes(b'abc')
    manifest=tmp_path/'m.json'
    manifest.write_text(json.dumps(dict(status='READY',files={'x':dict(bytes=4,sha256=c.c.sha(p))})))
    with pytest.raises(ValueError,match='hash'): t.verify(manifest,tmp_path)
