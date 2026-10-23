"""Synthetic byte orchestration; no real DICOM, archive, torch or GPU."""
import time
import unittest
import tempfile
import sqlite3
import zipfile
import json
from contextlib import closing
from pathlib import Path
from training.adapters.hc15_archive_decoder import ArchiveDecoder, check_single_frame, decode_records, produce_chunk, zip_info
from training.adapters.hc15_cache_store import CacheError, digest_file
from tests import test_hc15_cache_store as fixture
npy = fixture.npy


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.row={'StudyInstanceUID':'10','SeriesInstanceUID':'1.1','Anatomical_Plane':'Sagittal'}
        self.members=[{'path':f'train_series/10/1.1/{k}.dcm','zip_info':{}} for k in range(3)]

    def run_decode(self, members=None, frame=lambda b,p:b):
        return decode_records(self.row,self.members if members is None else members,lambda m:b'synthetic',
                              frame,lambda slices,plane:npy(len(slices)),time.monotonic()+5)

    def test_decode_order_and_short_series(self):
        seen=[]
        p,n,h=self.run_decode(frame=lambda b,path:seen.append(path) or b)
        self.assertEqual(n,3);self.assertEqual(len(h),64);self.assertEqual(seen,[m['path'] for m in self.members])
        p,n,h=self.run_decode(self.members[:2]);self.assertIsNone(p);self.assertEqual(n,2)

    def test_missing_foreign_duplicate_members_fail(self):
        for members in [[], self.members+[self.members[0]], [{'path':'train_series/20/1.1/a.dcm','zip_info':{}}]]:
            with self.assertRaises(CacheError):self.run_decode(members)

    def test_decode_error_is_not_an_ineligible_series(self):
        def fail(*args):raise RuntimeError('multiframe/corrupt source')
        with self.assertRaisesRegex(RuntimeError,'multiframe'):self.run_decode(frame=fail)

    def test_enhanced_or_bad_frame_count_rejected(self):
        for ds in [{'NumberOfFrames':2},{'NumberOfFrames':0},{'SharedFunctionalGroupsSequence':[]},{'PerFrameFunctionalGroupsSequence':[]}]:
            with self.assertRaisesRegex(CacheError,'Enhanced'):check_single_frame(ds)
        check_single_frame({});check_single_frame({'NumberOfFrames':1})

    def test_deadline_and_zip_identity(self):
        with self.assertRaisesRegex(CacheError,'deadline'):
            decode_records(self.row,self.members,None,None,None,time.monotonic()-1)
        with self.assertRaisesRegex(CacheError,'identity'):
            zip_info({'path':'x','zip_info':{'filename':'y'}})


class ChunkTests(unittest.TestCase):
    setUp = fixture.StoreTests.setUp
    store = fixture.StoreTests.store
    put = fixture.StoreTests.put
    def test_chunk_skips_only_validated_completed_series(self):
        calls=[]
        def decoder(row,deadline):calls.append(row['SeriesInstanceUID']);return npy(),3,'d'*64
        decoder.archive_contract=self.contract['source_archive'];decoder.index_sha256='f'*64
        with self.store() as s:
            result=produce_chunk(s,decoder,decoder_sha256='e'*64,max_series=1,deadline=time.monotonic()+5)
            self.assertFalse(result['complete']);self.assertEqual(calls,['1.1'])
        with self.store() as s:
            result=produce_chunk(s,decoder,decoder_sha256='e'*64,max_series=5,deadline=time.monotonic()+5)
            self.assertTrue(result['complete']);self.assertEqual(calls,['1.1','1.2','2.1'])

    def test_chunk_error_preserves_prior_work(self):
        def fail(row,deadline):raise RuntimeError('decode stopped')
        fail.archive_contract=self.contract['source_archive'];fail.index_sha256='f'*64
        with self.store() as s:
            self.put(s)
            with self.assertRaisesRegex(RuntimeError,'decode stopped'):
                produce_chunk(s,fail,decoder_sha256='e'*64,max_series=2,deadline=time.monotonic()+5)
            self.assertEqual(list(s.records),['1.1']);self.assertFalse((self.out/'cache.json').exists())

    def test_source_mismatch_and_expired_deadline_block_before_decode(self):
        def decoder(*args):self.fail('must not decode')
        decoder.archive_contract=self.contract['source_archive'];decoder.index_sha256='a'*64
        with self.store() as s:
            with self.assertRaisesRegex(CacheError,'index binding'):
                produce_chunk(s,decoder,decoder_sha256='e'*64,max_series=1,deadline=time.monotonic()+5)
            decoder.index_sha256='f'*64;decoder.archive_contract={'path':'wrong'}
            with self.assertRaisesRegex(CacheError,'archive binding'):
                produce_chunk(s,decoder,decoder_sha256='e'*64,max_series=1,deadline=time.monotonic()+5)
            decoder.archive_contract=self.contract['source_archive']
            for sid in ['1.1','1.2','2.1']:self.put(s,sid)
            for deadline in [time.monotonic()-1,float('nan'),float('inf')]:
                with self.assertRaisesRegex(CacheError,'deadline'):
                    produce_chunk(s,decoder,decoder_sha256='e'*64,max_series=1,deadline=deadline)
            with self.assertRaisesRegex(CacheError,'deadline'):s.finalize(deadline=time.monotonic()-1)
            self.assertFalse((self.out/'cache.json').exists())


class ArchiveTests(unittest.TestCase):
    def test_tiny_readonly_index_and_zip_connection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'synthetic.zip';index=root/'index.sqlite'
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
                for k in range(3):z.writestr(f'train_series/10/1.1/{k}.dcm',b'synthetic not DICOM')
            with closing(sqlite3.connect(index)) as db, zipfile.ZipFile(archive) as z:
                db.execute('CREATE TABLE files (path TEXT,info TEXT,split TEXT,uid TEXT,series TEXT)')
                for i in z.infolist():
                    info={k:getattr(i,k) for k in ['compress_type','header_offset','flag_bits','CRC','compress_size','file_size','date_time','filename','_end_offset']}
                    db.execute('INSERT INTO files VALUES(?,?,?,?,?)',(i.filename,json.dumps(info),'train','10','1.1'))
                db.commit()
            before=(digest_file(index),digest_file(archive));seen=[]
            contract={'index':str(index),'path':str(archive),'bytes':archive.stat().st_size}
            decoder=ArchiveDecoder(contract,before[0],lambda b,p:seen.append(b) or b,lambda slices,plane:npy(len(slices)))
            payload,n,source_hash=decoder({'StudyInstanceUID':'10','SeriesInstanceUID':'1.1','Anatomical_Plane':'Sagittal'},time.monotonic()+5)
            self.assertEqual(n,3);self.assertEqual(seen,[b'synthetic not DICOM']*3)
            self.assertEqual(before,(digest_file(index),digest_file(archive)))
            with self.assertRaisesRegex(CacheError,'index drift'):
                ArchiveDecoder(contract,'f'*64,None,None)
            archive.write_bytes(archive.read_bytes()+b'x')
            with self.assertRaisesRegex(CacheError,'changed during'):
                decoder({'StudyInstanceUID':'10','SeriesInstanceUID':'1.1'},time.monotonic()+5)
