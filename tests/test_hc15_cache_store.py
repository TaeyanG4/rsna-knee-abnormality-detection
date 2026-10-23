"""Tiny synthetic NPY payloads only; stdlib, no DICOM/numpy/torch/GPU."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.adapters.hc15_cache_store import CacheStore, CacheError, GEOMETRY, inspect_npy


def npy(n=3, dtype='|u1', fortran=False):
    header = repr({'descr': dtype, 'fortran_order': fortran, 'shape': (n, 384, 384)}).encode()
    header += b' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + b'\n'
    return b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header + b'\x01' * (n * 384 * 384)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / 'cache'
        self.contract = {'source_manifest_sha256': 'a'*64, 'adapter_sha256': 'b'*64,
                         'producer_sha256': 'c'*64, 'decoder_sha256': 'e'*64, 'geometry': GEOMETRY,
                         'archive_index_sha256':'f'*64, 'source_archive':{'path':'synthetic.zip','index':'synthetic.sqlite','bytes':123},
                         'study_uids': ['10','20'],
                         'series': [{'StudyInstanceUID':'10','SeriesInstanceUID':'1.1'},
                                    {'StudyInstanceUID':'10','SeriesInstanceUID':'1.2'},
                                    {'StudyInstanceUID':'20','SeriesInstanceUID':'2.1'}]}
        self.budget = dict(max_bytes=8*1024*1024, max_total_seconds=30, reserve_bytes=0)
        self.provenance = dict(source_members_sha256='d'*64, decoder_sha256='e'*64)

    def store(self, contract=None, **budget):
        return CacheStore(self.out, contract or self.contract, **(self.budget | budget))

    def put(self, store, sid='1.1'):
        return store.commit(sid, npy(), decoded_count=3, **self.provenance)

    def finish(self, store):
        for sid in ['1.1','1.2','2.1']:
            if not store.contains(sid): self.put(store,sid)
        return store.finalize()

    def test_chunk_resume_and_runner_schema(self):
        with self.store() as s:
            self.put(s)
            with self.assertRaisesRegex(CacheError, 'Not all'): s.finalize()
        original = (self.out/'arrays/1.1/volume.npy').read_bytes()
        with self.store() as s:
            self.assertTrue(s.contains('1.1'))
            path=self.finish(s)
            doc=json.loads(path.read_text())
            self.assertEqual(doc['schema'],'hc15-cache-v1')
            self.assertTrue(doc['complete'])
            self.assertFalse(doc['gpu_dispatch_allowed'])
            with self.assertRaisesRegex(CacheError,'immutable'): self.put(s)
        self.assertEqual(original,(self.out/'arrays/1.1/volume.npy').read_bytes())
        with self.store() as s: self.assertEqual(s.finalize(),path)

    def test_exception_retains_commits_and_releases_own_lock(self):
        with self.assertRaisesRegex(RuntimeError,'decode failed'):
            with self.store() as s:
                self.put(s);raise RuntimeError('decode failed')
        self.assertFalse((self.out/'WRITER.lock').exists())
        with self.store() as s: self.assertTrue(s.contains('1.1'))

    def test_lock_is_not_taken_or_removed(self):
        with self.store() as s:
            with self.assertRaisesRegex(CacheError,'lock exists'): self.store().__enter__()
            self.assertTrue((self.out/'WRITER.lock').exists());self.put(s)

    def test_changed_budget_source_and_series_order_refused(self):
        with self.store(): pass
        for field in ['source_manifest_sha256','adapter_sha256','producer_sha256','decoder_sha256']:
            c=copy.deepcopy(self.contract);c[field]='f'*64
            with self.assertRaisesRegex(CacheError,'identity drift'): self.store(c).__enter__()
        c=copy.deepcopy(self.contract);c['series'].reverse()
        with self.assertRaisesRegex(CacheError,'identity drift'): self.store(c).__enter__()
        with self.assertRaisesRegex(CacheError,'identity drift'): self.store(max_total_seconds=31).__enter__()

    def test_invalid_npy_does_not_commit(self):
        with self.store() as s:
            for bad in [npy(dtype='|O'),npy(fortran=True),npy()[:-1],npy()+b'x',b'junk']:
                with self.assertRaises((CacheError,ValueError)):
                    s.commit('1.1',bad,decoded_count=3,**self.provenance)
                self.assertFalse(s.contains('1.1'))

    def test_npy_types_are_exact(self):
        for n, f in [(3.0,False),(3,0)]:
            header=repr({'descr':'|u1','fortran_order':f,'shape':(n,384,384)}).encode()+b'\n'
            bad=b'\x93NUMPY\x01\x00'+len(header).to_bytes(2,'little')+header+b'\x00'*(3*384*384)
            with self.assertRaisesRegex(CacheError,'field types'):inspect_npy(io.BytesIO(bad),len(bad),3)

    def test_decoder_identity_is_frozen_in_commit_and_resume(self):
        with self.store() as s:
            with self.assertRaisesRegex(CacheError,'Decoder identity'):
                s.commit('1.1',npy(),decoded_count=3,source_members_sha256='d'*64,decoder_sha256='f'*64)
            self.put(s)
        p=self.out/'records/1.1.json';r=json.loads(p.read_text());r['decoder_sha256']='f'*64;p.write_text(json.dumps(r))
        with self.assertRaisesRegex(CacheError,'Decoder identity'):self.store().__enter__()

    def test_initial_budget_failure_creates_nothing(self):
        with self.assertRaisesRegex(CacheError,'Byte budget'):self.store(max_bytes=1).__enter__()
        self.assertFalse(self.out.exists())
        with patch('training.adapters.hc15_cache_store.shutil.disk_usage',return_value=type('D',(),{'free':0})()):
            with self.assertRaisesRegex(CacheError,'Disk reserve'):self.store().__enter__()
        self.assertFalse(self.out.exists())

    def test_duplicate_and_foreign_series_rejected(self):
        with self.store() as s:
            self.put(s)
            for sid in ['1.1','999']:
                with self.assertRaisesRegex(CacheError,'Foreign/already'):self.put(s,sid)

    def test_partial_directory_preserved(self):
        with self.store() as s:
            p=self.out/'arrays/1.1';p.mkdir(parents=True);(p/'old.part').write_bytes(b'interrupted')
            # Mimic an orphan found at a later clean session's resume.
        with self.store() as s:
            self.put(s)
            old=list((self.out/'abandoned').rglob('old.part'))
            self.assertEqual(len(old),1);self.assertEqual(old[0].read_bytes(),b'interrupted')

    def test_corrupt_completed_array_blocks_resume(self):
        with self.store() as s:self.put(s)
        p=self.out/'arrays/1.1/volume.npy';v=bytearray(p.read_bytes());v[-1]=4;p.write_bytes(v)
        with self.assertRaisesRegex(CacheError,'Array changed'):self.store().__enter__()
        self.assertFalse((self.out/'WRITER.lock').exists())

    def test_no_valid_series_for_study_blocks_finalization(self):
        with self.store() as s:
            self.put(s,'1.1');self.put(s,'1.2')
            s.commit('2.1',None,decoded_count=2,**self.provenance)
            with self.assertRaisesRegex(CacheError,'no eligible'):s.finalize()
            self.assertFalse((self.out/'cache.json').exists())

    def test_ineligible_alternative_series_permitted(self):
        with self.store() as s:
            self.put(s,'1.1');self.put(s,'2.1')
            s.commit('1.2',None,decoded_count=2,**self.provenance)
            self.assertTrue(s.finalize().exists())

    def test_bad_or_unsafe_membership(self):
        for sid in ['../escape','abc','1/2']:
            c=copy.deepcopy(self.contract);c['series'][0]['SeriesInstanceUID']=sid
            with self.assertRaisesRegex(CacheError,'Unsafe'):self.store(c)
        c=copy.deepcopy(self.contract);c['series'][0]['StudyInstanceUID']='99'
        with self.assertRaisesRegex(CacheError,'coverage'):self.store(c)

    def test_disk_and_byte_limits_prevent_write(self):
        with self.store(max_bytes=100000) as s:
            with self.assertRaisesRegex(CacheError,'Byte budget'):self.put(s)
            self.assertFalse((self.out/'arrays').exists())
        # New independent test directory, with a different explicit budget.
        self.out=Path(self.tmp.name)/'cache2'
        with self.store() as s:
            with patch('training.adapters.hc15_cache_store.shutil.disk_usage',return_value=type('D',(),{'free':0})()):
                with self.assertRaisesRegex(CacheError,'Disk reserve'):self.put(s)

    def test_total_elapsed_cannot_be_reset_by_resume(self):
        with self.store() as s:self.put(s)
        p=self.out/'state.json';p.write_text(json.dumps({'spent_seconds':31,'active':False}))
        with self.assertRaisesRegex(CacheError,'Time|time'):self.store().__enter__()
        self.assertEqual(json.loads(p.read_text())['spent_seconds'],31)

    def test_unclean_ledger_never_auto_reset(self):
        with self.store():pass
        p=self.out/'state.json';value={'spent_seconds':5,'active':True};p.write_text(json.dumps(value))
        with self.assertRaisesRegex(CacheError,'Unclean'):self.store().__enter__()
        self.assertEqual(json.loads(p.read_text()),value)

    def test_foreign_existing_directory_untouched(self):
        self.out.mkdir();p=self.out/'original';p.write_text('preserve')
        with self.assertRaisesRegex(CacheError,'not this cache'):self.store().__enter__()
        self.assertEqual(p.read_text(),'preserve')

    def test_no_cuda_or_numerical_package_import(self):
        import subprocess,sys
        code="import sys;import training.adapters.hc15_cache_store;assert not {'torch','numpy','pydicom'} & set(sys.modules)"
        subprocess.run([sys.executable,'-c',code],check=True,timeout=20)

    def test_caller_mutation_cannot_change_frozen_identity(self):
        store=self.store()
        self.contract['series'][0]['StudyInstanceUID']='20'
        self.contract['series'][2]['StudyInstanceUID']='10'
        with store as s:
            result=json.loads(self.finish(s).read_text())
            self.assertEqual(result['series']['1.1']['study'],'10')
            self.assertEqual(result['series']['2.1']['study'],'20')


if __name__ == '__main__':
    unittest.main()
