import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from tools import h216_oai_cache800_v1 as m


class ProducerContract(unittest.TestCase):
    def test_exact_partition_no_selection_replacement(self):
        ids=[f'synthetic-{i}' for i in range(800)]
        b=dict(schedules=dict(oai_required_ids=ids),pilot_reusable_ids=sorted(ids[:3]))
        order,reuse,missing=m.split_required(b,ids[:3]+['other']*5)
        self.assertEqual(order,ids);self.assertEqual(reuse,ids[:3]);self.assertEqual(missing,ids[3:])
        with self.assertRaisesRegex(ValueError,'PARTITION'):m.split_required(b,ids[:4])
        bad=dict(b,schedules=dict(oai_required_ids=ids[:-1]+[ids[0]]))
        with self.assertRaisesRegex(ValueError,'PARTITION'):m.split_required(bad,ids[:3])

    def test_bounds_include_retained_native_raw_and_two_v4_copies(self):
        rows=[dict(group_key='synthetic',frame_count=7,Anatomical_Plane='Sagittal')]
        h={'synthetic':dict(frame_count=7,rows=10,columns=20)}
        b=m.storage_bound(rows,h);native=7*10*20*2;v4=7*4*(384**2+48)
        self.assertEqual(b['required_free_bytes'],native+2*v4+4*native+7*2**20+101*m.GIB)
        h['synthetic']['frame_count']=8
        with self.assertRaisesRegex(ValueError,'HEADER'):m.storage_bound(rows,h)

    def test_safe_descendant_rejects_root_parent_sibling(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'.local/scratch/slot-b') as tmp:
            root=Path(tmp);self.assertEqual(m.safe_destination(root/'child',root),root/'child')
            for p in (root,root.parent,root.parent/'sibling'):
                with self.assertRaisesRegex(ValueError,'ESCAPE'):m.safe_destination(p,root)

    def archive(self,dest,names):
        with tarfile.open(dest,'w:gz') as tf:
            for name,data in names:
                i=tarfile.TarInfo(name);i.size=len(data);tf.addfile(i,io.BytesIO(data))
        st=dest.stat();return dict(path=str(dest),size=st.st_size,mtime_ns=st.st_mtime_ns)

    def test_archive_bytes_only_counts_selected_members(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'.local/scratch/slot-b') as tmp:
            ar=self.archive(Path(tmp)/'source.tar.gz',[('a',b'123'),('b',b'45678'),('projection',b'9'*12)])
            g=dict(members=['a','b'],frame_count=2)
            self.assertEqual(m.archive_member_bytes(ar,g),8)
            ar['mtime_ns']-=1
            with self.assertRaisesRegex(ValueError,'STAMP'):m.archive_member_bytes(ar,g)

    def test_duplicate_or_missing_archive_member_rejected(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'.local/scratch/slot-b') as tmp:
            p=Path(tmp)/'dup.tar.gz';ar=self.archive(p,[('a',b'1'),('a',b'2')])
            with self.assertRaisesRegex(ValueError,'INVALID'):m.archive_member_bytes(ar,dict(members=['a'],frame_count=1))
            with self.assertRaisesRegex(ValueError,'MISSING'):m.archive_member_bytes(ar,dict(members=['b'],frame_count=1))

    def test_copy_preserves_original_and_checks_hash_without_alias(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'.local/scratch/slot-b') as tmp:
            src=Path(tmp)/'source';src.write_bytes(b'original-synthetic'*1000);dst=Path(tmp)/'new'/'copy';events=[]
            expected=hashlib.sha256(src.read_bytes()).hexdigest()
            m.copy_checked(src,dst,expected,lambda:events.append(1))
            self.assertEqual(src.read_bytes(),dst.read_bytes());self.assertTrue(events)
            dst.write_bytes(b'different');self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(),expected)
            with self.assertRaisesRegex(ValueError,'EXISTS'):m.copy_checked(src,dst,expected,lambda:None)

    def test_stream_hash_reports_real_byte_progress(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'.local/scratch/slot-b') as tmp:
            p=Path(tmp)/'source';p.write_bytes(b'abc'*10000);events=[]
            self.assertEqual(m.stream_hash(p,events.append),hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(events,['checksum'])

    def test_absent_pilot_cannot_admit_or_allocate(self):
        with patch.object(m,'registration',return_value={}),patch.object(m,'pilot_ready',side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):m.admit_cpu()

    def test_unchanged_recipe_helpers_and_no_gpu_import_cpu(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        source=Path(m.__file__).read_text(encoding='utf-8')
        self.assertIn('extraction.extract_group(audit,group,directory)',source)
        self.assertIn('geometry.validate_record(rec,s,hs)',source)
        self.assertIn('pilot.render_one(study,anchor,CACHE,pxmod,ct,progress)',source)


if __name__=='__main__':unittest.main()
