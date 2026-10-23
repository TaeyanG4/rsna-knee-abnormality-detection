"""Synthetic HC15 process integration: stalled decoder and partial chunk.

Reuses the existing process supervisor unchanged. No MRI, torch or CUDA imports.
These tests do not create a production allocation or execution-admission CLI.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

from training.adapters.hc15_cache_store import CacheError, digest_file
from tests import test_hc15_cache_store as fixture

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/process_resource_receipt.py'
spec = importlib.util.spec_from_file_location('hc15_process_receipt', SOURCE)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)

CHILD = '''
import json,sys,time
from pathlib import Path
from training.adapters.hc15_cache_store import CacheStore
from training.adapters.hc15_archive_decoder import produce_chunk
from tests.test_hc15_cache_store import npy
root=Path(sys.argv[1]); mode=sys.argv[2]
config=json.loads((root/'config.json').read_text())
calls=[]
def decoder(row,deadline):
    calls.append(row['SeriesInstanceUID'])
    if mode=='hang' and len(calls)==2:
        (root/'entered_blocked_decoder').write_text('second series')
        time.sleep(30)
    return npy(),3,'d'*64
decoder.archive_contract=config['contract']['source_archive']
decoder.index_sha256=config['contract']['archive_index_sha256']
with CacheStore(root/'cache',config['contract'],**config['budget']) as store:
    result=produce_chunk(store,decoder,decoder_sha256='e'*64,
                         max_series=3 if mode=='hang' else 1,
                         deadline=time.monotonic()+20)
    (root/'chunk-result.json').write_text(json.dumps(result))
'''


class SupervisionTests(unittest.TestCase):
    setUp = fixture.StoreTests.setUp
    store = fixture.StoreTests.store

    def run_child(self, mode, timeout):
        root = self.out.parent
        (root/'config.json').write_text(json.dumps({'contract': self.contract,
                                                  'budget': self.budget}))
        result = supervisor.supervise(
            [sys.executable, '-c', CHILD, str(root), mode],
            root/'receipt', timeout, cwd=str(ROOT))
        self.assertEqual(result['retries'], 0)
        self.assertEqual(result['cleanup']['survivors'], [])
        self.assertEqual(result['cleanup']['errors'], [])
        self.assertEqual(result['supervisor_sha256'], digest_file(SOURCE))
        self.assertEqual(json.loads((root/'receipt/receipt.json').read_text()), result)
        return result

    def test_hung_decoder_is_stopped_and_partial_cache_cannot_auto_resume(self):
        sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        try:
            result = self.run_child('hang', 2.0)
            self.assertTrue((self.out.parent/'entered_blocked_decoder').exists())
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(result['failure']['type'], 'TimeoutError')
            self.assertNotEqual(result['child_exit_code'], 0)
            self.assertLess(result['elapsed_seconds'], 6)
            self.assertIsNone(sentinel.poll())
            self.assertNotIn(sentinel.pid, [p['pid'] for p in result['observed_identities']])
            self.assertTrue((self.out/'records/1.1.json').exists())
            self.assertFalse((self.out/'records/1.2.json').exists())
            self.assertFalse((self.out/'cache.json').exists())
            self.assertFalse((self.out.parent/'chunk-result.json').exists())
            self.assertTrue((self.out/'WRITER.lock').exists())
            self.assertTrue(json.loads((self.out/'state.json').read_text())['active'])
            saved = digest_file(self.out/'arrays/1.1/volume.npy')
            with self.assertRaisesRegex(CacheError, 'lock exists'):
                with self.store():
                    self.fail('unclean cache must not resume automatically')
            self.assertEqual(saved, digest_file(self.out/'arrays/1.1/volume.npy'))
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=3)

    def test_clean_partial_process_is_not_complete_cache_or_gpu_admission(self):
        result = self.run_child('partial', 5.0)
        self.assertEqual(result['status'], 'COMPLETE')
        chunk = json.loads((self.out.parent/'chunk-result.json').read_text())
        self.assertFalse(chunk['complete'])
        self.assertFalse(chunk['gpu_dispatch_allowed'])
        self.assertIsNone(chunk['cache_manifest'])
        self.assertFalse((self.out/'cache.json').exists())
        self.assertFalse((self.out/'WRITER.lock').exists())
        state = json.loads((self.out/'state.json').read_text())
        self.assertFalse(state['active'])
        with self.store() as store:
            self.assertTrue(store.contains('1.1'))
            self.assertEqual(store.spent, state['spent_seconds'])
            self.assertGreaterEqual(store.spent, 0)


if __name__ == '__main__':
    unittest.main()
