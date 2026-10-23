"""R1-only synthetic output-scan races; no models, data or old test suite."""
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import exp087_numerical_diagnostic_v1 as diagnostic


class Candidate:
    def __init__(self,mode=None,size=0,error=None):
        self.mode=mode;self.size=size;self.error=error;self.calls=0
    def stat(self):
        self.calls+=1
        if self.error is not None:raise self.error
        return SimpleNamespace(st_mode=self.mode,st_size=self.size)


def test_each_candidate_has_single_stat_and_regular_mode_check(monkeypatch):
    regular=Candidate(stat.S_IFREG|0o600,13)
    directory=Candidate(stat.S_IFDIR|0o700,4096)
    vanished=Candidate(error=FileNotFoundError('removed by replacement'))
    monkeypatch.setattr(Path,'rglob',lambda *a:iter([regular,directory,vanished]))
    assert diagnostic.output_bytes('synthetic')==13
    assert [x.calls for x in (regular,directory,vanished)]==[1,1,1]


def test_atomic_replacement_between_discovery_and_stat_keeps_remaining_file(monkeypatch,tmp_path):
    temporary=tmp_path/'current-attempt.json.tmp';final=tmp_path/'current-attempt.json'
    temporary.write_bytes(b'newer-state');final.write_bytes(b'old')
    def discovered_then_replaced(self,pattern):
        # The scanner discovered both names. Child removes temporary via replace
        # before parent stats that candidate, exactly the reported interleaving.
        candidates=[temporary,final]
        os.replace(temporary,final)
        yield from candidates
    monkeypatch.setattr(Path,'rglob',discovered_then_replaced)
    assert diagnostic.output_bytes(tmp_path)==len(b'newer-state')
    assert final.read_bytes()==b'newer-state' and not temporary.exists()


def test_regular_file_replaced_by_directory_is_skipped(monkeypatch,tmp_path):
    candidate=tmp_path/'changing';candidate.write_bytes(b'previous-file')
    def discover_then_change(self,pattern):
        candidate.unlink();candidate.mkdir()
        yield candidate
    monkeypatch.setattr(Path,'rglob',discover_then_change)
    assert diagnostic.output_bytes(tmp_path)==0


@pytest.mark.parametrize('error',[PermissionError('denied'),OSError('device error')])
def test_non_disappearance_io_errors_propagate_unchanged(monkeypatch,error):
    candidate=Candidate(error=error)
    monkeypatch.setattr(Path,'rglob',lambda *a:iter([candidate]))
    with pytest.raises(type(error)) as observed:diagnostic.output_bytes('synthetic')
    assert observed.value is error and candidate.calls==1
