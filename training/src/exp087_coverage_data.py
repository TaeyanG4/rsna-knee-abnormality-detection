"""Minimal compositional B8 input adapter; no model, labels or I/O at import."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


class CoverageStore:
    def __init__(self, manifest_path, expected_sha256):
        from tools import exp087_series_coverage as policy
        path=policy.local(manifest_path)
        if policy.sha(path)!=expected_sha256:
            raise ValueError("coverage manifest changed")
        m=policy.read_json(path)
        if m["status"]!="STAGED_INPUTS_ONLY_NO_TRAINING" or m["policy"]!=policy.POLICY:
            raise ValueError("incomplete or wrong coverage input contract")
        if policy.sha(Path(__file__))!=m['adapter_sha256'] or policy.sha(Path(policy.__file__))!=m['policy_helper_sha256']:
            raise ValueError('frozen input adapter/helper changed')
        self.entries={r["uid"]:r for r in m["entries"]}
        if len(self.entries)!=315 or len(m["entries"])!=315:
            raise ValueError("coverage manifest population differs")
        self.membership=m["membership"]
        self.baseline_corpus=policy.local(m["baseline"]["corpus"])
        if policy.sha(self.baseline_corpus/'manifest.json')!=m['baseline']['corpus_manifest_sha256']:
            raise ValueError('original corpus contract changed')
        for path,expected in policy.BASELINE_METADATA_SHA.items():
            if policy.sha(policy.local(path))!=expected:raise ValueError('original UID/mask contract changed')
        self._cached_uid=None; self._cached=None

    def __getstate__(self):
        # Windows DataLoader spawn must not pickle modules or a parent memmap.
        state=self.__dict__.copy(); state['_cached_uid']=None; state['_cached']=None
        return state

    def entry(self,uid,split):
        row=self.entries.get(uid)
        if row is not None and row["split"]!=split:
            raise ValueError("cross-G1 input request")
        return row

    def volume(self,row):
        from tools import exp087_series_coverage as policy
        uid=row["uid"]
        if self._cached_uid!=uid:
            path=policy.local(row["volume"])
            if policy.sha(path)!=row["file_sha256"]:
                raise ValueError("alternate cache changed")
            a=np.load(path,allow_pickle=False,mmap_mode="r")
            if a.shape!=(12,336,336) or a.dtype!=np.uint8:
                raise ValueError("alternate shape/dtype mismatch")
            self._cached_uid=uid; self._cached=a
        return self._cached


class CoverageStudyWindows:
    """Wrap the unchanged StudyWindows; retain its sample/labels/epoch semantics.

    The underlying dataset always constructs the original normalized tensor.
    Only declared replacement positions receive independently normalized alt
    triplets. All other tensor entries and the same label object are returned.
    """
    def __init__(self, baseline, store, *, enabled=True, force_original=False):
        expected=dict(res=224,k=12 if baseline.train else 24,norm="imagenet",seed=42,
                      jitter=.20,window_policy="within-slot-valid")
        if any(getattr(baseline,k,None)!=v for k,v in expected.items()):
            raise ValueError("B8 sampling/augmentation contract differs")
        if Path(baseline.corpus_dir).resolve()!=store.baseline_corpus:
            raise ValueError('B8 original corpus path differs')
        self.baseline,self.store,self.enabled,self.force_original=baseline,store,enabled,force_original
        self.split="train" if baseline.train else "val"
        if list(baseline.ids)!=store.membership[self.split]:
            raise ValueError("exact G1 study ordering differs")

    def __len__(self):return len(self.baseline)
    def set_epoch(self,epoch):self.baseline.set_epoch(epoch)

    def __getitem__(self,index):
        import torch
        from raptor_common import select_centers,study_rng,windows_from
        from raptor_data import IMAGENET_MEAN,IMAGENET_STD
        from tools.exp087_series_coverage import replacement_map
        b=self.baseline; epoch=int(b.epoch.item())
        x,y,uid=b[index]
        if int(b.epoch.item())!=epoch:
            raise RuntimeError("dataset epoch changed during sample")
        if not self.enabled:return x,y,uid
        row=self.store.entry(uid,self.split)
        if row is None or row["reason"] is not None:return x,y,uid
        centers=select_centers(b.corpus.mask(uid),b.corpus.depth,b.k,b.corpus.spec,
            b.window_policy,study_rng(b.seed,epoch,uid) if b.train else None)
        mapping,_=replacement_map(centers,row["mask"],row["native_indices"])
        if not mapping:return x,y,uid
        alt=windows_from(b.corpus.volume(uid),[centers[i] for i in mapping],b.res) if self.force_original else windows_from(self.store.volume(row),list(mapping.values()),b.res)
        if b.train:
            intensity=study_rng(b.seed,epoch,uid,"intensity").random()
            alt=np.clip(alt*(1+(intensity-.5)*b.jitter),0,1)
        alt=(alt-IMAGENET_MEAN)/IMAGENET_STD
        if not np.isfinite(alt).all():raise ValueError("nonfinite alternate windows")
        x=x.clone()
        for j,i in enumerate(mapping):x[i]=torch.from_numpy(alt[j])
        return x,y,uid
