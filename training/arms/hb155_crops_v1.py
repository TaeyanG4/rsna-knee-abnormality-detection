"""H-B-155 same MIL window recipe, alternate prebuilt geometry pixels v1.1.0."""
from training.arms import native_resolution_v2 as native
from training.arms.mil_screen_hooks_v1 import MILWindows
from tools.hb155_crop_v1 import CONTRACT

CONTROL_CONTRACT=native.CONTRACT
# v1.1.0: side flag is an authenticated auxiliary cache channel, not a new
# model channel. Keep the frozen RGB-like adjacent-slice recipe unchanged.
SIDE_CONTRACT='hb155_compartment64_448_side_v2'
CONTRACTS=(CONTROL_CONTRACT,CONTRACT,SIDE_CONTRACT)

def side_flag(corpus,uid):
    h=native.h
    if corpus.contract!=SIDE_CONTRACT:return 0
    if not hasattr(corpus,'_side_flags'):
        p=corpus.dir/'side_unknown.npy'
        h.need(h.sha(p)==corpus.manifest['side_unknown_sha256'],'side flag hash drift')
        flags=h.np.load(p,allow_pickle=False)
        h.need(flags.shape==(len(corpus.ids),64,1,1) and h.np.isin(flags,[0,1]).all(),'side flag shape/value drift')
        corpus._side_flags=flags
    row=corpus._side_flags[corpus.row[uid]]
    h.need(h.np.all(row==row[0]),'inconsistent side flag')
    return int(row[0,0,0])

class Windows(native.NativeWindows):
    def __getitem__(self,index):
        h=native.h
        if self.corpus is None: self.corpus=h.common.Corpus(self.corpus_dir)
        h.need(self.corpus.contract in CONTRACTS and self.corpus.spec['img']==448,'H-B-155 source differs')
        uid,epoch=self.ids[index],int(self.epoch.item())
        self.last_side_unknown=side_flag(self.corpus,uid)
        rng=h.common.study_rng(self.seed,epoch,uid) if self.train else None
        centers=h.common.select_centers(self.corpus.mask(uid),64,self.k,self.corpus.spec,'within-slot-valid',rng)
        x=native.scale_windows(self.corpus.volume(uid),centers,self.resolution)
        if self.train:
            intensity=h.common.study_rng(self.seed,epoch,uid,'intensity').random()
            x=h.np.clip(x*(1+(intensity-.5)*.20),0,1)
            x=h.augment_windows(x,centers,rng,self.corpus.spec)
        mean=h.np.array([.485,.456,.406],h.np.float32).reshape(3,1,1)
        std=h.np.array([.229,.224,.225],h.np.float32).reshape(3,1,1)
        x=(x-mean)/std
        h.need(x.dtype==h.np.float32 and h.np.isfinite(x).all(),'invalid H-B-155 windows')
        return h.torch.from_numpy(x),h.torch.from_numpy(self.labels[uid]),uid
