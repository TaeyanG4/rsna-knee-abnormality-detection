"""EXP190F data v1.0.0: CONTROL windows, EXP155 masks, Step7 fixed draws."""
from functools import lru_cache
import random
from training.arms import coatnet as c
from training.src import exp155_own0949_oai_v11 as mixed


class Dataset(mixed.MixedStudyDataset):
    def __init__(self, rsna, oai, p, seed, augment=True, smoke=False):
        super().__init__(rsna,oai,p['train_ids'],p['train_labels'],p['train_masks'],
                         p['rsna_train_ids'],p['oai_train_ids'],seed,augment)
        # Explicitly use CONTROL43's factories for both native corpora.
        self.rsna=c.StudyDataset(rsna,self.rsna.ids,self.labels,seed,c.PRESETS['control'],augment)
        self.oai=c.StudyDataset(oai,self.oai.ids,self.labels,seed,c.PRESETS['control'],augment)
        self.augment, self.smoke = augment, smoke

    @lru_cache(maxsize=32)
    def selection(self, epoch):
        if self.smoke or not self.augment:
            return tuple(range(len(self.ids)))
        rsna=[i for i,u in enumerate(self.ids) if u in self.rsna_ids]
        oai=[i for i,u in enumerate(self.ids) if u in self.oai_ids]
        random.Random(42+1009*epoch).shuffle(rsna)
        random.Random(43+1009*epoch).shuffle(oai)
        r,o=2407,1048
        rsna=rsna[(epoch*r)%len(rsna):]+rsna[:(epoch*r)%len(rsna)]
        oai=oai[(epoch*o)%len(oai):]+oai[:(epoch*o)%len(oai)]
        selected=rsna[:r]+oai[:o]
        random.Random(44+1009*epoch).shuffle(selected)
        if len(selected)!=3455 or len(set(selected))!=3455:
            raise ValueError('fixed-step selection differs')
        return tuple(selected)

    def __len__(self):
        return len(self.ids) if self.smoke or not self.augment else 3455

    def __getitem__(self,index):
        epoch=int(self.rsna.epoch.item())
        if epoch!=int(self.oai.epoch.item()):
            raise ValueError('mixed epoch mismatch')
        return super().__getitem__(self.selection(epoch)[index])
