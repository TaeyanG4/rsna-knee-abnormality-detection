"""R+ v1.0.0: exact EXP190F mixed draws, or all L2 studies once."""
import torch
from training.arms import coatnet as c
from training.arms.exp190f_data import Dataset as OaiDataset, mixed


class L2Dataset(c.StudyDataset):
    def __getitem__(self,index):
        x,y,uid=super().__getitem__(index)
        return x,y,torch.ones_like(y),uid


def Dataset(rsna,oai,p,seed,augment=True,smoke=False):
    if p['oai_train_ids']:
        return OaiDataset(rsna,oai,p,seed,augment=augment,smoke=smoke)
    return L2Dataset(rsna,p['train_ids'],p['train_labels'],seed,c.PRESETS['control'],augment=augment)
