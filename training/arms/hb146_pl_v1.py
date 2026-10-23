"""H-B-146 1.0.0: replace training targets only, preserving shared MIL hooks."""
from training.arms import mil_screen_hooks_v1 as hooks


class SoftSource(hooks.torch.utils.data.Dataset):
    def __init__(self, base, labels):
        self.base, self.labels = base, labels
        self.k, self.ids = base.k, base.ids

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __getitem__(self, index):
        x, _, uid = self.base[index]
        return x, hooks.torch.from_numpy(self.labels[uid]), uid


class Arm(hooks.Arm):
    def __init__(self, labels=None):
        self.labels = labels

    def dataset(self, dataset, spec=None, stage=None, *, train):
        return SoftSource(dataset, self.labels) if train and self.labels is not None else dataset
