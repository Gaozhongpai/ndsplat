import torch
from torch.utils.data import Dataset, Sampler


class CameraDataset(Dataset):
    def __init__(self, viewpoint_stack):
        self.viewpoint_stack = viewpoint_stack

    def __getitem__(self, index):
        return self.viewpoint_stack[index]

    def __len__(self):
        return len(self.viewpoint_stack)


class InfiniteSampler(Sampler[int]):
    def __init__(self, dataset_len: int, shuffle: bool = True, seed: int = 42):
        self.n = dataset_len
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                idx = torch.randperm(self.n, generator=g)
            else:
                idx = torch.arange(self.n)
            for i in idx.tolist():
                yield i

    def __len__(self):
        return 2**31
