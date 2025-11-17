import numpy as np
import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):


    def __init__(self, labels, batch_size=16, repeat_minority=3, drop_last=True):
        self.labels = np.asarray(labels, dtype=np.int64)
        assert batch_size % 2 == 0, "batch_size must be even for 1:1 balancing."
        self.batch_size = int(batch_size)
        self.half = self.batch_size // 2
        self.drop_last = bool(drop_last)

        idx_0 = np.where(self.labels == 0)[0]
        idx_1 = np.where(self.labels == 1)[0]
        if len(idx_0) >= len(idx_1):
            self.major_idx = idx_0
            self.minor_idx = np.concatenate([idx_1] * int(repeat_minority), axis=0)
        else:
            self.major_idx = idx_1
            self.minor_idx = np.concatenate([idx_0] * int(repeat_minority), axis=0)

        self.n_batches = min(len(self.major_idx) // self.half, len(self.minor_idx) // self.half)

    def __len__(self):
        return self.n_batches if self.drop_last else self.n_batches

    def __iter__(self):
        major = self.major_idx.copy()
        minor = self.minor_idx.copy()
        np.random.shuffle(major)
        np.random.shuffle(minor)

        major = major[: self.n_batches * self.half]
        minor = minor[: self.n_batches * self.half]

        for b in range(self.n_batches):
            m0 = major[b * self.half : (b + 1) * self.half]
            m1 = minor[b * self.half : (b + 1) * self.half]
            batch_idx = np.concatenate([m0, m1], axis=0)
            np.random.shuffle(batch_idx)
            yield torch.as_tensor(batch_idx, dtype=torch.int64)
