"""
Drop-in replacement for torch.utils.data.DataLoader over an in-memory TensorDataset.

The default DataLoader fetches every sample individually and collates it, which
dominates the training time of the small MLP used here. FastLoader slices the
tensors directly and consumes the global PyTorch random generator exactly as
DataLoader does in PyTorch 1.13 (num_workers=0):

  * one draw for the iterator's base seed, on every iteration over the loader;
  * with shuffle=True, one more draw that seeds the private generator of the
    RandomSampler, followed by torch.randperm(n) on that generator.

Batches, their order and the random stream seen by the rest of the program are
therefore identical to those of DataLoader, so results are bit-for-bit
reproducible with the original implementation (checked by
revision/check_fastload.py).
"""

import torch


class FastLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.X, self.y = dataset.tensors
        self.batch_size = int(batch_size)
        self.shuffle = shuffle

    def __len__(self):
        n = len(self.y)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # _BaseDataLoaderIter.__init__: base seed drawn from the global generator
        torch.empty((), dtype=torch.int64).random_()
        n, bs = len(self.y), self.batch_size
        if self.shuffle:
            # RandomSampler.__iter__ (generator=None)
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            g = torch.Generator()
            g.manual_seed(seed)
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                yield self.X[idx], self.y[idx]
        else:
            for i in range(0, n, bs):
                yield self.X[i:i + bs], self.y[i:i + bs]
