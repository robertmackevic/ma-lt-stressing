import random
from typing import Iterator

from torch.utils.data import Sampler

from src.data.dataset import StressingDataset


class BucketSampler(Sampler[int]):
    def __init__(self, dataset: StressingDataset, batch_size: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        # Precompute the lengths of the source sequences in the dataset
        self.indices = list(range(len(dataset)))
        # Sort by length of source sequence
        self.indices.sort(key=lambda idx: dataset[idx][0].size(0))

    def __iter__(self) -> Iterator[int]:
        # Divide the sorted indices into batches
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]

        # Shuffle the order of the batches to maintain some randomness.
        random.shuffle(batches)

        for batch in batches:
            yield from batch

    def __len__(self) -> int:
        return len(self.dataset)
