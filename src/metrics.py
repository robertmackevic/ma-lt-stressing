import torch
from torch import Tensor

from src.data.vocab import Vocab


class AverageMeter:
    def __init__(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float) -> None:
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count


def compute_stress_mark_confusion_matrix(output: Tensor, target: Tensor) -> Tensor:
    pass


def compute_sequence_accuracy(output: Tensor, target: Tensor) -> float:
    special_ids = torch.tensor(list(Vocab.SPECIAL_TO_ID.values()), device=target.device)
    special_mask = torch.isin(target, special_ids)
    matching = ((target == output) | special_mask).all(dim=1)
    return matching.float().mean().item()
