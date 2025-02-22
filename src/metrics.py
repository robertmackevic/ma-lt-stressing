from typing import List, Tuple

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


def compute_confusion_matrix_for_tokens(
        output: Tensor, target: Tensor, token_ids: List[int]
) -> Tuple[int, int, int, int]:
    token_ids = torch.tensor(token_ids, device=target.device)

    target_positive = torch.isin(target, token_ids)
    output_positive = torch.isin(output, token_ids)

    tp = torch.sum(target_positive & (output == target)).item()
    fn = torch.sum(target_positive & (output != target)).item()
    fp = torch.sum((~target_positive) & output_positive).item()
    tn = torch.sum((~target_positive) & (~output_positive)).item()

    return tp, tn, fp, fn


def compute_sequence_accuracy(output: Tensor, target: Tensor) -> float:
    special_ids = torch.tensor(list(Vocab.SPECIAL_TO_ID.values()), device=target.device)
    special_mask = torch.isin(target, special_ids)
    matching = ((target == output) | special_mask).all(dim=1)
    return matching.float().mean().item()
