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
    non_padding_mask = target != Vocab.PAD.id

    valid_target = target[non_padding_mask]
    valid_output = output[non_padding_mask]

    target_positive = torch.isin(valid_target, token_ids)
    output_positive = torch.isin(valid_output, token_ids)

    tp = torch.sum(target_positive & (valid_output == valid_target)).item()
    fn = torch.sum(target_positive & (valid_output != valid_target)).item()
    fp = torch.sum((~target_positive) & output_positive).item()
    tn = torch.sum((~target_positive) & (~output_positive)).item()

    return tp, tn, fp, fn


def compute_sequence_accuracy(output: Tensor, target: Tensor) -> float:
    padding_mask = target == Vocab.PAD.id
    matching = ((target == output) | padding_mask).all(dim=1)
    return matching.float().mean().item()
