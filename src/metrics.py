from typing import List, Tuple, Dict

import torch
from torch import Tensor

from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab, GRAVE_ACCENT, ACUTE_ACCENT, TILDE_ACCENT


class AverageMeter:
    def __init__(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float) -> None:
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count


def init_metrics() -> Dict[str, AverageMeter]:
    return {
        "sequence_accuracy": AverageMeter(),
        "token_precision": AverageMeter(),
        "token_recall": AverageMeter(),
        "token_f1": AverageMeter(),
        "grave_token_precision": AverageMeter(),
        "grave_token_recall": AverageMeter(),
        "grave_token_f1": AverageMeter(),
        "acute_token_precision": AverageMeter(),
        "acute_token_recall": AverageMeter(),
        "acute_token_f1": AverageMeter(),
        "tilde_token_precision": AverageMeter(),
        "tilde_token_recall": AverageMeter(),
        "tilde_token_f1": AverageMeter(),
    }


def update_metrics(
        metrics: Dict[str, AverageMeter],
        output: Tensor,
        target: Tensor,
        tokenizer: Tokenizer,
) -> Dict[str, AverageMeter]:
    def _update(prefix: str, tokens: List[int]):
        tp, tn, fp, fn = compute_confusion_matrix_for_tokens(output, target, tokens)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"{prefix}_precision"].update(precision)
        metrics[f"{prefix}_recall"].update(recall)
        metrics[f"{prefix}_f1"].update(f1)

    metrics["sequence_accuracy"].update(compute_sequence_accuracy(output, target))
    _update("token", list(tokenizer.vocab.stress_token_to_id.values()))
    _update("grave_token", [tokenizer.vocab.stress_token_to_id[GRAVE_ACCENT]])
    _update("acute_token", [tokenizer.vocab.stress_token_to_id[ACUTE_ACCENT]])
    _update("tilde_token", [tokenizer.vocab.stress_token_to_id[TILDE_ACCENT]])
    return metrics


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
