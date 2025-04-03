from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import torch
from torch import Tensor

from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab, GRAVE_ACCENT, ACUTE_ACCENT, TILDE_ACCENT


class MetricMeter(ABC):
    @abstractmethod
    def update(self, *_) -> None:
        pass


class AverageMeter(MetricMeter):
    def __init__(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float) -> None:
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count


class AccuracyMeter(MetricMeter):
    def __init__(self, total: int) -> None:
        self.total = total
        self.count = 0
        self.accuracy = 0

    def update(self, count: int) -> None:
        self.count += count
        self.accuracy = self.count / self.total


class ConfusionMatrixMeter(MetricMeter):
    def __init__(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.recall = 0
        self.precision = 0
        self.f1 = 0

    def update(self, tp: int, tn: int, fp: int, fn: int) -> None:
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

        tp_fp = self.tp + self.fp
        tp_fn = self.tp + self.fn
        precision_recall = self.precision + self.recall

        self.precision = self.tp / tp_fp if tp_fp > 0 else 0.0
        self.recall = self.tp / tp_fn if tp_fn > 0 else 0.0
        self.f1 = 2 * self.precision * self.recall / precision_recall if precision_recall > 0 else 0.0


def init_metrics(num_total_samples: int) -> Dict[str, MetricMeter]:
    return {
        "sequence_accuracy": AccuracyMeter(num_total_samples),
        "token": ConfusionMatrixMeter(),
        "grave_token": ConfusionMatrixMeter(),
        "acute_token": ConfusionMatrixMeter(),
        "tilde_token": ConfusionMatrixMeter(),
    }


def update_metrics(
        metrics: Dict[str, MetricMeter],
        output: Tensor,
        target: Tensor,
        tokenizer: Tokenizer,
) -> Dict[str, MetricMeter]:
    metrics["sequence_accuracy"].update(count_matching_sequences(output, target))
    metrics["token"].update(*compute_confusion_matrix_for_tokens(
        output, target, list(tokenizer.vocab.stress_token_to_id.values())
    ))
    metrics["grave_token"].update(*compute_confusion_matrix_for_tokens(
        output, target, [tokenizer.vocab.stress_token_to_id[GRAVE_ACCENT]]
    ))
    metrics["acute_token"].update(*compute_confusion_matrix_for_tokens(
        output, target, [tokenizer.vocab.stress_token_to_id[ACUTE_ACCENT]]
    ))
    metrics["tilde_token"].update(*compute_confusion_matrix_for_tokens(
        output, target, [tokenizer.vocab.stress_token_to_id[TILDE_ACCENT]]
    ))
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


def count_matching_sequences(output: Tensor, target: Tensor) -> int:
    padding_mask = target == Vocab.PAD.id
    matching = ((target == output) | padding_mask).all(dim=1)
    return int(torch.sum(matching).item())


def compile_metrics_message(metrics: Dict[str, MetricMeter]) -> str:
    message = "\n"
    for metric, value in metrics.items():
        if isinstance(value, AccuracyMeter):
            message += f"\t{metric}: {value.accuracy:.3f}\n"

        elif isinstance(value, ConfusionMatrixMeter):
            message += f"\t{metric}_precision: {value.precision:.3f}\n"
            message += f"\t{metric}_recall: {value.recall:.3f}\n"
            message += f"\t{metric}_f1: {value.f1:.3f}\n"

        else:
            raise ValueError(f"Unknown metric type {type(value)}")

    return message
