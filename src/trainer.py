from argparse import Namespace
from os import listdir, makedirs
from typing import Dict, Optional, Tuple
from warnings import filterwarnings

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab
from src.metrics import AverageMeter, AccuracyMeter, ConfusionMatrixMeter, init_metrics, update_metrics
from src.model.transformer import Seq2SeqTransformer
from src.paths import RUNS_DIR, CONFIG_FILE, SOURCE_TOKENIZER_FILE, TARGET_TOKENIZER_FILE
from src.utils import get_available_device, save_config, save_weights, get_logger

filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")


class Trainer:
    def __init__(self, config: Namespace, source_tokenizer: Tokenizer, target_tokenizer: Tokenizer) -> None:
        self.config = config
        self.device = get_available_device()
        self.logger = get_logger(__name__)

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        self.model = Seq2SeqTransformer(
            config,
            source_vocab_size=len(source_tokenizer.vocab),
            target_vocab_size=len(target_tokenizer.vocab)
        ).to(self.device)

        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.loss_fn = CrossEntropyLoss(
            weight=target_tokenizer.compute_class_weights(),
            ignore_index=Vocab.PAD.id,
        ).to(self.device)

    def fit(self, train_dl: DataLoader, val_dl: DataLoader) -> None:
        RUNS_DIR.mkdir(exist_ok=True, parents=True)
        model_dir = RUNS_DIR / f"v{len(listdir(RUNS_DIR)) + 1}"

        summary_writer_train = SummaryWriter(log_dir=model_dir / "train")
        summary_writer_eval = SummaryWriter(log_dir=model_dir / "eval")

        makedirs(summary_writer_train.log_dir, exist_ok=True)
        makedirs(summary_writer_eval.log_dir, exist_ok=True)

        save_config(self.config, model_dir / CONFIG_FILE.name)
        self.source_tokenizer.save(model_dir / SOURCE_TOKENIZER_FILE.name)
        self.target_tokenizer.save(model_dir / TARGET_TOKENIZER_FILE.name)

        best_score = 0
        best_score_metric = self.config.best_score_metric

        for epoch in range(1, self.config.epochs + 1):
            self.logger.info(f"[Epoch {epoch} / {self.config.epochs}]")
            metrics = self._train_for_epoch(train_dl)
            self.log_metrics(metrics, summary_writer_train, epoch=epoch)

            if epoch % self.config.eval_interval == 0:
                metrics = self.eval(val_dl)
                self.log_metrics(metrics, summary_writer_eval, epoch=epoch)

                score = metrics[best_score_metric].avg

                if score > best_score:
                    best_score = score
                    self.logger.info(f"Saving best weights with {best_score_metric}: {score:.3f}")
                    save_weights(model_dir / "weights_best.pth", self.model)

            if epoch % self.config.save_interval == 0:
                self.logger.info(f"Saving model weights at epoch: {epoch}")
                save_weights(model_dir / f"weights_{epoch}.pth", self.model)

    def _train_for_epoch(self, dataloader: DataLoader) -> Dict[str, AverageMeter]:
        self.model.train()
        metrics = {
            "loss": AverageMeter(),
        }

        for batch in tqdm(dataloader):
            self.optimizer.zero_grad()
            loss, *_ = self._forward_batch(batch)
            loss.backward()
            self.optimizer.step()
            metrics["loss"].update(loss.item())

        return metrics

    def eval(self, dataloader: DataLoader) -> Dict[str, AverageMeter]:
        self.model.eval()
        metrics = {
            "loss": AverageMeter(),
            **init_metrics(len(dataloader.dataset)),  # type: ignore
        }

        for batch in tqdm(dataloader):
            with torch.no_grad():
                loss, output, target = self._forward_batch(batch)

            metrics["loss"].update(loss.item())
            update_metrics(metrics, output, target, self.target_tokenizer)

        return metrics

    def _forward_batch(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        source = batch[0].to(self.device)
        target = batch[1].to(self.device)

        target_input = target[:, :-1]
        target_output = target[:, 1:]

        output = self.model(
            source=source,
            target=target_input,
            target_mask=self.model.transformer.generate_square_subsequent_mask(
                target_input.size(1), device=self.device, dtype=torch.bool
            ),
            source_padding_mask=source == Vocab.PAD.id,
            target_padding_mask=target_input == Vocab.PAD.id,
        )

        loss = self.loss_fn(output.transpose(1, 2), target_output)
        output = output.argmax(dim=-1)
        return loss, output, target_output

    def log_metrics(
            self,
            metrics: Dict[str, AverageMeter],
            summary_writer: Optional[SummaryWriter] = None,
            epoch: Optional[int] = None
    ) -> None:
        def log_scalar(tag: str, scalar: float):
            if epoch is not None and summary_writer is not None:
                summary_writer.add_scalar(tag=tag, scalar_value=scalar, global_step=epoch)

        message = "\n"
        for metric, value in metrics.items():
            if isinstance(value, AverageMeter):
                message += f"\t{metric}: {value.avg:.3f}\n"
                log_scalar(metric, value.avg)

            elif isinstance(value, AccuracyMeter):
                message += f"\t{metric}: {value.accuracy:.3f}\n"
                log_scalar(metric, value.accuracy)

            elif isinstance(value, ConfusionMatrixMeter):
                precision_metric = f"{metric}_precision"
                recall_metric = f"{metric}_recall"
                f1_metric = f"{metric}_f1"

                message += f"\t{precision_metric}: {value.precision:.3f}\n"
                message += f"\t{recall_metric}: {value.recall:.3f}\n"
                message += f"\t{f1_metric}: {value.f1:.3f}\n"

                log_scalar(precision_metric, value.precision)
                log_scalar(recall_metric, value.recall)
                log_scalar(f1_metric, value.f1)

            else:
                raise ValueError(f"Unknown metric type {type(value)}")

        self.logger.info(message)
