from argparse import Namespace
from os import listdir, makedirs
from typing import Dict, Optional
from warnings import filterwarnings

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab
from src.model.transformer import Seq2SeqTransformer
from src.paths import RUNS_DIR, CONFIG_FILE, SOURCE_TOKENIZER_FILE, TARGET_TOKENIZER_FILE
from src.utils import get_available_device, save_config, save_weights, get_logger

filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")


class AverageMeter:
    def __init__(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float) -> None:
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count


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
        self.loss_fn = CrossEntropyLoss().to(self.device)

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
        pad_id = Vocab.SPECIAL_TO_ID[Vocab.PAD_TOKEN]
        metrics = {
            "loss": AverageMeter(),
        }

        for batch in tqdm(dataloader):
            self.optimizer.zero_grad()
            source = batch[0].to(self.device)
            target = batch[1].to(self.device)

            target_input = target[:, :-1]
            target_output = target[:, 1:]

            source_padding_mask = source == pad_id

            output = self.model(
                source, target_input,
                target_mask=self.model.transformer.generate_square_subsequent_mask(
                    target_input.size(1), device=self.device, dtype=torch.bool
                ),
                source_padding_mask=source_padding_mask,
                target_padding_mask=target_input == pad_id,
                memory_padding_mask=source_padding_mask,
            )

            loss = self.loss_fn(output.transpose(1, 2), target_output)
            loss.backward()
            self.optimizer.step()
            metrics["loss"].update(loss.item())

        return metrics

    def eval(self, dataloader: DataLoader) -> Dict[str, AverageMeter]:
        self.model.eval()
        pad_id = Vocab.SPECIAL_TO_ID[Vocab.PAD_TOKEN]
        metrics = {
            "loss": AverageMeter(),
        }

        for batch in tqdm(dataloader):
            source = batch[0].to(self.device)
            target = batch[1].to(self.device)

            target_input = target[:, :-1]
            target_output = target[:, 1:]

            source_padding_mask = source == pad_id

            with torch.no_grad():
                output = self.model(
                    source, target_input,
                    target_mask=self.model.transformer.generate_square_subsequent_mask(
                        target_input.size(1), device=self.device, dtype=torch.bool
                    ),
                    source_padding_mask=source == pad_id,
                    target_padding_mask=target_input == pad_id,
                    memory_padding_mask=source_padding_mask,
                )

            loss = self.loss_fn(output.transpose(1, 2), target_output)
            metrics["loss"].update(loss.item())

        return metrics

    def log_metrics(
            self,
            metrics: Dict[str, AverageMeter],
            summary_writer: Optional[SummaryWriter] = None,
            epoch: Optional[int] = None
    ) -> None:
        message = "\n"
        for metric, value in metrics.items():
            message += f"\t{metric}: {value.avg:.3f}\n"

            if epoch is not None and summary_writer is not None:
                summary_writer.add_scalar(tag=metric, scalar_value=value.avg, global_step=epoch)

        self.logger.info(message)
