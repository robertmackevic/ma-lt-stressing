from argparse import Namespace
from os import listdir, makedirs

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from src.data.tokenizer import Tokenizer
from src.paths import RUNS_DIR, CONFIG_FILE, SOURCE_TOKENIZER_FILE, TARGET_TOKENIZER_FILE
from src.utils import get_available_device, save_config, get_logger


class Trainer:
    def __init__(self, config: Namespace, source_tokenizer: Tokenizer, target_tokenizer: Tokenizer) -> None:
        self.config = config
        self.device = get_available_device()
        self.logger = get_logger(__name__)

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

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

        for epoch in range(1, self.config.epochs + 1):
            self.logger.info(f"[Epoch {epoch} / {self.config.epochs}]")
