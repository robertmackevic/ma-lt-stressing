from torch.utils.data import DataLoader

from src.data.dataset import StressingDataset
from src.data.processing import load_texts, collate_fn
from src.data.sampler import BucketSampler
from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab
from src.paths import DATA_DIR
from src.trainer import Trainer
from src.utils import get_logger, load_config, seed_everything, count_parameters


def run() -> None:
    logger = get_logger()
    config = load_config()
    seed_everything(config.seed)

    logger.info("Preparing the data...")
    train_texts = load_texts(DATA_DIR / "train.txt")
    val_texts = load_texts(DATA_DIR / "val.txt")

    source_tokenizer = Tokenizer(Vocab.init_source_vocab(train_texts))
    target_tokenizer = Tokenizer(Vocab.init_target_vocab(train_texts))

    train_dataset = StressingDataset(train_texts, source_tokenizer, target_tokenizer)
    val_dataset = StressingDataset(val_texts, source_tokenizer, target_tokenizer)

    train_dl = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        sampler=BucketSampler(train_dataset, batch_size=config.batch_size),
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        sampler=BucketSampler(val_dataset, batch_size=config.batch_size),
    )

    trainer = Trainer(config, source_tokenizer, target_tokenizer)
    logger.info(f"Number of trainable parameters: {count_parameters(trainer.model):,}")

    try:
        logger.info("Starting training...")
        trainer.fit(train_dl, val_dl)
    except KeyboardInterrupt:
        logger.info("Training terminated.")


if __name__ == "__main__":
    run()
