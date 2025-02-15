from src.data.dataset import StressingDataset
from src.data.processing import load_texts
from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab
from src.paths import DATA_DIR
from src.utils import get_logger, load_config, seed_everything


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

    print(train_dataset[0])
    print(val_dataset[0])


if __name__ == "__main__":
    run()
