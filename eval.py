from argparse import Namespace, ArgumentParser

from torch.utils.data import DataLoader

from src.data.dataset import StressingDataset
from src.data.processing import load_texts, collate_fn
from src.data.sampler import BucketSampler
from src.data.tokenizer import Tokenizer
from src.paths import RUNS_DIR, CONFIG_FILE, DATA_DIR, SOURCE_TOKENIZER_FILE, TARGET_TOKENIZER_FILE
from src.trainer import Trainer
from src.utils import get_logger, load_config, seed_everything, load_weights, count_parameters, log_systems_info


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-v", "--version", type=str, required=True, help="v1, v2, v3, etc.")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Name of the .pth file")
    return parser.parse_args()


def run(version: str, weights: str) -> None:
    logger = get_logger()
    logger.info("Preparing the data...")
    model_dir = RUNS_DIR / version

    config = load_config(model_dir / CONFIG_FILE.name)
    source_tokenizer = Tokenizer.init_from_file(model_dir / SOURCE_TOKENIZER_FILE.name)
    target_tokenizer = Tokenizer.init_from_file(model_dir / TARGET_TOKENIZER_FILE.name)

    seed_everything(config.seed)

    trainer = Trainer(config, source_tokenizer, target_tokenizer)
    trainer.model = load_weights(filepath=model_dir / weights, model=trainer.model)
    logger.info(f"Number of trainable parameters: {count_parameters(trainer.model, trainable=True):,}")

    for subset in ["train", "val"]:
        texts = load_texts(DATA_DIR / f"{subset}.txt")
        dataset = StressingDataset(texts, source_tokenizer, target_tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            sampler=BucketSampler(dataset, batch_size=config.batch_size),
        )

        try:
            logger.info(f"Evaluating on {subset} data...")
            trainer.log_metrics(trainer.eval(dataloader))
        except KeyboardInterrupt:
            logger.info("Evaluation terminated.")


if __name__ == "__main__":
    log_systems_info()
    run(**vars(parse_args()))
