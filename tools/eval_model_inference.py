from argparse import Namespace, ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import StressingDataset
from src.data.processing import load_texts
from src.data.sampler import BucketSampler
from src.data.vocab import Vocab
from src.metrics import init_metrics, update_metrics, compile_metrics_message
from src.model.inference import Inference
from src.paths import RUNS_DIR, DATA_DIR
from src.utils import log_systems_info, get_logger


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--version",
        type=str,
        required=True,
        help="v1, v2, v3, etc.",
    )
    parser.add_argument(
        "-w", "--weights",
        type=str,
        required=True,
        help="Name of the .pth file",
    )
    parser.add_argument(
        "-s", "--subset",
        type=str,
        required=False,
        default="val",
        help="Name of the .pth file",
    )
    return parser.parse_args()


def run(version: str, weights: str, subset: str) -> None:
    logger = get_logger()

    logger.info("Preparing the model...")
    inference = Inference(
        model_dir=RUNS_DIR / version,
        weights_filename=weights,
    )

    logger.info("Preparing the dataset...")
    texts = load_texts(DATA_DIR / f"{subset}.txt")
    dataset = StressingDataset(texts, inference.source_tokenizer, inference.target_tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=BucketSampler(dataset, batch_size=1),
    )

    try:
        logger.info(f"Evaluating on {subset} data...")
        metrics = init_metrics(len(dataset))

        for batch in tqdm(dataloader):
            source = batch[0].to(inference.device)
            target = batch[1].to(inference.device)
            output = inference.tensor_greedy_decoding_with_rules(source, seed=inference.config.seed)

            # In case of inadequate inference, pad the sequences to compensate for missing or excess tokens
            if output.size(1) < target.size(1):
                padding = torch.full((1, target.size(1) - output.size(1)), Vocab.EOS.id, device=output.device)
                output = torch.cat([output, padding], dim=1)

            elif target.size(1) < output.size(1):
                padding = torch.full((1, output.size(1) - target.size(1)), Vocab.PAD.id, device=target.device)
                target = torch.cat([target, padding], dim=1)

            update_metrics(metrics, output, target, inference.target_tokenizer)

        logger.info(compile_metrics_message(metrics))

    except KeyboardInterrupt:
        logger.info("Evaluation terminated.")


if __name__ == "__main__":
    log_systems_info()
    run(**vars(parse_args()))
