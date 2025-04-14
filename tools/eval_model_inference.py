from argparse import Namespace, ArgumentParser

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import StressingDataset
from src.data.processing import load_texts
from src.data.sampler import BucketSampler
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
    parser.add_argument(
        "--beams",
        type=int,
        required=False,
        default=None,
        help="Number of beams to use during beam search, if not specified, then greedy decoding will be used",
    )
    return parser.parse_args()


def run(version: str, weights: str, subset: str, beams: int) -> None:
    logger = get_logger()

    if beams is not None and beams <= 0:
        raise ValueError("Number of beams must be greater than 0")

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

            if beams is None:
                output = inference.tensor_greedy_decoding_with_rules(source, seed=inference.config.seed)

            else:
                output = inference.tensor_beam_search_decoding(source, beams, seed=inference.config.seed)

            update_metrics(metrics, output, target, inference.target_tokenizer)

        logger.info(compile_metrics_message(metrics))

    except KeyboardInterrupt:
        logger.info("Evaluation terminated.")


if __name__ == "__main__":
    log_systems_info()
    run(**vars(parse_args()))
