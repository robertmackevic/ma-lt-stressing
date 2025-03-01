from argparse import Namespace, ArgumentParser
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from src.data.processing import split_text_into_segments_by_length, normalize_text, filter_punctuations
from src.model.inference import Inference
from src.paths import RUNS_DIR
from src.utils import get_logger, log_systems_info


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
        "-f", "--file",
        type=Path,
        required=True,
        dest="filepath",
        help="Input text file",
    )
    return parser.parse_args()


def run(version: str, weights: str, filepath: Path) -> None:
    logger = get_logger()
    texts = []

    with filepath.open("r", encoding="utf-8") as file:
        for text in file.readlines():
            texts.extend(split_text_into_segments_by_length(normalize_text(text), max_length=150))

    words = [word for text in texts for word in filter_punctuations(text).split()]
    word_counts_per_text = [len(filter_punctuations(text).split()) for text in texts]
    char_counts_per_text = [len(text) for text in texts]
    char_counts_per_word = [len(word) for word in words]
    original_text = " ".join(texts)

    logger.info(f"""
    Analyzing the data...
        Text length:                {len(original_text):>10}
        Number of sequences:        {len(texts):>10}
        Number of words:            {len(words):>10}
        Number of unique words:     {len(set(words)):>10}
        Mean words per text:        {mean(word_counts_per_text) if len(word_counts_per_text) > 0 else 0:>10.2f}
        Number of characters:       {sum(char_counts_per_text):>10}
        Mean characters per text:   {mean(char_counts_per_text) if len(char_counts_per_text) > 0 else 0:>10.2f}
        Mean characters per word:   {mean(char_counts_per_word) if len(char_counts_per_word) > 0 else 0:>10.2f}
    """)

    logger.info("Preparing the model...")
    inference = Inference(
        model_dir=RUNS_DIR / version,
        weights_filename=weights,
    )

    logger.info("Stressing text...")
    logger.info(f"""OUTPUT:
    {" ".join(inference.text_greedy_decoding_with_rules(text, seed=inference.config.seed) for text in tqdm(texts))}
    """)


if __name__ == "__main__":
    log_systems_info()
    run(**vars(parse_args()))
