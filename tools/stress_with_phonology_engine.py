from argparse import Namespace, ArgumentParser
from pathlib import Path

from phonology_engine import PhonologyEngine

from src.data.processing import normalize_text, remove_stress_marks
from src.utils import get_logger


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--file",
        type=Path,
        required=True,
        dest="filepath",
        help="Input text file",
    )

    return parser.parse_args()


def run(filepath: Path) -> None:
    logger = get_logger()
    pe = PhonologyEngine()

    outpath = filepath.with_name("pe.txt")
    logger.info(f"Reading: {filepath}")
    logger.info(f"Writing: {outpath}")

    with filepath.open("r", encoding="utf-8") as fin, outpath.open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(
                pe.process_and_collapse(
                    normalize_text(remove_stress_marks(line)), "utf8_stressed_word", normalize=True
                ).strip() + "\n")


if __name__ == "__main__":
    run(**vars(parse_args()))
