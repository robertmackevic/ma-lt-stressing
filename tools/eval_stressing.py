import re
from argparse import Namespace, ArgumentParser
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from src.data.const import STRESS_MARKS
from src.data.processing import (
    normalize_text,
    split_text_into_segments_by_length,
    filter_punctuations,
    remove_character_before_stress_marks,
)
from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab
from src.metrics import init_metrics, update_metrics, compile_metrics_message
from src.paths import DATA_DIR
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
    parser.add_argument(
        "--exclude-punctuations",
        action="store_true",
        required=False,
        default=False,
        help="PhonologyEngine excludes punctuations from output, "
             "so for evaluation punctuations need to be excluded from the GT as well.",
    )
    return parser.parse_args()


def remove_following_punctuations(text: str) -> str:
    return re.sub(rf"[^a-zA-ZąčęėįšųūžĄČĘĖĮŠŲŪŽ{STRESS_MARKS}]+$", "", text)


def clean_texts(texts: List[str]) -> List[str]:
    cleaned = []
    for text in texts:
        text = normalize_text(text)
        for segment in split_text_into_segments_by_length(text):
            filtered = filter_punctuations(segment)
            if len(filtered) > 0:
                cleaned.append(remove_following_punctuations(segment))
    return cleaned


def has_stress(word: str) -> bool:
    return any(ch in STRESS_MARKS for ch in word)


def words_from_text(text: str) -> List[str]:
    return normalize_text(filter_punctuations(text)).split()


def format_counter(title: str, counter: Counter, *, min_count: int = 1) -> str:
    if not counter:
        return ""

    lines = [f"\n{title}\n"]
    for (g, p), c in counter.most_common():
        if c < min_count:
            break
        lines.append(f"    {c:>6}x  {g}  ->  {p}")
    return "\n".join(lines) + "\n"


def compute_word_stats(
        gt_texts: List[str],
        pred_texts: List[str],
) -> Tuple[dict, Counter, Counter]:
    total_words = 0
    correct_words = 0
    pred_unstressed = 0
    incorrect_stressed = 0
    mismatched_lines = 0

    # Count unique examples with multiplicity
    incorrect_counter: Counter = Counter()  # (gt_word, pred_word) -> count
    unstressed_counter: Counter = Counter()  # (gt_word, pred_word) -> count

    for gt, pr in zip(gt_texts, pred_texts):
        gt_w = words_from_text(gt)
        pr_w = words_from_text(pr)

        if len(gt_w) != len(pr_w):
            mismatched_lines += 1

        n = min(len(gt_w), len(pr_w))
        for j in range(n):
            g = gt_w[j]
            p = pr_w[j]
            total_words += 1

            if p == g:
                correct_words += 1
                continue

            # PRED has no stress mark
            if not has_stress(p):
                pred_unstressed += 1
                # Only count as an "unstressed example" if GT *should* be stressed
                if has_stress(g):
                    unstressed_counter[(g, p)] += 1
                continue

            # PRED has stress but differs from GT
            incorrect_stressed += 1
            incorrect_counter[(g, p)] += 1

    stats = {
        "total_words": total_words,
        "correct_words": correct_words,
        "pred_unstressed": pred_unstressed,
        "incorrect_stressed": incorrect_stressed,
        "mismatched_lines_wordcount": mismatched_lines,
        "unique_incorrect_examples": len(incorrect_counter),
        "unique_unstressed_examples": len(unstressed_counter),
    }
    return stats, incorrect_counter, unstressed_counter


def run(filepath: Path, exclude_punctuations: bool) -> None:
    logger = get_logger()

    with (DATA_DIR / "val.txt").open("r", encoding="utf-8") as file:
        val_texts = clean_texts(file.readlines())
        if exclude_punctuations:
            val_texts = [normalize_text(filter_punctuations(text)) for text in val_texts]

    with filepath.open("r", encoding="utf-8") as file:
        pred_texts = [normalize_text(line) for line in file.readlines()]

    assert len(val_texts) == len(pred_texts)

    metrics = init_metrics(len(val_texts))
    tokenizer = Tokenizer(Vocab.init_target_vocab(val_texts))
    error_list = []

    for i in tqdm(range(len(val_texts))):
        target = tokenizer.encode(remove_character_before_stress_marks(val_texts[i])).unsqueeze(0)
        output = tokenizer.encode(remove_character_before_stress_marks(pred_texts[i])).unsqueeze(0)
        try:
            update_metrics(metrics, output, target, tokenizer)
        except RuntimeError:
            logger.info(f"Skipping entry {i} due to error.")

        if val_texts[i] != pred_texts[i]:
            error_list.append((val_texts[i], pred_texts[i]))

    word_stats, incorrect_counter, unstressed_counter = compute_word_stats(val_texts, pred_texts)
    logger.info(compile_metrics_message(metrics))

    logger.info(
        "\n"
        "Word-level analysis:\n"
        f"    Total words:                                {word_stats['total_words']}\n"
        f"    Words stressed correctly (exact match):      {word_stats['correct_words']}\n"
        f"    Words left unstressed in prediction:         {word_stats['pred_unstressed']}\n"
        f"    Words stressed incorrectly (excl unstressed): {word_stats['incorrect_stressed']}\n"
        f"    Lines with word-count mismatch (GT vs PRED): {word_stats['mismatched_lines_wordcount']}\n"
        f"    Unique incorrect examples:                   {word_stats['unique_incorrect_examples']}\n"
        f"    Unique unstressed examples:                  {word_stats['unique_unstressed_examples']}\n"
    )

    msg = format_counter("Examples: Incorrect stressed words (GT -> PRED)", incorrect_counter)
    if msg:
        logger.info(msg)

    msg = format_counter("Examples: Predicted unstressed where GT is stressed (GT -> PRED)", unstressed_counter)
    if msg:
        logger.info(msg)


if __name__ == "__main__":
    run(**vars(parse_args()))
