import re

from tqdm import tqdm

from src.data.processing import normalize_text
from src.data.tokenizer import Tokenizer
from src.data.vocab import STRESS_MARKS, Vocab, remove_character_before_stress_marks
from src.metrics import init_metrics, update_metrics, compile_metrics_message
from src.paths import DATA_DIR
from src.utils import get_logger


def remove_following_punctuations(text: str) -> str:
    return re.sub(rf"[^a-zA-ZąčęėįšųūžĄČĘĖĮŠŲŪŽ{STRESS_MARKS}]+$", "", text)


def run() -> None:
    logger = get_logger()

    with (DATA_DIR / "val.txt").open("r", encoding="utf-8") as file:
        val_texts = [normalize_text(remove_following_punctuations(line)) for line in file.readlines()]

    with (DATA_DIR / "vdu.txt").open("r", encoding="utf-8") as file:
        vdu_texts = [normalize_text(line) for line in file.readlines()]

    assert len(val_texts) == len(vdu_texts)

    metrics = init_metrics(len(val_texts))
    tokenizer = Tokenizer(Vocab.init_target_vocab(val_texts))
    error_list = []

    for i in tqdm(range(len(val_texts))):
        target = tokenizer.encode(remove_character_before_stress_marks(val_texts[i])).unsqueeze(0)
        output = tokenizer.encode(remove_character_before_stress_marks(vdu_texts[i])).unsqueeze(0)
        update_metrics(metrics, output, target, tokenizer)

        if val_texts[i] != vdu_texts[i]:
            error_list.append((val_texts[i], vdu_texts[i]))

    message = "\n"
    for item in error_list:
        message += f"VAL: {item[0]}\nVDU: {item[1]}\n"

    logger.info(message)
    logger.info(compile_metrics_message(metrics))


if __name__ == "__main__":
    run()
