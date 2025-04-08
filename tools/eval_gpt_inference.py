from argparse import Namespace, ArgumentParser
from typing import Optional

import torch
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.data.const import GRAVE_ACCENT, ACUTE_ACCENT, TILDE_ACCENT, STRESS_LETTERS, EOS, PAD
from src.data.processing import load_texts, remove_character_before_stress_marks, remove_stress_marks
from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab
from src.metrics import init_metrics, update_metrics, compile_metrics_message
from src.paths import DATA_DIR, ENV_FILE

SYSTEM_MESSAGE = {
    "role": "system",
    "content": f"""
        You will be given a Lithuanian text.
        Your role is to stress the text using Lithuanian text stressing rules (kirčiavimo taisyklės).
        Return only the stressed text and nothing else - DO NOT TALK BACK.
        Stressed text is a combination of the original text with stress marks added in the correct places.
        Each word MUST have only a single stress mark added.
        To add a stress mark, place the appropriate character after the letter that should be stressed.
        Stress marks are these 3 characters: {GRAVE_ACCENT}{ACUTE_ACCENT}{TILDE_ACCENT}
        They can be placed after one of these letters: {STRESS_LETTERS}
    """
}


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--gpt-version",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        required=False,
    )
    return parser.parse_args()


def run(gpt_version: str, sample_size: Optional[int]) -> None:
    load_dotenv(ENV_FILE, override=True)

    print("Preparing the dataset...")
    texts = load_texts(DATA_DIR / "val.txt")
    if sample_size is not None:
        texts = texts[:sample_size]

    print(f"Running inference with GPT version: {gpt_version}")
    client = OpenAI()

    metrics = init_metrics(len(texts))
    tokenizer = Tokenizer(Vocab.init_target_vocab(texts))
    error_list = []

    try:
        for target_text in tqdm(texts):
            source_text = remove_stress_marks(target_text)

            completion = client.chat.completions.create(
                model=gpt_version,
                messages=[SYSTEM_MESSAGE, {"role": "user", "content": source_text}]
            )

            output_text = completion.choices[0].message.content

            target = tokenizer.encode(remove_character_before_stress_marks(target_text)).unsqueeze(0)
            output = tokenizer.encode(remove_character_before_stress_marks(output_text)).unsqueeze(0)

            # In case of inadequate inference, pad the sequences to compensate for missing or excess tokens
            if output.size(1) < target.size(1):
                padding = torch.full((1, target.size(1) - output.size(1)), EOS.id, device=output.device)
                output = torch.cat([output, padding], dim=1)

            elif target.size(1) < output.size(1):
                padding = torch.full((1, output.size(1) - target.size(1)), PAD.id, device=target.device)
                target = torch.cat([target, padding], dim=1)

            update_metrics(metrics, output, target, tokenizer)

            if target_text != output_text:
                error_list.append((target_text, output_text))

    except KeyboardInterrupt:
        print("Evaluation terminated.")

    finally:
        message = "\n"
        for item in error_list:
            message += f"VAL: {item[0]}\nGPT: {item[1]}\n"

        print(message)
        print(compile_metrics_message(metrics))


if __name__ == "__main__":
    run(**vars(parse_args()))
