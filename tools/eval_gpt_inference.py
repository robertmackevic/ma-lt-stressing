from argparse import Namespace, ArgumentParser
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.data.processing import load_texts
from src.data.vocab import remove_stress_marks, GRAVE_ACCENT, ACUTE_ACCENT, TILDE_ACCENT, STRESS_LETTERS
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

    correct = 0

    try:
        for target_text in tqdm(texts):
            source_text = remove_stress_marks(target_text)

            completion = client.chat.completions.create(
                model=gpt_version,
                messages=[SYSTEM_MESSAGE, {"role": "user", "content": source_text}]
            )

            output_text = completion.choices[0].message.content

            if output_text == target_text:
                correct += 1

    except KeyboardInterrupt:
        print("Evaluation terminated.")

    finally:
        print(f"Accuracy: {correct / len(texts):.3f}")


if __name__ == "__main__":
    run(**vars(parse_args()))
