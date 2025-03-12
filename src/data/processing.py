import re
from pathlib import Path
from typing import List, Optional, Tuple

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from src.data.vocab import STRESS_LETTERS, STRESS_MARKS, Vocab


def load_texts(filepath: Path, clean: bool = True) -> Optional[List[str]]:
    with open(filepath, "r", encoding="utf-8") as file:
        texts = [line for line in file.readlines()]
        return clean_texts(texts) if clean else texts


def clean_texts(texts: List[str]) -> List[str]:
    cleaned_texts = []
    for text in texts:
        text = normalize_text(text)

        for segment in split_text_into_segments_by_length(text):
            filtered_text = filter_punctuations(segment)

            if len(filtered_text) > 0 and is_valid_stressing(filtered_text):
                cleaned_texts.append(segment)

    return cleaned_texts


def normalize_text(text: str) -> str:
    return (
        re.sub(r"\s+", " ", re.sub(r"[‐‑–—―]", "-", text))
        .replace("…", "...")
        .replace("\ufeff", "")
        .replace("'", "")
        .replace("̇", "")
        .strip()
        .lower()
    )


def split_text_into_segments_by_length(text: str, max_length: int = 200) -> List[str]:
    segments = []
    text_length = len(text)
    split_characters = ",.?!:;\n"
    start = 0

    while start < text_length:
        # If the remaining text is shorter than max_sequence_length, add it and exit
        if start + max_length >= text_length:
            segments.append(text[start:].strip())
            break

        # Find the last occurrence of any punctuation in candidate
        candidate = text[start:start + max_length]
        split_index = max(candidate.rfind(character) for character in split_characters)

        # If no punctuation found, try to split on the last space
        if split_index == -1:
            split_index = candidate.rfind(" ")

        # If no splitting character is found, force split at max_length
        end = start + (split_index + 1 if split_index != -1 else max_length)
        segments.append(text[start:end].strip())
        start = end

    return segments


def filter_punctuations(text: str) -> str:
    return normalize_text("".join(
        character
        if character.isalpha() or character.isspace() or character in STRESS_MARKS
        else " " for character in text
    ))


def is_valid_stressing(text: str) -> bool:
    """ Input is normalized text without non-alphabetic characters except for stress marks """
    return all(
        sum(1 for character in word if character in STRESS_MARKS) == 1
        and
        all(word[i - 1] in STRESS_LETTERS for i in range(len(word)) if word[i] in STRESS_MARKS)
        for word in text.split()
    )


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    source, target = zip(*batch)
    padded_source = pad_sequence(source, batch_first=True, padding_value=Vocab.PAD.id)
    padded_target = pad_sequence(target, batch_first=True, padding_value=Vocab.PAD.id)
    return padded_source, padded_target
