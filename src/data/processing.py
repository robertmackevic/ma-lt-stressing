import re
from pathlib import Path
from typing import List, Optional

GRAVE = "\u0300"
ACUTE = "\u0301"
TILDE = "\u0303"
STRESS_MARKS = GRAVE + ACUTE + TILDE


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


def clean_texts(texts: List[str]) -> List[str]:
    cleaned_texts = []
    for text in texts:
        text = normalize_text(text)

        for segment in split_text_into_segments_by_length(text):
            if len(remove_non_alphabetic_characters(segment)) > 0:
                cleaned_texts.append(segment)

    return cleaned_texts


def load_texts(filepath: Path, clean: bool = True) -> Optional[List[str]]:
    with open(filepath, "r", encoding="utf-8") as file:
        texts = [line for line in file.readlines()]
        return clean_texts(texts) if clean else texts


def remove_non_alphabetic_characters(text: str) -> str:
    return " ".join("".join(
        character
        if character.isalpha() or character.isspace() or character in STRESS_MARKS
        else "" for character in text
    ).split())


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
