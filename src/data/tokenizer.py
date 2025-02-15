import json
from pathlib import Path
from typing import Self

import torch

from src.data.vocab import Vocab
from src.paths import TOKENIZER_FILE


class Tokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    @classmethod
    def init_from_file(cls, filepath: Path = TOKENIZER_FILE) -> Self:
        with filepath.open("r") as file:
            tokenizer_data = json.load(file)
            return cls(Vocab(tokenizer_data["token_to_id"], tokenizer_data["token_freq"]))

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([Vocab.SPECIAL_TO_ID[Vocab.SOS_TOKEN]] + [
            self.vocab.token_to_id.get(token, Vocab.SPECIAL_TO_ID[Vocab.UNK_TOKEN])
            for token in text
        ] + [Vocab.SPECIAL_TO_ID[Vocab.EOS_TOKEN]])

    def decode(self, tensor: torch.Tensor) -> str:
        return "".join(
            self.vocab.id_to_token.get(token_id, Vocab.UNK_TOKEN)
            for token_id in tensor.tolist()
        ).replace(Vocab.PAD_TOKEN, "")[1:-1]

    def save(self, filepath: Path) -> None:
        with filepath.open("w") as file:
            json.dump({
                "token_to_id": self.vocab.token_to_id,
                "token_freq": self.vocab.token_freq,
            }, file, indent=4)  # type: ignore
