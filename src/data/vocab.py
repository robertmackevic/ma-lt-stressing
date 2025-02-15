import re
from typing import Dict, Self, List

GRAVE_ACCENT = "\u0300"
ACUTE_ACCENT = "\u0301"
TILDE_ACCENT = "\u0303"
STRESS_MARKS = GRAVE_ACCENT + ACUTE_ACCENT + TILDE_ACCENT
STRESS_LETTERS = "aąeęėiįylmnoruųū"


class Vocab:
    PAD_TOKEN = "*"
    UNK_TOKEN = "#"
    SOS_TOKEN = "<"
    EOS_TOKEN = ">"
    SPECIAL_TO_ID = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        SOS_TOKEN: 2,
        EOS_TOKEN: 3,
    }

    def __init__(self, token_to_id: Dict[str, int], token_freq: Dict[str, int]) -> None:
        self.token_freq = token_freq
        self.token_to_id = token_to_id
        self.id_to_token = {value: key for key, value in token_to_id.items()}

    @classmethod
    def init_from_texts(cls, texts: List[str]) -> Self:
        token_to_id = {**cls.SPECIAL_TO_ID}
        token_freq = {cls.PAD_TOKEN: 0, cls.UNK_TOKEN: 0, cls.SOS_TOKEN: 0, cls.EOS_TOKEN: 0}

        for text in texts:
            for token in text:
                if token not in token_to_id:
                    token_id = len(token_to_id)
                    token_to_id[token] = token_id

                if token not in token_freq:
                    token_freq[token] = 0

                token_freq[token] += 1

        return cls(token_to_id, token_freq)

    @classmethod
    def init_source_vocab(cls, texts: List[str]) -> Self:
        texts = [re.sub(rf"[{re.escape(STRESS_MARKS)}]", "", text) for text in texts]
        return cls.init_from_texts(texts)

    @classmethod
    def init_target_vocab(cls, texts: List[str]) -> Self:
        texts = [re.sub(rf"[^{re.escape(STRESS_MARKS)}\s]", cls.UNK_TOKEN, text) for text in texts]
        return cls.init_from_texts(texts)

    def __len__(self) -> int:
        return len(self.token_to_id)
