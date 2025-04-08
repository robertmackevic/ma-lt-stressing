import re
from typing import Dict, Self, List

from src.data.const import STRESS_MARKS, PAD, UNK, SOS, EOS
from src.data.processing import remove_stress_marks, remove_character_before_stress_marks


class Vocab:
    SPECIAL_TO_ID = {
        PAD.token: PAD.id,
        UNK.token: UNK.id,
        SOS.token: SOS.id,
        EOS.token: EOS.id,
    }

    def __init__(self, token_to_id: Dict[str, int], token_freq: Dict[str, int]) -> None:
        self.token_freq = token_freq
        self.token_to_id = token_to_id
        self.id_to_token = {value: key for key, value in token_to_id.items()}

        self.stress_token_to_id = {
            token: token_to_id[token] for token in STRESS_MARKS if token in token_to_id
        }

    @classmethod
    def init_from_texts(cls, texts: List[str]) -> Self:
        token_to_id = {**cls.SPECIAL_TO_ID}
        token_freq = {
            token: len(texts) if token in (SOS.token, EOS.token) else 0
            for token in token_to_id.keys()
        }

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
        return cls.init_from_texts([remove_stress_marks(text) for text in texts])

    @classmethod
    def init_target_vocab(cls, texts: List[str]) -> Self:
        return cls.init_from_texts([
            # Replace non-stress mark characters with UNK tokens
            re.sub(
                rf"[^{re.escape(STRESS_MARKS)}]",
                UNK.token,
                remove_character_before_stress_marks(text)
            )
            for text in texts
        ])

    def __len__(self) -> int:
        return len(self.token_to_id)
