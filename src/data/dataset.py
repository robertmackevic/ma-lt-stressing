from typing import List, Tuple

from torch import Tensor
from torch.utils.data import Dataset

from src.data.processing import remove_stress_marks, remove_character_before_stress_marks
from src.data.tokenizer import Tokenizer


class StressingDataset(Dataset):
    def __init__(
            self,
            texts: List[str],
            source_tokenizer: Tokenizer,
            target_tokenizer: Tokenizer,
    ) -> None:
        super().__init__()
        self.texts = texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        text = self.texts[index]
        source = self.source_tokenizer.encode(remove_stress_marks(text))
        target = self.target_tokenizer.encode(remove_character_before_stress_marks(text))

        assert len(source) == len(target)
        return source, target
