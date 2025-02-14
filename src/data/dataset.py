from argparse import Namespace
from typing import List

from torch.utils.data import Dataset

from src.data.tokenizer import Tokenizer


class StressingDataset(Dataset):
    def __init__(
            self,
            config: Namespace,
            texts: List[str],
            source_tokenizer: Tokenizer,
            target_tokenizer: Tokenizer,
    ) -> None:
        super().__init__()
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
