from src.data.vocab import Vocab


class Tokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab
