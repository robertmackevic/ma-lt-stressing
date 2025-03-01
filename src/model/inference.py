from pathlib import Path
from typing import Optional

import torch

from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab, STRESS_LETTERS, STRESS_MARKS
from src.model.transformer import Seq2SeqTransformer
from src.paths import CONFIG_FILE, SOURCE_TOKENIZER_FILE, TARGET_TOKENIZER_FILE
from src.utils import load_config, get_available_device, load_weights, seed_everything


class Inference:
    def __init__(self, model_dir: Path, weights_filename: str) -> None:
        self.config = load_config(model_dir / CONFIG_FILE.name)
        self.device = get_available_device()
        self.source_tokenizer = Tokenizer.init_from_file(model_dir / SOURCE_TOKENIZER_FILE.name)
        self.target_tokenizer = Tokenizer.init_from_file(model_dir / TARGET_TOKENIZER_FILE.name)

        self.model = load_weights(
            filepath=model_dir / weights_filename,
            model=Seq2SeqTransformer(
                config=self.config,
                source_vocab_size=len(self.source_tokenizer.vocab),
                target_vocab_size=len(self.target_tokenizer.vocab),
            ),
        ).to(self.device)
        self.model.eval()

    def by_greedy_decoding_with_rules(self, text: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            seed_everything(seed)

        self.model.eval()
        stressed_text = ""
        source = self.source_tokenizer.encode(text).unsqueeze(0).to(self.device)
        context_ids = [Vocab.SOS.id]
        is_word_stressed = False

        for char in text:
            if char.isspace():
                is_word_stressed = False
                token_id = self.target_tokenizer.vocab.token_to_id[char]

            else:
                token_id = Vocab.UNK.id

            stressed_text += char
            context_ids.append(token_id)

            if not is_word_stressed and char in STRESS_LETTERS:
                context = torch.tensor([context_ids]).to(self.device)

                with torch.no_grad():
                    output = self.model(source, context)

                output_id = output.topk(1)[1].view(-1)[-1].item()
                output_token = self.target_tokenizer.vocab.id_to_token.get(output_id)

                if output_token is not None and output_token in STRESS_MARKS:
                    context_ids.append(output_id)
                    stressed_text += output_token
                    is_word_stressed = True

        return stressed_text
