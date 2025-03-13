from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from src.data.tokenizer import Tokenizer
from src.data.vocab import Vocab, STRESS_LETTERS, remove_stress_marks
from src.model.transformer import Seq2SeqTransformer
from src.paths import CONFIG_FILE, SOURCE_TOKENIZER_FILE, TARGET_TOKENIZER_FILE
from src.utils import load_config, get_available_device, load_weights, seed_everything


class Inference:
    def __init__(self, model_dir: Path, weights_filename: str) -> None:
        self.config = load_config(model_dir / CONFIG_FILE.name)
        self.device = get_available_device()
        self.source_tokenizer = Tokenizer.init_from_file(model_dir / SOURCE_TOKENIZER_FILE.name)
        self.target_tokenizer = Tokenizer.init_from_file(model_dir / TARGET_TOKENIZER_FILE.name)

        self.stress_letter_ids = [self.source_tokenizer.vocab.token_to_id[letter] for letter in STRESS_LETTERS]

        self.model = load_weights(
            filepath=model_dir / weights_filename,
            model=Seq2SeqTransformer(
                config=self.config,
                source_vocab_size=len(self.source_tokenizer.vocab),
                target_vocab_size=len(self.target_tokenizer.vocab),
            ),
        ).to(self.device)
        self.model.eval()

    def text_greedy_decoding_with_rules(self, text: str, seed: Optional[int] = None) -> str:
        source = self.source_tokenizer.encode(remove_stress_marks(text)).unsqueeze(0)
        output = self.tensor_greedy_decoding_with_rules(source, seed)
        output_tokens = self.target_tokenizer.decode(output.squeeze())

        stressed_text = ""
        text_iterator = iter(text)

        for token in output_tokens:
            stressed_text += next(text_iterator)

            if token in self.target_tokenizer.vocab.stress_token_to_id:
                stressed_text += token

        return stressed_text

    def tensor_greedy_decoding_with_rules(self, source: Tensor, seed: Optional[int] = None) -> Tensor:
        if source.size(0) != 1:
            raise RuntimeError(f"Tensor must have a batch size of 1, got {source.size()}")

        if seed is not None:
            seed_everything(seed)

        self.model.eval()
        source = source.to(self.device)

        # Exclude the SOS and EOS tokens
        source_ids = source.squeeze().tolist()[1:-1]
        source_length = len(source_ids)

        context_ids = [Vocab.SOS.id]
        is_word_stressed = False
        i = 0

        def infer_and_append() -> bool:
            context = torch.tensor([context_ids]).to(self.device)
            with torch.no_grad():
                output = self.model(
                    source=source,
                    target=context,
                    target_mask=self.model.transformer.generate_square_subsequent_mask(
                        context.size(1), device=self.device, dtype=torch.bool
                    )
                )
            output_id = output.topk(1)[1].view(-1)[-1].item()
            context_ids.append(output_id)
            return output_id in self.target_tokenizer.vocab.stress_token_to_id.values()

        while i < source_length:
            current_id = source_ids[i]

            if not self.source_tokenizer.vocab.id_to_token[current_id].isalpha():
                is_word_stressed = False

            if not is_word_stressed and current_id in self.stress_letter_ids:
                is_word_stressed = infer_and_append()
                i += 1

            else:
                i += 1
                next_id = source_ids[i] if i < source_length else None
                context_ids.append(Vocab.UNK.id)

                if not is_word_stressed and next_id in self.stress_letter_ids:
                    is_word_stressed = infer_and_append()
                    i += 1

        context_ids.append(Vocab.EOS.id)
        return torch.tensor([context_ids]).to(self.device)
