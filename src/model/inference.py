from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.nn.functional import log_softmax

from src.data.const import STRESS_LETTERS, UNK, SOS, EOS
from src.data.processing import remove_stress_marks
from src.data.tokenizer import Tokenizer
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

    def text_decoding(
            self,
            text: str,
            num_beams: Optional[int] = None,
            seed: Optional[int] = None,
            with_rules: bool = False
    ) -> str:
        source = self.source_tokenizer.encode(remove_stress_marks(text)).unsqueeze(0)

        if num_beams is not None:
            output = self.tensor_beam_search_decoding(source, num_beams, seed)

        elif with_rules:
            output = self.tensor_greedy_decoding_with_rules(source, seed)

        else:
            output = self.tensor_greedy_decoding(source, seed)

        output_tokens = self.target_tokenizer.decode(output.squeeze())

        stressed_text = ""
        text_iterator = iter(text)

        for token in output_tokens:
            try:
                stressed_text += next(text_iterator)
            except StopIteration:
                print(f"Skipping token {token} in text `{stressed_text}`")

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

        context_ids = [SOS.id]
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
                context_ids.append(UNK.id)

                if not is_word_stressed and next_id in self.stress_letter_ids:
                    is_word_stressed = infer_and_append()
                    i += 1

        context_ids.append(EOS.id)
        return torch.tensor([context_ids]).to(self.device)

    def tensor_greedy_decoding(self, source: torch.Tensor, seed: Optional[int] = None) -> Tensor:
        if seed is not None:
            seed_everything(seed)

        self.model.eval()
        source = source.to(self.device)
        batch_size = source.size(0)

        # Start each sequence with SOS token
        decoded_ids = torch.full((batch_size, 1), SOS.id, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step in range(self.config.max_sequence_length):
            with torch.no_grad():
                output = self.model(
                    source=source,
                    target=decoded_ids,
                    target_mask=self.model.transformer.generate_square_subsequent_mask(
                        decoded_ids.size(1), device=self.device, dtype=torch.bool
                    )
                )
            # Take the last time-step's output for each item and select the token with highest probability (greedy)
            next_tokens = output[:, -1, :].argmax(-1)
            decoded_ids = torch.cat([decoded_ids, next_tokens.unsqueeze(1)], dim=1)

            # Update which sequences have finished (EOS)
            finished |= (next_tokens == EOS.id)
            if finished.all():
                break

        return decoded_ids

    def tensor_beam_search_decoding(self, source: Tensor, beam_size: int = 3, seed: Optional[int] = None) -> Tensor:
        if source.size(0) != 1:
            raise RuntimeError(f"Beam search decoding only supports batch size of 1, got {source.size(0)}")

        if seed is not None:
            seed_everything(seed)

        self.model.eval()
        source = source.to(self.device)

        beams = [([SOS.id], 0.0)]
        completed_beams = []

        for _ in range(self.config.max_sequence_length):
            new_beams = []

            # For each sequence in the beam, extend it if not already terminated
            for sequence, score in beams:
                if sequence[-1] == EOS.id:
                    completed_beams.append((sequence, score))
                    continue

                # Build the target context for the current sequence
                context = torch.tensor([sequence], device=self.device)

                with torch.no_grad():
                    output = self.model(
                        source=source,
                        target=context,
                        target_mask=self.model.transformer.generate_square_subsequent_mask(
                            context.size(1), device=self.device, dtype=torch.bool
                        )
                    )
                # Get logits for the last token in the sequence [1, vocab_size]
                logits = output[:, -1, :]
                log_probs = log_softmax(logits, dim=-1)

                # Get the top beam_size token candidates
                top_k_log_probs, top_k_indices = log_probs.topk(beam_size)

                for log_prob, token_id in zip(top_k_log_probs[0], top_k_indices[0]):
                    new_seq = sequence + [token_id.item()]
                    new_score = score + log_prob.item()
                    new_beams.append((new_seq, new_score))

            if not new_beams:
                break

            # Retain the top beam_size candidates
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

        # If any sequence has finished with EOS, choose the best finished sequence;
        # otherwise, take the best sequence so far.
        if completed_beams:
            best_seq = max(completed_beams, key=lambda x: x[1])[0]

        else:
            best_seq = beams[0][0]

        return torch.tensor([best_seq], device=self.device)
