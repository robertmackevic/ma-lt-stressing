from argparse import Namespace
from typing import Optional

from torch import Tensor
from torch.nn import Module, Embedding, Transformer, Linear

from src.model.encoding import PositionalEncoding


class Seq2SeqTransformer(Module):
    def __init__(self, config: Namespace, source_vocab_size: int, target_vocab_size: int) -> None:
        super().__init__()
        self.source_embedding = Embedding(source_vocab_size, config.embedding_dim)
        self.target_embedding = Embedding(target_vocab_size, config.embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim=config.embedding_dim,
            max_length=config.max_sequence_length,
            dropout=config.dropout,
        )
        self.transformer = Transformer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.generator = Linear(config.embedding_dim, target_vocab_size)

    def forward(
            self,
            source: Tensor,
            target: Tensor,
            source_mask: Optional[Tensor] = None,
            target_mask: Optional[Tensor] = None,
            source_padding_mask: Optional[Tensor] = None,
            target_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.transformer(
            src=self.positional_encoding(self.source_embedding(source)),
            tgt=self.positional_encoding(self.target_embedding(target)),
            src_mask=source_mask,
            tgt_mask=target_mask,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
        )
        return self.generator(output)
