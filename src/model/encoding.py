import math

import torch
from torch import Tensor
from torch.nn import Module, Dropout


class PositionalEncoding(Module):
    def __init__(self, embedding_dim: int, max_length: int, dropout: float) -> None:
        super().__init__()
        self.dropout = Dropout(p=dropout)
        self.embedding_dim = embedding_dim

        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        position = torch.arange(max_length).unsqueeze(1) * div_term

        pe = torch.zeros(1, max_length, embedding_dim)
        pe[:, :, 0::2] = torch.sin(position)
        pe[:, :, 1::2] = torch.cos(position)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # [batch_size, seq_length, d_model]
        return self.dropout(x + self.pe[:, :x.size(1)])
