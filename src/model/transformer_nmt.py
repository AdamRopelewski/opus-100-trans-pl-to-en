from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerNMTConfig:
    vocab_size: int
    pad_id: int = 0
    max_seq_len: int = 128
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    tie_decoder_embeddings: bool = True


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerNMT(nn.Module):
    def __init__(self, config: TransformerNMTConfig) -> None:
        super().__init__()
        self.config = config
        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        encoder_norm = nn.LayerNorm(config.d_model)
        encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=encoder_norm,
            enable_nested_tensor=False,
        )
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            custom_encoder=encoder,
        )
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_decoder_embeddings:
            self.output_projection.weight = self.tgt_embedding.weight

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        tgt_causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        src = self.position(self.src_embedding(src_ids) * math.sqrt(self.config.d_model))
        tgt = self.position(self.tgt_embedding(tgt_in_ids) * math.sqrt(self.config.d_model))
        hidden = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_projection(hidden)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
