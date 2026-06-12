from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = q.size(0)
        q_len = q.size(1)

        Q = self.q(q).view(batch_size, q_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k(k).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v(v).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class TransformerNMT(nn.Module):
    def __init__(self, config: TransformerNMTConfig) -> None:
        super().__init__()
        self.config = config
        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout)
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout)
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._reset_parameters()
        if config.tie_decoder_embeddings:
            self.output_projection.weight = self.tgt_embedding.weight

    def _reset_parameters(self) -> None:
        std = self.config.d_model**-0.5
        nn.init.normal_(self.src_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.tgt_embedding.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.src_embedding.weight[self.config.pad_id].zero_()
            self.tgt_embedding.weight[self.config.pad_id].zero_()

    def make_src_attention_mask(self, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        return src_key_padding_mask.logical_not().unsqueeze(1).unsqueeze(2)

    def make_tgt_attention_mask(
        self,
        tgt_key_padding_mask: torch.Tensor,
        tgt_causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        pad_mask = tgt_key_padding_mask.logical_not().unsqueeze(1).unsqueeze(2)
        causal_mask = tgt_causal_mask.logical_not().unsqueeze(0).unsqueeze(1)
        return pad_mask & causal_mask

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.position(self.src_embedding(src_ids) * math.sqrt(self.config.d_model))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt_in_ids: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.position(self.tgt_embedding(tgt_in_ids) * math.sqrt(self.config.d_model))
        for layer in self.decoder:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.output_projection(x)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        tgt_causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_mask = self.make_src_attention_mask(src_key_padding_mask)
        tgt_mask = self.make_tgt_attention_mask(tgt_key_padding_mask, tgt_causal_mask)
        enc_out = self.encode(src_ids, src_mask)
        return self.decode(tgt_in_ids, enc_out, src_mask, tgt_mask)


def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)


def greedy_decode(
    model: TransformerNMT,
    src_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
) -> list[int]:
    model.eval()
    device = src_ids.device
    src_key_padding_mask = src_ids.eq(model.config.pad_id)
    src_mask = model.make_src_attention_mask(src_key_padding_mask)
    with torch.no_grad():
        enc_out = model.encode(src_ids, src_mask)

    tgt_ids = [bos_id]
    for _ in range(max_new_tokens):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        tgt_key_padding_mask = tgt.eq(model.config.pad_id)
        tgt_mask = model.make_tgt_attention_mask(tgt_key_padding_mask, _causal_mask(tgt.size(1), device))
        with torch.no_grad():
            logits = model.decode(tgt, enc_out, src_mask, tgt_mask)
        next_id = int(logits[0, -1].argmax().item())
        tgt_ids.append(next_id)
        if next_id == eos_id:
            break
    return tgt_ids


def beam_search_decode(
    model: TransformerNMT,
    src_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    beam_size: int,
    length_penalty: float,
) -> list[int]:
    if beam_size <= 1:
        return greedy_decode(model, src_ids, bos_id, eos_id, max_new_tokens)

    model.eval()
    device = src_ids.device
    src_key_padding_mask = src_ids.eq(model.config.pad_id)
    src_mask = model.make_src_attention_mask(src_key_padding_mask)
    with torch.no_grad():
        enc_out = model.encode(src_ids, src_mask)

    beams: list[tuple[list[int], float]] = [([bos_id], 0.0)]
    finished: list[tuple[list[int], float]] = []

    for _ in range(max_new_tokens):
        candidates: list[tuple[list[int], float]] = []
        for tokens, score in beams:
            if tokens[-1] == eos_id:
                finished.append((tokens, score))
                continue
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_key_padding_mask = tgt.eq(model.config.pad_id)
            tgt_mask = model.make_tgt_attention_mask(tgt_key_padding_mask, _causal_mask(tgt.size(1), device))
            with torch.no_grad():
                logits = model.decode(tgt, enc_out, src_mask, tgt_mask)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
            top_scores, top_ids = torch.topk(log_probs, k=beam_size)
            for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist(), strict=True):
                candidates.append((tokens + [int(token_id)], score + float(token_score)))

        if not candidates:
            break
        beams = sorted(candidates, key=lambda item: _score_with_length_penalty(item[1], len(item[0]), length_penalty), reverse=True)[
            :beam_size
        ]
        if all(tokens[-1] == eos_id for tokens, _score in beams):
            finished.extend(beams)
            break

    best_pool = finished if finished else beams
    best_tokens, _best_score = max(
        best_pool,
        key=lambda item: _score_with_length_penalty(item[1], len(item[0]), length_penalty),
    )
    return best_tokens


def _score_with_length_penalty(score: float, length: int, length_penalty: float) -> float:
    if length_penalty <= 0:
        return score
    return score / (max(1, length) ** length_penalty)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
