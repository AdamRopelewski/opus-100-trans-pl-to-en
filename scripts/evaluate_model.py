from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import torch

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.translate import (
    _checkpoint_path,
    _load_sentencepiece,
    _model_config,
    _resolve_device,
    _tokenizer_path,
)
from src.model.transformer_nmt import TransformerNMT
from src.utils.config import get_nested, load_config
from src.utils.tokenizer import iter_parallel_rows


T = TypeVar("T")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PL->EN Transformer checkpoint on the test split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/project_config.yaml"),
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint path. Defaults to stage7_eval.inference_checkpoint when it exists, then stage6_train.output_best_checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--decode",
        choices=("beam", "greedy"),
        default="beam",
        help="Decoding strategy.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation decode batch size. Defaults to stage7_eval.batch_size.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N rows.")
    return parser


def _test_split_path(config_data: dict[str, Any]) -> Path:
    processed_dir = Path(
        get_nested(
            config_data, "stage2_cleaning.outputs.processed_dir", "data/processed/en-pl"
        )
    )
    pattern = str(get_nested(config_data, "dataset.splits.test_pattern", "test-*.parquet"))
    matches = sorted(processed_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one test parquet in '{processed_dir}' using '{pattern}', found {len(matches)}."
        )
    return matches[0]


def _stage7_settings(config_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "beam_size": int(get_nested(config_data, "stage7_eval.beam_size", 4)),
        "batch_size": int(get_nested(config_data, "stage7_eval.batch_size", 16)),
        "length_penalty": float(get_nested(config_data, "stage7_eval.length_penalty", 1.0)),
        "max_new_tokens": int(get_nested(config_data, "stage7_eval.max_new_tokens", 128)),
        "num_sample_translations": int(
            get_nested(config_data, "stage7_eval.num_sample_translations", 100)
        ),
        "metrics_json": Path(
            get_nested(
                config_data,
                "stage7_eval.outputs.metrics_json",
                "reports/eval_metrics.json",
            )
        ),
        "translations_tsv": Path(
            get_nested(
                config_data,
                "stage7_eval.outputs.translations_tsv",
                "reports/translations/test_translations.tsv",
            )
        ),
        "samples_md": Path(
            get_nested(
                config_data,
                "stage7_eval.outputs.samples_md",
                "reports/translations/sample_translations.md",
            )
        ),
    }


def _compute_metrics(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    if sacrebleu is None:
        raise RuntimeError("Missing dependency: install sacrebleu to run Stage 7 eval.")
    return {
        "sacrebleu": float(sacrebleu.corpus_bleu(hypotheses, [references]).score),
        "chrf": float(sacrebleu.corpus_chrf(hypotheses, [references]).score),
    }


def _write_translations_tsv(
    path: Path, rows: list[tuple[int, str, str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["index", "source_pl", "reference_en", "hypothesis_en"])
        writer.writerows(rows)


def _write_samples_markdown(
    path: Path, rows: list[tuple[int, str, str, str]], limit: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Sample Translations", ""]
    for index, source, reference, hypothesis in rows[:limit]:
        lines.extend(
            [
                f"## Row {index}",
                "",
                f"PL: {source}",
                "",
                f"Reference EN: {reference}",
                "",
                f"Hypothesis EN: {hypothesis}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_model_bundle(
    config_data: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> tuple[Any, TransformerNMT, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = _load_sentencepiece(_tokenizer_path(config_data, checkpoint))
    model_cfg = _model_config(
        config_data, checkpoint, int(tokenizer.vocab_size()), int(tokenizer.pad_id())
    )
    model = TransformerNMT(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return tokenizer, model, model_cfg


def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1
    )


def _chunks(items: list[T], size: int) -> list[list[T]]:
    if size <= 0:
        raise ValueError("Evaluation batch size must be positive.")
    return [items[start : start + size] for start in range(0, len(items), size)]


def _encode_sources(
    sources: list[str], tokenizer: Any, model_cfg: Any, device: torch.device
) -> torch.Tensor:
    pad_id = int(tokenizer.pad_id())
    eos_id = int(tokenizer.eos_id())
    encoded = [list(tokenizer.encode(source)) + [eos_id] for source in sources]
    max_source_len = int(model_cfg.max_seq_len)
    too_long = [len(tokens) for tokens in encoded if len(tokens) > max_source_len]
    if too_long:
        raise ValueError(
            f"Source too long: {max(too_long)} tokens > max_seq_len={max_source_len}"
        )
    max_len = max(len(tokens) for tokens in encoded)
    padded = [tokens + [pad_id] * (max_len - len(tokens)) for tokens in encoded]
    return torch.tensor(padded, dtype=torch.long, device=device)


def _decode_piece_ids(tokenizer: Any, output_ids: list[int]) -> str:
    special_ids = {
        int(tokenizer.pad_id()),
        int(tokenizer.bos_id()),
        int(tokenizer.eos_id()),
    }
    piece_ids = [token_id for token_id in output_ids if token_id not in special_ids]
    return str(tokenizer.decode(piece_ids))


def _batched_greedy_decode(
    model: TransformerNMT,
    src_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
) -> list[list[int]]:
    model.eval()
    src_key_padding_mask = src_ids.eq(model.config.pad_id)
    src_mask = model.make_src_attention_mask(src_key_padding_mask)
    batch_size = src_ids.size(0)
    device = src_ids.device

    with torch.inference_mode():
        enc_out = model.encode(src_ids, src_mask)
        tgt_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            tgt_key_padding_mask = tgt_ids.eq(model.config.pad_id)
            tgt_mask = model.make_tgt_attention_mask(
                tgt_key_padding_mask, _causal_mask(tgt_ids.size(1), device)
            )
            logits = model.decode(tgt_ids, enc_out, src_mask, tgt_mask)
            next_ids = logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
            tgt_ids = torch.cat([tgt_ids, next_ids.unsqueeze(1)], dim=1)
            finished |= next_ids.eq(eos_id)
            if bool(finished.all().item()):
                break

    return [row.tolist() for row in tgt_ids]


def _score_with_length_penalty(score: float, length: int, length_penalty: float) -> float:
    if length_penalty <= 0:
        return score
    return score / (max(1, length) ** length_penalty)


def _sequence_length(tokens: list[int], eos_id: int) -> int:
    try:
        return tokens.index(eos_id) + 1
    except ValueError:
        return len(tokens)


def _batched_beam_search_decode(
    model: TransformerNMT,
    src_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    beam_size: int,
    length_penalty: float,
) -> list[list[int]]:
    if beam_size <= 1:
        return _batched_greedy_decode(model, src_ids, bos_id, eos_id, max_new_tokens)

    model.eval()
    batch_size = src_ids.size(0)
    device = src_ids.device
    src_key_padding_mask = src_ids.eq(model.config.pad_id)
    src_mask = model.make_src_attention_mask(src_key_padding_mask)

    with torch.inference_mode():
        enc_out = model.encode(src_ids, src_mask)
        enc_out = enc_out.repeat_interleave(beam_size, dim=0)
        src_mask = src_mask.repeat_interleave(beam_size, dim=0)

        beam_tokens = torch.full(
            (batch_size, beam_size, 1), bos_id, dtype=torch.long, device=device
        )
        beam_scores = torch.full((batch_size, beam_size), -torch.inf, device=device)
        beam_scores[:, 0] = 0.0

        for _ in range(max_new_tokens):
            flat_tokens = beam_tokens.reshape(batch_size * beam_size, -1)
            tgt_key_padding_mask = flat_tokens.eq(model.config.pad_id)
            tgt_mask = model.make_tgt_attention_mask(
                tgt_key_padding_mask, _causal_mask(flat_tokens.size(1), device)
            )
            logits = model.decode(flat_tokens, enc_out, src_mask, tgt_mask)
            log_probs = torch.log_softmax(logits[:, -1], dim=-1).view(
                batch_size, beam_size, -1
            )

            ended = beam_tokens[:, :, -1].eq(eos_id)
            if bool(ended.any().item()):
                log_probs = log_probs.masked_fill(ended.unsqueeze(-1), -torch.inf)
                log_probs[:, :, eos_id] = torch.where(
                    ended,
                    torch.zeros_like(beam_scores),
                    log_probs[:, :, eos_id],
                )

            vocab_size = log_probs.size(-1)
            candidate_scores = (beam_scores.unsqueeze(-1) + log_probs).view(
                batch_size, beam_size * vocab_size
            )
            top_scores, top_indices = torch.topk(candidate_scores, k=beam_size, dim=-1)
            source_beams = top_indices // vocab_size
            next_ids = top_indices % vocab_size

            gathered_tokens = beam_tokens.gather(
                1,
                source_beams.unsqueeze(-1).expand(
                    batch_size, beam_size, beam_tokens.size(-1)
                ),
            )
            beam_tokens = torch.cat([gathered_tokens, next_ids.unsqueeze(-1)], dim=-1)
            beam_scores = top_scores

            if bool(beam_tokens[:, :, -1].eq(eos_id).all().item()):
                break

    decoded: list[list[int]] = []
    token_lists = beam_tokens.detach().cpu().tolist()
    score_lists = beam_scores.detach().cpu().tolist()
    for beams, scores in zip(token_lists, score_lists, strict=True):
        best_index = max(
            range(len(beams)),
            key=lambda index: _score_with_length_penalty(
                float(scores[index]),
                _sequence_length(beams[index], eos_id),
                length_penalty,
            ),
        )
        decoded.append(beams[best_index][:_sequence_length(beams[best_index], eos_id)])
    return decoded


def _translate_batch(
    sources: list[str],
    tokenizer: Any,
    model: TransformerNMT,
    model_cfg: Any,
    device: torch.device,
    decode: str,
    max_new_tokens: int,
    beam_size: int,
    length_penalty: float,
) -> list[str]:
    src_ids = _encode_sources(sources, tokenizer, model_cfg, device)
    bos_id = int(tokenizer.bos_id())
    eos_id = int(tokenizer.eos_id())
    if decode == "greedy":
        output_ids = _batched_greedy_decode(
            model, src_ids, bos_id, eos_id, max_new_tokens
        )
    else:
        output_ids = _batched_beam_search_decode(
            model, src_ids, bos_id, eos_id, max_new_tokens, beam_size, length_penalty
        )
    return [_decode_piece_ids(tokenizer, ids) for ids in output_ids]


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_rows(
    rows: list[tuple[str, str]],
    tokenizer: Any,
    model: TransformerNMT,
    model_cfg: Any,
    device: torch.device,
    decode: str,
    settings: dict[str, Any],
) -> list[tuple[int, str, str, str]]:
    outputs: list[tuple[int, str, str, str]] = []
    batch_size = int(settings["batch_size"])
    max_new_tokens = min(int(settings["max_new_tokens"]), int(model_cfg.max_seq_len) - 1)
    row_batches = _chunks(list(enumerate(rows)), batch_size)
    iterator = (
        tqdm(row_batches, desc="Evaluating", unit="batch")
        if tqdm is not None
        else row_batches
    )
    for batch in iterator:
        batch_indices = [index for index, _row in batch]
        sources = [source for _index, (source, _reference) in batch]
        references = [reference for _index, (_source, reference) in batch]
        hypotheses = _translate_batch(
            sources,
            tokenizer,
            model,
            model_cfg,
            device,
            decode,
            max_new_tokens,
            int(settings["beam_size"]),
            float(settings["length_penalty"]),
        )
        outputs.extend(
            (index, source, reference, hypothesis)
            for index, source, reference, hypothesis in zip(
                batch_indices, sources, references, hypotheses, strict=True
            )
        )
        _clear_cuda_cache(device)
    return outputs


def main() -> int:
    args = _build_parser().parse_args()
    config_data = load_config(args.config)
    if not bool(get_nested(config_data, "stage7_eval.enabled", True)):
        print("stage7_eval.enabled is false, skipping evaluation.")
        return 0

    checkpoint_path = _checkpoint_path(config_data, args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    try:
        test_path = _test_split_path(config_data)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    settings = _stage7_settings(config_data)
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    device = _resolve_device(args.device)
    print(
        f"Evaluating checkpoint {checkpoint_path} on test split {test_path} "
        f"using device {device}."
    )
    tokenizer, model, model_cfg = _load_model_bundle(config_data, checkpoint_path, device)
    max_supported_new_tokens = max(1, int(model_cfg.max_seq_len) - 1)
    if int(settings["max_new_tokens"]) > max_supported_new_tokens:
        print(
            f"Capping max_new_tokens from {settings['max_new_tokens']} "
            f"to {max_supported_new_tokens} for max_seq_len={model_cfg.max_seq_len}."
        )
        settings["max_new_tokens"] = max_supported_new_tokens
    rows = list(iter_parallel_rows(test_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print("No test rows to evaluate.", file=sys.stderr)
        return 4

    translations = _evaluate_rows(
        rows, tokenizer, model, model_cfg, device, args.decode, settings
    )
    references = [row[2] for row in translations]
    hypotheses = [row[3] for row in translations]
    try:
        metrics = _compute_metrics(hypotheses, references)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 5

    _write_translations_tsv(settings["translations_tsv"], translations)
    _write_samples_markdown(
        settings["samples_md"], translations, int(settings["num_sample_translations"])
    )
    _write_metrics_json(
        settings["metrics_json"],
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "test_split": str(test_path),
            "rows_evaluated": len(translations),
            "decode": args.decode,
            "beam_size": settings["beam_size"],
            "batch_size": settings["batch_size"],
            "length_penalty": settings["length_penalty"],
            "max_new_tokens": settings["max_new_tokens"],
            "metrics": metrics,
        },
    )

    print(
        f"Evaluation complete: rows={len(translations)} "
        f"sacrebleu={metrics['sacrebleu']:.2f} chrf={metrics['chrf']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
