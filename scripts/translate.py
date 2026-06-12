from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model.transformer_nmt import (
    TransformerNMT,
    TransformerNMTConfig,
    beam_search_decode,
    greedy_decode,
)
from src.utils.config import get_nested, load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate PL text to EN with trained Transformer checkpoint."
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
        help="Model checkpoint path. Defaults to stage6_train.output_best_checkpoint.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Source text. If omitted, text is read from stdin.",
    )
    parser.add_argument(
        "--decode",
        choices=("beam", "greedy"),
        default="beam",
        help="Decoding strategy.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size. Defaults to stage7_eval.beam_size.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="Length penalty. Defaults to stage7_eval.length_penalty.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Token limit. Defaults to stage7_eval.max_new_tokens.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    return parser


def _load_sentencepiece(model_path: Path):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install sentencepiece to translate."
        ) from exc
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    return spm.SentencePieceProcessor(model_file=str(model_path))


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_path(config_data: dict[str, Any], path: Path | None) -> Path:
    if path is not None:
        return path
    return Path(
        get_nested(
            config_data,
            "stage6_train.output_best_checkpoint",
            "checkpoints/model_best.pt",
        )
    )


def _tokenizer_path(config_data: dict[str, Any], checkpoint: dict[str, Any]) -> Path:
    saved_path = checkpoint.get("tokenizer_path")
    if saved_path:
        return Path(str(saved_path))
    prefix = Path(
        get_nested(config_data, "stage3_tokenizer.model_prefix", "tokenizers/spm_pl_en")
    )
    return prefix.with_suffix(".model")


def _model_config(
    config_data: dict[str, Any],
    checkpoint: dict[str, Any],
    vocab_size: int,
    pad_id: int,
) -> TransformerNMTConfig:
    saved_config = checkpoint.get("model_config")
    if isinstance(saved_config, dict):
        allowed = {field.name for field in fields(TransformerNMTConfig)}
        values = {key: value for key, value in saved_config.items() if key in allowed}
        values["vocab_size"] = int(values.get("vocab_size", vocab_size))
        values["pad_id"] = int(values.get("pad_id", pad_id))
        return TransformerNMTConfig(**values)

    preset_name = str(get_nested(config_data, "stage5_model.preset", "small"))
    preset = get_nested(config_data, f"stage5_model.presets.{preset_name}", {})
    return TransformerNMTConfig(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_seq_len=int(get_nested(config_data, "stage4_dataloader.max_seq_len", 128)),
        d_model=int(preset.get("d_model", 256)),
        nhead=int(preset.get("nhead", 8)),
        num_encoder_layers=int(preset.get("num_encoder_layers", 4)),
        num_decoder_layers=int(preset.get("num_decoder_layers", 4)),
        dim_feedforward=int(preset.get("dim_feedforward", 1024)),
        dropout=float(preset.get("dropout", 0.0)),
        tie_decoder_embeddings=bool(preset.get("tie_decoder_embeddings", True)),
    )


def _source_text(text: str | None) -> str:
    if text is not None:
        return text
    return sys.stdin.read().strip()


def _translate_text(
    text: str,
    tokenizer: Any,
    model: TransformerNMT,
    model_cfg: TransformerNMTConfig,
    device: torch.device,
    decode: str,
    max_new_tokens: int,
    beam_size: int,
    length_penalty: float,
) -> str:
    src_tokens = tokenizer.encode(text) + [int(tokenizer.eos_id())]
    max_source_len = model_cfg.max_seq_len
    if len(src_tokens) > max_source_len:
        raise ValueError(
            f"Source too long: {len(src_tokens)} tokens > max_seq_len={max_source_len}"
        )

    src_ids = torch.tensor([src_tokens], dtype=torch.long, device=device)
    if decode == "greedy":
        output_ids = greedy_decode(
            model,
            src_ids,
            int(tokenizer.bos_id()),
            int(tokenizer.eos_id()),
            max_new_tokens,
        )
    else:
        output_ids = beam_search_decode(
            model,
            src_ids,
            int(tokenizer.bos_id()),
            int(tokenizer.eos_id()),
            max_new_tokens,
            beam_size,
            length_penalty,
        )

    special_ids = {
        int(tokenizer.pad_id()),
        int(tokenizer.bos_id()),
        int(tokenizer.eos_id()),
    }
    piece_ids = [token_id for token_id in output_ids if token_id not in special_ids]
    return str(tokenizer.decode(piece_ids))


def _interactive_loop(
    tokenizer: Any,
    model: TransformerNMT,
    model_cfg: TransformerNMTConfig,
    device: torch.device,
    decode: str,
    max_new_tokens: int,
    beam_size: int,
    length_penalty: float,
) -> int:
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        try:
            print(
                _translate_text(
                    text,
                    tokenizer,
                    model,
                    model_cfg,
                    device,
                    decode,
                    max_new_tokens,
                    beam_size,
                    length_penalty,
                )
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)


def main() -> int:
    args = _build_parser().parse_args()
    config_data = load_config(args.config)
    checkpoint_path = _checkpoint_path(config_data, args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    device = _resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = _load_sentencepiece(_tokenizer_path(config_data, checkpoint))
    model_cfg = _model_config(
        config_data, checkpoint, int(tokenizer.vocab_size()), int(tokenizer.pad_id())
    )
    model = TransformerNMT(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(
            get_nested(config_data, "stage7_eval.max_new_tokens", model_cfg.max_seq_len)
        )
    beam_size = args.beam_size
    if beam_size is None:
        beam_size = int(get_nested(config_data, "stage7_eval.beam_size", 4))
    length_penalty = args.length_penalty
    if length_penalty is None:
        length_penalty = float(
            get_nested(config_data, "stage7_eval.length_penalty", 1.0)
        )

    if args.text is None and sys.stdin.isatty():
        return _interactive_loop(
            tokenizer,
            model,
            model_cfg,
            device,
            args.decode,
            max_new_tokens,
            beam_size,
            length_penalty,
        )

    text = _source_text(args.text)
    if not text:
        print("No source text provided.", file=sys.stderr)
        return 2

    try:
        print(
            _translate_text(
                text,
                tokenizer,
                model,
                model_cfg,
                device,
                args.decode,
                max_new_tokens,
                beam_size,
                length_penalty,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
