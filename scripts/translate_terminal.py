from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.amp import autocast

from src.data.collate import make_causal_mask
from src.model.transformer_nmt import TransformerNMT, TransformerNMTConfig
from src.train.device import CudaRequiredError, resolve_training_device, select_amp_precision
from src.utils.config import get_nested, load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate Polish terminal input to English using a Stage 4 checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/project_config.yaml"), help="Path to project config YAML.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stage4_best.pt"), help="Stage 4 checkpoint path.")
    parser.add_argument("--device", default="cuda", help="Torch device to use. Defaults to cuda and requires a GPU.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum generated English tokens.")
    parser.add_argument("--precision", default=None, choices=("bf16", "fp16", "fp32"), help="Inference precision. Defaults to config stage6_train.precision.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast and run inference in fp32.")
    parser.add_argument("--text", default=None, help="Single Polish sentence to translate, then exit.")
    return parser


def _load_sentencepiece(model_path: Path):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install sentencepiece to translate.") from exc
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    return spm.SentencePieceProcessor(model_file=str(model_path))


def _checkpoint_model_config(checkpoint: dict, tokenizer, config_data: dict) -> TransformerNMTConfig:
    payload = checkpoint.get("model_config")
    if payload:
        return TransformerNMTConfig(**payload)

    preset_name = str(get_nested(config_data, "stage5_model.preset", "small"))
    preset = get_nested(config_data, f"stage5_model.presets.{preset_name}", {})
    return TransformerNMTConfig(
        vocab_size=int(tokenizer.vocab_size()),
        pad_id=int(tokenizer.pad_id()),
        max_seq_len=int(get_nested(config_data, "stage4_dataloader.max_seq_len", 128)),
        d_model=int(preset.get("d_model", 256)),
        nhead=int(preset.get("nhead", 8)),
        num_encoder_layers=int(preset.get("num_encoder_layers", 4)),
        num_decoder_layers=int(preset.get("num_decoder_layers", 4)),
        dim_feedforward=int(preset.get("dim_feedforward", 1024)),
        dropout=float(preset.get("dropout", 0.1)),
        tie_decoder_embeddings=bool(preset.get("tie_decoder_embeddings", True)),
    )


def _resolve_tokenizer_path(checkpoint: dict, config_data: dict) -> Path:
    checkpoint_tokenizer = checkpoint.get("tokenizer_path")
    if checkpoint_tokenizer:
        return Path(str(checkpoint_tokenizer))
    tokenizer_prefix = Path(get_nested(config_data, "stage3_tokenizer.model_prefix", "tokenizers/spm_pl_en"))
    return tokenizer_prefix.with_suffix(".model")


def _encode_source(text: str, tokenizer, eos_id: int, max_seq_len: int) -> list[int]:
    ids = list(tokenizer.encode(text, out_type=int)) + [eos_id]
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
        ids[-1] = eos_id
    return ids


def _decode_greedy(
    model: TransformerNMT,
    tokenizer,
    text: str,
    device: torch.device,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_seq_len: int,
    max_new_tokens: int,
    amp_dtype: torch.dtype | None,
) -> str:
    src_ids = torch.tensor([_encode_source(text, tokenizer, eos_id, max_seq_len)], dtype=torch.long, device=device)
    src_key_padding_mask = src_ids.eq(pad_id)
    generated = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            tgt_key_padding_mask = generated.eq(pad_id)
            tgt_causal_mask = make_causal_mask(generated.size(1), device=device)
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda" and amp_dtype is not None):
                logits = model(src_ids, generated, src_key_padding_mask, tgt_key_padding_mask, tgt_causal_mask)
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            if next_id == eos_id:
                break
            generated = torch.cat(
                [generated, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            if generated.size(1) >= max_seq_len:
                break

    output_ids = generated.squeeze(0).tolist()[1:]
    return tokenizer.decode(output_ids).strip()


def _iter_input_lines(single_text: str | None):
    if single_text is not None:
        yield single_text
        return
    if sys.stdin.isatty():
        print("Enter Polish text. Press Ctrl-D or Ctrl-C to exit.")
        while True:
            try:
                line = input("pl> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            yield line
    else:
        for line in sys.stdin:
            yield line.rstrip("\n")


def main() -> int:
    args = _build_parser().parse_args()
    config_data = load_config(args.config)

    try:
        device_info = resolve_training_device(args.device, require_cuda=True, allow_cpu_fallback=False)
    except CudaRequiredError as exc:
        print(str(exc), file=sys.stderr)
        return 6

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    checkpoint = torch.load(args.checkpoint, map_location=device_info.device)
    tokenizer_path = _resolve_tokenizer_path(checkpoint, config_data)
    tokenizer = _load_sentencepiece(tokenizer_path)
    model_config = _checkpoint_model_config(checkpoint, tokenizer, config_data)
    model = TransformerNMT(model_config).to(device_info.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pad_id = int(checkpoint.get("tokenizer_special_ids", {}).get("pad_id", tokenizer.pad_id()))
    bos_id = int(checkpoint.get("tokenizer_special_ids", {}).get("bos_id", tokenizer.bos_id()))
    eos_id = int(checkpoint.get("tokenizer_special_ids", {}).get("eos_id", tokenizer.eos_id()))
    max_new_tokens = args.max_new_tokens or int(get_nested(config_data, "stage7_eval.max_new_tokens", model_config.max_seq_len))
    max_new_tokens = max(1, min(max_new_tokens, model_config.max_seq_len - 1))

    precision = args.precision or str(get_nested(config_data, "stage6_train.precision", "bf16"))
    fallback_precision = str(get_nested(config_data, "stage6_train.fallback_precision", "fp16"))
    precision_name, amp_dtype, _ = select_amp_precision(precision, fallback_precision)
    if args.no_amp:
        precision_name, amp_dtype = "fp32", None

    print(f"Using {device_info.device} ({device_info.gpu_name}); checkpoint={args.checkpoint}; precision={precision_name}", file=sys.stderr)
    for line in _iter_input_lines(args.text):
        text = line.strip()
        if not text:
            continue
        translation = _decode_greedy(
            model,
            tokenizer,
            text,
            device_info.device,
            pad_id,
            bos_id,
            eos_id,
            model_config.max_seq_len,
            max_new_tokens,
            amp_dtype,
        )
        print(translation)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
