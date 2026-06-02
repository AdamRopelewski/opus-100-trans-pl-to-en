from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_nested, load_config
from src.utils.tokenizer import (
    TokenizerConfig,
    collect_long_word_split_stats,
    collect_split_stats,
    collect_word_frequencies,
    common_long_word_examples,
    load_sentencepiece_encoder,
    train_sentencepiece,
    write_tokenizer_stats_markdown,
    write_training_corpus,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train shared SentencePiece tokenizer for cleaned OPUS-100 en-pl splits.")
    parser.add_argument("--config", type=Path, default=Path("configs/project_config.yaml"), help="Path to project config YAML.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing SentencePiece model files.")
    return parser


def _resolve_split_file(data_dir: Path, split_name: str, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one parquet for split '{split_name}' in '{data_dir}' using '{pattern}', found {len(matches)}."
        )
    return matches[0]


def _build_tokenizer_config(config_data: dict) -> TokenizerConfig:
    tokens = get_nested(config_data, "stage3_tokenizer.special_tokens", {})
    return TokenizerConfig(
        model_type=str(get_nested(config_data, "stage3_tokenizer.model_type", "bpe")),
        vocab_size=int(get_nested(config_data, "stage3_tokenizer.vocab_size", 16000)),
        character_coverage=float(get_nested(config_data, "stage3_tokenizer.character_coverage", 1.0)),
        model_prefix=Path(get_nested(config_data, "stage3_tokenizer.model_prefix", "tokenizers/spm_pl_en")),
        input_sentence_size=int(get_nested(config_data, "stage3_tokenizer.input_sentence_size", 2_000_000)),
        shuffle_input_sentence=bool(get_nested(config_data, "stage3_tokenizer.shuffle_input_sentence", True)),
        pad_token=str(tokens.get("pad", "<pad>")),
        unk_token=str(tokens.get("unk", "<unk>")),
        bos_token=str(tokens.get("bos", "<bos>")),
        eos_token=str(tokens.get("eos", "<eos>")),
    )


def main() -> int:
    args = _build_parser().parse_args()
    config_data = load_config(args.config)

    if not bool(get_nested(config_data, "stage3_tokenizer.enabled", True)):
        print("stage3_tokenizer.enabled is false, skipping tokenizer training.")
        return 0

    if str(get_nested(config_data, "stage3_tokenizer.type", "sentencepiece")) != "sentencepiece":
        print("Only stage3_tokenizer.type=sentencepiece is supported.", file=sys.stderr)
        return 2
    if not bool(get_nested(config_data, "stage3_tokenizer.joint_vocab", True)):
        print("Only joint/shared vocabulary is supported for Stage 3.", file=sys.stderr)
        return 2
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        print("Missing dependency: install sentencepiece to train Stage 3 tokenizer.", file=sys.stderr)
        return 5

    processed_dir = Path(get_nested(config_data, "stage2_cleaning.outputs.processed_dir", "data/processed/en-pl"))
    reports_dir = Path(get_nested(config_data, "paths.reports_dir", "reports"))
    report_path = Path(get_nested(config_data, "stage3_tokenizer.outputs.report_md", str(reports_dir / "tokenizer_stats.md")))
    training_corpus_path = Path(
        get_nested(config_data, "stage3_tokenizer.outputs.training_corpus", "tokenizers/spm_pl_en_train.txt")
    )
    split_patterns = {
        "train": str(get_nested(config_data, "dataset.splits.train_pattern", "train-*.parquet")),
        "validation": str(get_nested(config_data, "dataset.splits.validation_pattern", "validation-*.parquet")),
        "test": str(get_nested(config_data, "dataset.splits.test_pattern", "test-*.parquet")),
    }

    try:
        split_files = {
            split: _resolve_split_file(processed_dir, split, split_patterns[split])
            for split in ("train", "validation", "test")
        }
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    tokenizer_cfg = _build_tokenizer_config(config_data)
    model_path = tokenizer_cfg.model_prefix.with_suffix(".model")
    vocab_path = tokenizer_cfg.model_prefix.with_suffix(".vocab")
    if not args.force and (model_path.exists() or vocab_path.exists()):
        print(f"Tokenizer outputs already exist: {model_path}, {vocab_path}. Use --force to overwrite.", file=sys.stderr)
        return 4

    corpus_lines = write_training_corpus(split_files["train"], training_corpus_path)
    try:
        train_sentencepiece(training_corpus_path, tokenizer_cfg)
        encode, unk_id = load_sentencepiece_encoder(model_path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 5

    frequencies = collect_word_frequencies(split_files["train"])
    split_stats = [collect_split_stats(split, path, encode, unk_id) for split, path in split_files.items()]
    long_word_stats = collect_long_word_split_stats(frequencies, encode)
    common_long_words = common_long_word_examples(frequencies, encode)
    write_tokenizer_stats_markdown(
        report_path,
        tokenizer_cfg,
        corpus_lines,
        split_stats,
        long_word_stats,
        common_long_words,
    )

    print(f"Tokenizer trained: {model_path}")
    print(f"Vocabulary saved: {vocab_path}")
    print(f"Stats written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
