from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import pyarrow.parquet as pq

WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class TokenizerConfig:
    model_type: str = "bpe"
    vocab_size: int = 16000
    character_coverage: float = 1.0
    model_prefix: Path = Path("tokenizers/spm_pl_en")
    input_sentence_size: int = 2_000_000
    shuffle_input_sentence: bool = True
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"


@dataclass
class SplitTokenizerStats:
    split: str
    rows: int
    whitespace_words: int
    subword_pieces: int
    unk_pieces: int
    mean_pieces_per_sentence: float
    p95_pieces_per_sentence: int
    compression_ratio: float
    unk_rate: float


def iter_parallel_rows(path: Path) -> Iterator[tuple[str, str]]:
    table = pq.read_table(path)
    names = table.column_names
    if "translation" in names:
        for row in table["translation"].to_pylist():
            yield str(row.get("pl", "")), str(row.get("en", ""))
        return
    if "pl" not in names or "en" not in names:
        raise ValueError(f"Expected columns 'pl' and 'en' or 'translation' in {path}.")
    for pl_text, en_text in zip(table["pl"].to_pylist(), table["en"].to_pylist(), strict=True):
        yield str(pl_text), str(en_text)


def write_training_corpus(train_file: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for pl_text, en_text in iter_parallel_rows(train_file):
            f.write(pl_text.replace("\n", " ") + "\n")
            f.write(en_text.replace("\n", " ") + "\n")
            line_count += 2
    return line_count


def train_sentencepiece(input_path: Path, config: TokenizerConfig) -> None:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Missing dependency: install sentencepiece to train Stage 3 tokenizer.") from exc

    config.model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(config.model_prefix),
        model_type=config.model_type,
        vocab_size=config.vocab_size,
        character_coverage=config.character_coverage,
        input_sentence_size=config.input_sentence_size,
        shuffle_input_sentence=config.shuffle_input_sentence,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=config.pad_token,
        unk_piece=config.unk_token,
        bos_piece=config.bos_token,
        eos_piece=config.eos_token,
    )


def load_sentencepiece_encoder(model_path: Path) -> tuple[Callable[[str], list[int]], int]:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Missing dependency: install sentencepiece to inspect Stage 3 tokenizer.") from exc

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    return processor.encode, int(processor.unk_id())


def collect_word_frequencies(train_file: Path) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for pl_text, en_text in iter_parallel_rows(train_file):
        for text in (pl_text, en_text):
            frequencies.update(word.lower() for word in WORD_RE.findall(text))
    return frequencies


def _percentile95(values: list[int]) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, int(len(values) * 0.95))
    return values[index]


def collect_split_stats(
    split: str,
    split_file: Path,
    encode: Callable[[str], list[int]],
    unk_id: int,
) -> SplitTokenizerStats:
    rows = 0
    whitespace_words = 0
    subword_pieces = 0
    unk_pieces = 0
    sentence_piece_counts: list[int] = []

    for pl_text, en_text in iter_parallel_rows(split_file):
        rows += 1
        for text in (pl_text, en_text):
            pieces = encode(text)
            piece_count = len(pieces)
            sentence_piece_counts.append(piece_count)
            subword_pieces += piece_count
            unk_pieces += sum(1 for piece_id in pieces if piece_id == unk_id)
            whitespace_words += len(text.split())

    sentence_count = len(sentence_piece_counts)
    return SplitTokenizerStats(
        split=split,
        rows=rows,
        whitespace_words=whitespace_words,
        subword_pieces=subword_pieces,
        unk_pieces=unk_pieces,
        mean_pieces_per_sentence=(subword_pieces / sentence_count) if sentence_count else 0.0,
        p95_pieces_per_sentence=_percentile95(sentence_piece_counts),
        compression_ratio=(subword_pieces / whitespace_words) if whitespace_words else 0.0,
        unk_rate=(unk_pieces / subword_pieces) if subword_pieces else 0.0,
    )


def collect_long_word_split_stats(
    frequencies: Counter[str],
    encode: Callable[[str], list[int]],
    min_length: int = 12,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for word, count in frequencies.items():
        if len(word) < min_length:
            continue
        piece_count = len(encode(word))
        if count == 1:
            bucket = "freq_1"
        elif count <= 5:
            bucket = "freq_2_5"
        elif count <= 50:
            bucket = "freq_6_50"
        else:
            bucket = "freq_gt_50"
        buckets[bucket].append(piece_count)

    output: dict[str, dict[str, float]] = {}
    for bucket in ("freq_1", "freq_2_5", "freq_6_50", "freq_gt_50"):
        values = buckets.get(bucket, [])
        output[bucket] = {
            "word_types": float(len(values)),
            "mean_pieces_per_word": (sum(values) / len(values)) if values else 0.0,
            "p95_pieces_per_word": float(_percentile95(values)),
        }
    return output


def common_long_word_examples(
    frequencies: Counter[str],
    encode: Callable[[str], list[int]],
    min_length: int = 12,
    limit: int = 30,
) -> list[tuple[str, int, int]]:
    examples: list[tuple[str, int, int]] = []
    for word, count in frequencies.most_common():
        if len(word) >= min_length:
            examples.append((word, count, len(encode(word))))
        if len(examples) >= limit:
            break
    return examples


def write_tokenizer_stats_markdown(
    report_path: Path,
    config: TokenizerConfig,
    corpus_lines: int,
    split_stats: list[SplitTokenizerStats],
    long_word_stats: dict[str, dict[str, float]],
    common_long_words: list[tuple[str, int, int]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tokenizer Report (Stage 3)",
        "",
        f"Generated at (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"Model type: `{config.model_type}`",
        f"Vocab size: `{config.vocab_size}`",
        f"Character coverage: `{config.character_coverage}`",
        f"Training corpus lines: `{corpus_lines}`",
        "",
        "## Split Stats",
        "",
        "| Split | Rows | Whitespace words | Subword pieces | Compression | Mean pieces/sentence | P95 pieces/sentence | UNK rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stat in split_stats:
        lines.append(
            "| "
            f"{stat.split} | {stat.rows} | {stat.whitespace_words} | {stat.subword_pieces} | "
            f"{stat.compression_ratio:.3f} | {stat.mean_pieces_per_sentence:.2f} | "
            f"{stat.p95_pieces_per_sentence} | {stat.unk_rate:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Long Word Split Stats",
            "",
            "Long word means length >= 12. Desired pattern: common long words split less than rare long words.",
            "",
            "| Frequency bucket | Word types | Mean pieces/word | P95 pieces/word |",
            "|---|---:|---:|---:|",
        ]
    )
    for bucket, values in long_word_stats.items():
        lines.append(
            f"| {bucket} | {int(values['word_types'])} | "
            f"{values['mean_pieces_per_word']:.2f} | {values['p95_pieces_per_word']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Common Long Word Examples",
            "",
            "| Word | Train frequency | Pieces |",
            "|---|---:|---:|",
        ]
    )
    for word, count, pieces in common_long_words:
        lines.append(f"| `{word}` | {count} | {pieces} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
