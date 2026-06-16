from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.translation_dataset import SpecialTokenIds
from src.utils.config import get_nested, load_config
from src.utils.tokenizer import iter_parallel_rows


@dataclass(frozen=True)
class TokenizedSplitStats:
    split: str
    rows_in: int
    rows_out: int
    overlength_rows: int
    files_written: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize Tatoeba parquet splits into token-id parquet shards."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tatoeba_config.yaml"),
        help="Path to Tatoeba config YAML.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tokenized output directory files.",
    )
    return parser


def _load_sentencepiece(model_path: Path):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install sentencepiece.") from exc
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    return spm.SentencePieceProcessor(model_file=str(model_path))


def _resolve_split_file(data_dir: Path, split_name: str, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one parquet for split '{split_name}' in '{data_dir}' using '{pattern}', found {len(matches)}."
        )
    return matches[0]


def _write_rows(path: Path, rows: list[dict[str, list[int]]]) -> None:
    schema = pa.schema(
        [
            pa.field("src_ids", pa.list_(pa.int32())),
            pa.field("tgt_ids", pa.list_(pa.int32())),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def tokenize_split(
    split: str,
    input_path: Path,
    output_dir: Path,
    tokenizer,
    token_ids: SpecialTokenIds,
    max_seq_len: int,
    drop_overlength: bool,
    shard_rows: int,
    overwrite: bool,
    progress: bool = True,
) -> TokenizedSplitStats:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    existing = list(split_dir.glob(f"{split}-tokenized-*.parquet"))
    if existing and not overwrite:
        raise FileExistsError(
            f"Tokenized files already exist in {split_dir}; pass --overwrite to replace them."
        )
    for path in existing:
        path.unlink()

    rows: list[dict[str, list[int]]] = []
    rows_in = 0
    rows_out = 0
    overlength_rows = 0
    files_written = 0
    written_paths: list[Path] = []
    total_rows = int(pq.read_metadata(input_path).num_rows)

    def flush() -> None:
        nonlocal rows, files_written, written_paths
        if not rows:
            return
        output_path = split_dir / f"{split}-tokenized-{files_written + 1:05d}.parquet"
        _write_rows(output_path, rows)
        written_paths.append(output_path)
        files_written += 1
        rows = []

    row_iter = tqdm(
        iter_parallel_rows(input_path),
        total=total_rows,
        desc=f"tokenize {split}",
        unit="row",
        disable=not progress,
    )
    def update_progress() -> None:
        row_iter.set_postfix(out=rows_out, overlength=overlength_rows, shards=files_written)

    for pl_text, en_text in row_iter:
        rows_in += 1
        src_ids = tokenizer.encode(pl_text) + [token_ids.eos_id]
        tgt_ids = tokenizer.encode(en_text)
        if max(len(src_ids), len(tgt_ids) + 1) > max_seq_len:
            overlength_rows += 1
            if drop_overlength:
                if rows_in % 1000 == 0:
                    update_progress()
                continue
        rows.append(
            {
                "src_ids": src_ids,
                "tgt_ids": tgt_ids,
            }
        )
        rows_out += 1
        if len(rows) >= shard_rows:
            flush()
            update_progress()
        elif rows_in % 1000 == 0:
            update_progress()
    flush()
    update_progress()
    _rename_with_total_shards(split, written_paths)
    return TokenizedSplitStats(split, rows_in, rows_out, overlength_rows, files_written)


def _rename_with_total_shards(split: str, paths: list[Path]) -> None:
    total = len(paths)
    width = max(5, len(str(total)))
    for index, path in enumerate(paths, start=1):
        final_path = path.with_name(
            f"{split}-tokenized-{index:0{width}d}-of-{total:0{width}d}.parquet"
        )
        path.rename(final_path)


def run(args: argparse.Namespace) -> list[TokenizedSplitStats]:
    config = load_config(args.config)
    processed_dir = Path(
        get_nested(
            config, "stage2_cleaning.outputs.processed_dir", "data/processed/tatoeba-en-pl/splits"
        )
    )
    output_dir = Path(
        get_nested(
            config,
            "stage4_dataloader.tokenized_splits_dir",
            "data/processed/tatoeba-en-pl/tokenized",
        )
    )
    tokenizer_prefix = Path(
        get_nested(config, "stage3_tokenizer.model_prefix", "tokenizers/tatoeba_spm_pl_en")
    )
    tokenizer = _load_sentencepiece(tokenizer_prefix.with_suffix(".model"))
    token_ids = SpecialTokenIds(
        pad_id=int(tokenizer.pad_id()),
        unk_id=int(tokenizer.unk_id()),
        bos_id=int(tokenizer.bos_id()),
        eos_id=int(tokenizer.eos_id()),
    )
    if token_ids != SpecialTokenIds():
        raise ValueError(f"Unexpected tokenizer special token ids: {token_ids}")

    split_patterns = {
        "train": str(get_nested(config, "dataset.splits.train_pattern", "train-*.parquet")),
        "validation": str(
            get_nested(config, "dataset.splits.validation_pattern", "validation-*.parquet")
        ),
        "test": str(get_nested(config, "dataset.splits.test_pattern", "test-*.parquet")),
    }
    max_seq_len = int(get_nested(config, "stage4_dataloader.max_seq_len", 128))
    drop_overlength = bool(get_nested(config, "stage4_dataloader.drop_overlength", True))
    shard_rows = int(get_nested(config, "stage4_dataloader.tokenized_shard_rows", 500000))

    stats = []
    for split in ("train", "validation", "test"):
        input_path = _resolve_split_file(processed_dir, split, split_patterns[split])
        split_stats = tokenize_split(
            split,
            input_path,
            output_dir,
            tokenizer,
            token_ids,
            max_seq_len,
            drop_overlength,
            shard_rows,
            args.overwrite,
        )
        stats.append(split_stats)
        print(
            f"Tokenized {split}: {split_stats.rows_out}/{split_stats.rows_in} rows; "
            f"overlength={split_stats.overlength_rows}; files={split_stats.files_written}"
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([asdict(item) for item in stats], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stats


def main() -> int:
    try:
        run(_build_parser().parse_args())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
