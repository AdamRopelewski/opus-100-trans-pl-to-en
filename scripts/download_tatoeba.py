from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ParquetRange = tuple[int, int]


@dataclass(frozen=True)
class DownloadConfig:
    dataset: str
    config: str
    split: str
    raw_dir: Path
    force: bool
    max_parquet_files: int | None
    parquet_range: ParquetRange | None


def parse_parquet_range(value: str) -> ParquetRange:
    try:
        start_text, end_text = value.split("-", 1)
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range must use START-END, e.g. 1-10") from exc
    if start < 0:
        raise argparse.ArgumentTypeError("range start must be at least 0")
    if end < start:
        raise argparse.ArgumentTypeError("range end must be greater than or equal to start")
    return start, end


def parse_args() -> DownloadConfig:
    parser = argparse.ArgumentParser(description="Download Tatoeba eng-pol dataset to local raw data directory.")
    parser.add_argument("--dataset", default="Helsinki-NLP/tatoeba_mt_train")
    parser.add_argument("--config", default="eng-pol")
    parser.add_argument("--split", default="train")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/tatoeba"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing local dataset directory.")
    parser.add_argument(
        "--max-parquet-files",
        "--max-parts",
        type=int,
        default=None,
        help="Download only first N Parquet files from the dataset split.",
    )
    parser.add_argument(
        "--parquet-range",
        "--part-range",
        type=parse_parquet_range,
        default=None,
        metavar="START-END",
        help="Download zero-based inclusive Parquet file range, e.g. 1-10 skips file 0 and downloads 10 files.",
    )
    args = parser.parse_args()
    if args.max_parquet_files is not None and args.max_parquet_files < 1:
        parser.error("--max-parquet-files must be at least 1")
    if args.max_parquet_files is not None and args.parquet_range is not None:
        parser.error("--max-parquet-files and --parquet-range cannot be used together")
    return DownloadConfig(**vars(args))


def load_dataset_func():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: datasets. Run: python -m pip install -r requirements.txt") from exc
    return load_dataset


def list_parquet_files(cfg: DownloadConfig) -> list[str]:
    try:
        from huggingface_hub import list_repo_files
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: huggingface_hub. Run: python -m pip install -r requirements.txt") from exc

    prefix = f"{cfg.config}/"
    parquet_files = sorted(
        path
        for path in list_repo_files(cfg.dataset, repo_type="dataset")
        if path.startswith(prefix) and path.endswith(".parquet") and Path(path).name.startswith(f"{cfg.split}-")
    )
    if not parquet_files:
        raise RuntimeError(f"No Parquet files found for {cfg.dataset}/{cfg.config}/{cfg.split}")
    return parquet_files


def download_parquet_files(cfg: DownloadConfig, dataset_dir: Path) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: huggingface_hub. Run: python -m pip install -r requirements.txt") from exc

    parquet_files = list_parquet_files(cfg)
    if cfg.parquet_range is not None:
        start, end = cfg.parquet_range
        selected_files = parquet_files[start : end + 1]
    elif cfg.max_parquet_files is not None:
        selected_files = parquet_files[: cfg.max_parquet_files]
    else:
        selected_files = parquet_files
    if not selected_files:
        raise RuntimeError(f"No Parquet files selected from {len(parquet_files)} available files")
    local_files = []
    for parquet_file in selected_files:
        local_path = dataset_dir / Path(parquet_file).name
        if not local_path.exists():
            cached_path = hf_hub_download(repo_id=cfg.dataset, repo_type="dataset", filename=parquet_file)
            shutil.copy2(cached_path, local_path)
        local_files.append(local_path)
    return local_files


def write_metadata(cfg: DownloadConfig, dataset_dir: Path, rows: int, parquet_files: Sequence[str]) -> None:
    metadata = {
        "dataset": cfg.dataset,
        "config": cfg.config,
        "split": cfg.split,
        "rows": rows,
        "max_parquet_files": cfg.max_parquet_files,
        "parquet_range": list(cfg.parquet_range) if cfg.parquet_range is not None else None,
        "parquet_files": list(parquet_files),
        "dataset_dir": str(dataset_dir),
        "python": sys.executable,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    cfg = parse_args()
    dataset_dir = cfg.raw_dir / cfg.config / cfg.split
    if cfg.force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    load_dataset = load_dataset_func()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    download_parquet_files(cfg, dataset_dir)
    parquet_paths = sorted(dataset_dir.glob("*.parquet"))
    dataset = load_dataset("parquet", data_files={cfg.split: [str(path) for path in parquet_paths]}, split=cfg.split)
    write_metadata(cfg, dataset_dir, len(dataset), [path.name for path in parquet_paths])
    print(f"Downloaded rows: {len(dataset)}")
    print(f"Local Parquet files: {len(parquet_paths)}")
    print(f"Saved: {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
