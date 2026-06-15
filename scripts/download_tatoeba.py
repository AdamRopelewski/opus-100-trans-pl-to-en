from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DownloadConfig:
    dataset: str
    config: str
    split: str
    raw_dir: Path
    force: bool
    max_parquet_files: int | None


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
    args = parser.parse_args()
    if args.max_parquet_files is not None and args.max_parquet_files < 1:
        parser.error("--max-parquet-files must be at least 1")
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
    selected_files = parquet_files[: cfg.max_parquet_files] if cfg.max_parquet_files is not None else parquet_files
    local_files = []
    for parquet_file in selected_files:
        cached_path = hf_hub_download(repo_id=cfg.dataset, repo_type="dataset", filename=parquet_file)
        local_path = dataset_dir / Path(parquet_file).name
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
        "parquet_files": list(parquet_files),
        "dataset_dir": str(dataset_dir),
        "python": sys.executable,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    cfg = parse_args()
    dataset_dir = cfg.raw_dir / cfg.config / cfg.split
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        if not cfg.force:
            print(f"Dataset already exists: {dataset_dir}")
            print("Use --force to overwrite.")
            return 0
        shutil.rmtree(dataset_dir)

    load_dataset = load_dataset_func()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = download_parquet_files(cfg, dataset_dir)
    dataset = load_dataset("parquet", data_files={cfg.split: [str(path) for path in parquet_paths]}, split=cfg.split)
    write_metadata(cfg, dataset_dir, len(dataset), [path.name for path in parquet_paths])
    print(f"Downloaded rows: {len(dataset)}")
    print(f"Downloaded Parquet files: {len(parquet_paths)}")
    print(f"Saved: {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
