from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from datasets import Dataset, load_dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("Missing dependency: datasets. Run: python -m pip install -r requirements.txt") from exc

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


WHITESPACE_RE = re.compile(r"\s+")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
DEFAULT_BICLEANER_MODEL = "bitextor/bicleaner-ai-full-en-pl"
BICLEANER_VENV_DIR = PROJECT_ROOT / ".venv-bicleaner"
DATASET = "Helsinki-NLP/tatoeba_mt_train"
CONFIG = "eng-pol"
SPLIT = "train"
UNICODE_NORMALIZATION = "NFKC"
MAX_TOKENS = 128


@dataclass(frozen=True)
class CleanConfig:
    raw_dir: Path
    out_dir: Path
    work_dir: Path
    threshold: float
    max_rows: int
    min_tokens: int
    max_tokens: int
    max_length_ratio: float
    clean_batch_size: int
    source_range: tuple[int, int] | None
    bicleaner_model: str
    bicleaner_require_gpu: bool
    skip_bicleaner: bool
    overwrite: bool
    skip_existing_shards: bool


@dataclass(frozen=True)
class Schema:
    kind: str
    en_field: str
    pl_field: str
    column_names: list[str]
    first_row: dict[str, Any]


@dataclass(frozen=True)
class Pair:
    pl: str
    en: str
    score: float | None = None
    source_parquet: str = ""
    source_row_index: int = -1


def parse_args() -> CleanConfig:
    parser = argparse.ArgumentParser(description="Clean Tatoeba eng-pol data for PL -> EN training.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/tatoeba"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/tatoeba-en-pl"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/work/tatoeba"))
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--max-length-ratio", type=float, default=2.5)
    parser.add_argument("--clean-batch-size", type=int, default=50_000)
    parser.add_argument("--source-range", type=parse_source_range, default=None, metavar="START-END", help="Only clean local raw Parquet files in zero-based inclusive range.")
    parser.add_argument("--bicleaner-model", default=DEFAULT_BICLEANER_MODEL)
    parser.add_argument("--bicleaner-require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-bicleaner", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing-shards", action="store_true", help="Skip raw Parquet files whose output shard already exists.")
    args = parser.parse_args()

    if args.max_rows < 0:
        parser.error("--max-rows must be >= 0")
    if args.min_tokens < 0:
        parser.error("--min-tokens must be >= 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.max_length_ratio < 1.0:
        parser.error("--max-length-ratio must be >= 1.0")
    if args.clean_batch_size <= 0:
        parser.error("--clean-batch-size must be > 0")

    return CleanConfig(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        work_dir=args.work_dir,
        threshold=args.threshold,
        max_rows=args.max_rows,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_length_ratio=args.max_length_ratio,
        clean_batch_size=args.clean_batch_size,
        source_range=args.source_range,
        bicleaner_model=args.bicleaner_model,
        bicleaner_require_gpu=args.bicleaner_require_gpu,
        skip_bicleaner=args.skip_bicleaner,
        overwrite=args.overwrite,
        skip_existing_shards=args.skip_existing_shards,
    )


def parse_source_range(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split("-", 1)
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range must use START-END, e.g. 0-10") from exc
    if start < 0:
        raise argparse.ArgumentTypeError("range start must be at least 0")
    if end < start:
        raise argparse.ArgumentTypeError("range end must be greater than or equal to start")
    return start, end


def source_index(path: Path) -> int:
    match = re.search(r"-(\d+)-of-\d+\.parquet$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse source index from {path.name}")
    return int(match.group(1))


def shard_stem(path: Path, split: str) -> str:
    return f"{source_index(path):05d}.{split}"


def list_local_sources(cfg: CleanConfig) -> list[Path]:
    local_dir = cfg.raw_dir / CONFIG / SPLIT
    sources = sorted(local_dir.glob("*.parquet")) if local_dir.exists() else []
    if cfg.source_range is not None:
        start, end = cfg.source_range
        sources = [path for path in sources if start <= source_index(path) <= end]
    if not sources:
        raise FileNotFoundError(f"No local Parquet files found in {local_dir}")
    if cfg.skip_existing_shards:
        sources = [path for path in sources if not (cfg.out_dir / f"{shard_stem(path, SPLIT)}.parquet").exists()]
    if not sources:
        raise FileExistsError("All selected output shards already exist.")
    return sources


def load_source(path: Path, split: str) -> Dataset:
    dataset = load_dataset("parquet", data_files={split: str(path)}, split=split)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected datasets.Dataset, got {type(dataset)!r}")
    return dataset


def short_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        text = repr(value)
        out[key] = text[:300] + ("..." if len(text) > 300 else "")
    return out


def detect_schema(dataset: Dataset) -> Schema:
    if len(dataset) == 0:
        raise ValueError("Dataset split empty.")
    first_row = dict(dataset[0])
    columns = list(dataset.column_names)
    translation = first_row.get("translation")
    if isinstance(translation, dict):
        keys = set(translation.keys())
        if {"en", "pl"}.issubset(keys):
            return Schema("translation", "en", "pl", columns, short_row(first_row))
        if {"eng", "pol"}.issubset(keys):
            return Schema("translation", "eng", "pol", columns, short_row(first_row))
    for en_field, pl_field in [
        ("source_text", "target_text"),
        ("sourceString", "targetString"),
        ("source", "target"),
        ("src", "trg"),
        ("sentence1", "sentence2"),
    ]:
        if en_field in columns and pl_field in columns:
            return Schema("columns", en_field, pl_field, columns, short_row(first_row))
    raise ValueError(f"Unknown schema. Columns: {columns}. First row: {short_row(first_row)}")


def normalize(text: str, cfg: CleanConfig) -> str:
    out = unicodedata.normalize(UNICODE_NORMALIZATION, text)
    out = CONTROL_CHAR_RE.sub("", out)
    out = WHITESPACE_RE.sub(" ", out)
    out = out.strip()
    return out


def extract_pair(row: dict[str, Any], schema: Schema, cfg: CleanConfig) -> Pair:
    if schema.kind == "translation":
        translation = row.get("translation")
        if not isinstance(translation, dict):
            return Pair("", "")
        en = translation.get(schema.en_field, "")
        pl = translation.get(schema.pl_field, "")
    else:
        en = row.get(schema.en_field, "")
        pl = row.get(schema.pl_field, "")
    return Pair(pl=normalize(str(pl), cfg), en=normalize(str(en), cfg))


def token_count(text: str) -> int:
    return len(text.split())


def mostly_punctuation(text: str) -> bool:
    non_space = sum(1 for char in text if not char.isspace())
    if non_space == 0:
        return True
    alnum = sum(1 for char in text if char.isalnum())
    return alnum / non_space < 0.35


def reject_reason(pair: Pair, cfg: CleanConfig) -> str | None:
    if not pair.pl or not pair.en:
        return "empty"
    pl_tokens = token_count(pair.pl)
    en_tokens = token_count(pair.en)
    if pl_tokens < cfg.min_tokens or en_tokens < cfg.min_tokens:
        return "min_tokens"
    if pl_tokens > cfg.max_tokens or en_tokens > cfg.max_tokens:
        return "max_tokens"
    if max(pl_tokens, en_tokens) / max(1, min(pl_tokens, en_tokens)) > cfg.max_length_ratio:
        return "length_ratio"
    combined = f"{pair.pl}\n{pair.en}"
    if URL_RE.search(combined):
        return "url"
    if HTML_TAG_RE.search(combined) or "&nbsp;" in combined.lower():
        return "html"
    if mostly_punctuation(pair.pl) or mostly_punctuation(pair.en):
        return "mostly_punctuation"
    return None


def basic_clean_source(source: Path, cfg: CleanConfig, seen: set[tuple[str, str]], max_rows: int | None = None) -> tuple[list[Pair], dict[str, int], Schema]:
    dataset = load_source(source, SPLIT)
    schema = detect_schema(dataset)
    counts = {"raw_rows": len(dataset), "after_basic_clean": 0, "after_dedup": 0, "after_max_rows": 0, "dropped_duplicate": 0}
    dropped: dict[str, int] = {}
    deduped: list[Pair] = []
    iterator = dataset.iter(batch_size=cfg.clean_batch_size)
    total_batches = (len(dataset) + cfg.clean_batch_size - 1) // cfg.clean_batch_size
    if tqdm is not None:
        iterator = tqdm(iterator, total=total_batches, desc=f"Clean {source.name}", unit="batch")
    row_index = 0
    for batch in iterator:
        if schema.kind == "translation":
            translations = batch["translation"]
            en_values = [item.get(schema.en_field, "") if isinstance(item, dict) else "" for item in translations]
            pl_values = [item.get(schema.pl_field, "") if isinstance(item, dict) else "" for item in translations]
        else:
            en_values = batch[schema.en_field]
            pl_values = batch[schema.pl_field]

        for en_raw, pl_raw in zip(en_values, pl_values, strict=True):
            pair = Pair(pl=normalize(str(pl_raw), cfg), en=normalize(str(en_raw), cfg), source_parquet=source.name, source_row_index=row_index)
            row_index += 1
            reason = reject_reason(pair, cfg)
            if reason:
                dropped[reason] = dropped.get(reason, 0) + 1
                counts[f"dropped_{reason}"] = counts.get(f"dropped_{reason}", 0) + 1
                continue
            counts["after_basic_clean"] += 1
            key = (pair.pl.lower(), pair.en.lower())
            if key in seen:
                dropped["duplicate"] = dropped.get("duplicate", 0) + 1
                counts["dropped_duplicate"] += 1
                continue
            seen.add(key)
            deduped.append(pair)
    counts["after_dedup"] = len(deduped)

    if max_rows is not None:
        deduped = deduped[:max_rows]
    counts["after_max_rows"] = len(deduped)
    counts.update({f"dropped_{key}": value for key, value in sorted(dropped.items())})
    return deduped, counts, schema


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite without --overwrite: " + ", ".join(str(path) for path in existing))


def write_candidates(pairs: list[Pair], path: Path, overwrite: bool) -> None:
    meta_path = path.with_suffix(".meta.jsonl")
    ensure_writable([path, meta_path], overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle, meta_path.open("w", encoding="utf-8", newline="\n") as meta_handle:
        for pair in pairs:
            handle.write(f"dummy_en\tdummy_pl\t{pair.en}\t{pair.pl}\n")
            meta_handle.write(json.dumps({"source_parquet": pair.source_parquet, "source_row_index": pair.source_row_index}, ensure_ascii=True) + "\n")


def read_candidates(path: Path, cfg: CleanConfig) -> list[Pair]:
    pairs: list[Pair] = []
    meta_path = path.with_suffix(".meta.jsonl")
    metadata: list[dict[str, Any]] = []
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = [json.loads(line) for line in handle]
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(f"Cannot recover EN/PL at {path}:{line_number}")
            meta = metadata[line_number - 1] if line_number <= len(metadata) else {}
            pairs.append(Pair(pl=normalize(parts[3], cfg), en=normalize(parts[2], cfg), score=1.0, source_parquet=str(meta.get("source_parquet", "")), source_row_index=int(meta.get("source_row_index", -1))))
    return pairs


def resolve_bicleaner_executable() -> Path | str:
    candidates = [BICLEANER_VENV_DIR / "Scripts" / "bicleaner-ai-classify.exe"]
    if os.name != "nt":
        candidates.append(BICLEANER_VENV_DIR / "bin" / "bicleaner-ai-classify")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    executable = shutil.which("bicleaner-ai-classify")
    if executable is not None:
        return executable
    raise RuntimeError("bicleaner-ai-classify not found. Create .venv-bicleaner with scripts/setup_bicleaner_venv.sh, or use --skip-bicleaner.")


def bicleaner_env(executable: Path | str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    executable_path = Path(executable)
    if BICLEANER_VENV_DIR in executable_path.parents:
        bin_dir = executable_path.parent
        env["VIRTUAL_ENV"] = str(BICLEANER_VENV_DIR)
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")

        site_packages = sorted((BICLEANER_VENV_DIR / "lib").glob("python*/site-packages"))
        if site_packages:
            nvidia_libs = [
                "cuda_runtime/lib",
                "cublas/lib",
                "cuda_cupti/lib",
                "cudnn/lib",
                "cufft/lib",
                "curand/lib",
                "cusolver/lib",
                "cusparse/lib",
                "nccl/lib",
                "nvjitlink/lib",
            ]
            cuda_paths = [str(site_packages[0] / "nvidia" / lib) for lib in nvidia_libs]
            env["LD_LIBRARY_PATH"] = os.pathsep.join(cuda_paths + [env.get("LD_LIBRARY_PATH", "")])
    return env


def run_bicleaner(candidates_path: Path, scored_path: Path, cfg: CleanConfig) -> None:
    executable = resolve_bicleaner_executable()
    scored_path.unlink(missing_ok=True)
    model_path = ensure_bicleaner_model(cfg.bicleaner_model)
    command = [
        str(executable),
        "-s",
        "en",
        "-t",
        "pl",
        "--scol",
        "3",
        "--tcol",
        "4",
        "--disable_hardrules",
    ]
    if cfg.bicleaner_require_gpu:
        command.append("--require_gpu")
    command.append("--mixed_precision")
    command.extend([str(candidates_path), str(scored_path), str(model_path)])
    total_rows = count_lines(candidates_path)
    log_path = cfg.work_dir / "bicleaner.log"
    env = bicleaner_env(executable)
    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        process = subprocess.Popen(command, text=True, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        if tqdm is not None and total_rows > 0:
            with tqdm(total=total_rows, desc="Bicleaner", unit="rows", mininterval=10.0) as progress:
                last_rows = 0
                while process.poll() is None:
                    current_rows = count_lines(scored_path) if scored_path.exists() else 0
                    progress.update(max(0, current_rows - last_rows))
                    last_rows = current_rows
                    time.sleep(10.0)
                current_rows = count_lines(scored_path) if scored_path.exists() else last_rows
                progress.update(max(0, current_rows - last_rows))
                if process.returncode == 0 and progress.n < total_rows:
                    progress.update(total_rows - progress.n)
        else:
            process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Bicleaner failed. Command: {' '.join(command)}\nLog: {log_path}\n{tail_text(log_path)}")


def ensure_bicleaner_model(model: str) -> Path | str:
    model_path = Path(model)
    if model_path.exists():
        return model_path
    if "/" not in model:
        return model

    local_dir = PROJECT_ROOT / "models" / model
    complete_marker = local_dir / ".download_complete"
    if complete_marker.exists():
        return local_dir

    executable = shutil.which("hf") or shutil.which("huggingface-cli")
    if executable is None:
        raise RuntimeError(f"Bicleaner model missing: {local_dir}. Install Hugging Face CLI or download {model} manually.")

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(load_dotenv_values(PROJECT_ROOT / ".env"))
    command = [executable, "download", model, "--local-dir", str(local_dir)]
    print(f"Downloading Bicleaner model: {model} -> {local_dir}")
    result = subprocess.run(command, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download Bicleaner model. Command: {' '.join(command)}")
    complete_marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return local_dir


def load_dotenv_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def last_float(parts: list[str]) -> float | None:
    for value in reversed(parts):
        try:
            return float(value)
        except ValueError:
            continue
    return None


def parse_scored(path: Path, cfg: CleanConfig, candidates: list[Pair]) -> list[Pair]:
    kept: list[Pair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(f"Cannot recover EN/PL at {path}:{line_number}")
            score = last_float(parts)
            if score is None:
                raise ValueError(f"No score at {path}:{line_number}")
            if score >= cfg.threshold:
                source_pair = candidates[line_number - 1]
                kept.append(Pair(pl=normalize(parts[3], cfg), en=normalize(parts[2], cfg), score=score, source_parquet=source_pair.source_parquet, source_row_index=source_pair.source_row_index))
    return kept


def pair_schema() -> pa.Schema:
    return pa.schema(
        [
            ("pl", pa.string()),
            ("en", pa.string()),
            ("score", pa.float64()),
            ("source_parquet", pa.string()),
            ("source_row_index", pa.int64()),
        ]
    )


def write_shard(pairs: list[Pair], cfg: CleanConfig, schema: Schema, counts: dict[str, int], source: Path) -> dict[str, Any]:
    index = source_index(source)
    shard_path = cfg.out_dir / f"{index:05d}.{SPLIT}.parquet"
    manifest_path = cfg.out_dir / f"{index:05d}.{SPLIT}.manifest.json"
    ensure_writable([shard_path, manifest_path], cfg.overwrite)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "pl": pair.pl,
                "en": pair.en,
                "score": 1.0 if pair.score is None else pair.score,
                "source_parquet": pair.source_parquet,
                "source_row_index": pair.source_row_index,
            }
            for pair in pairs
        ],
        schema=pair_schema(),
    )
    pq.write_table(table, shard_path, compression="zstd")
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "source_parquet": source.name,
        "source_stats": counts,
        "rows_out": len(pairs),
        "columns": ["pl", "en", "score", "source_parquet", "source_row_index"],
        "compression": "zstd",
        "threshold": cfg.threshold,
        "skip_bicleaner": cfg.skip_bicleaner,
        "schema": asdict(schema),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"path": str(shard_path), "manifest": str(manifest_path), "source_parquet": source.name, "rows": len(pairs)}


def write_run_manifest(cfg: CleanConfig, sources: list[Path], written_shards: list[dict[str, Any]], counts: dict[str, int], schema: Schema | None) -> None:
    manifest_path = cfg.out_dir / f"manifest.{work_file_stem(cfg)}.json"
    ensure_writable([manifest_path], cfg.overwrite)
    stats = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "raw_dir": str(cfg.raw_dir),
        "out_dir": str(cfg.out_dir),
        "work_dir": str(cfg.work_dir),
        "threshold": cfg.threshold,
        "min_tokens": cfg.min_tokens,
        "max_tokens": cfg.max_tokens,
        "max_length_ratio": cfg.max_length_ratio,
        "clean_batch_size": cfg.clean_batch_size,
        "skip_bicleaner": cfg.skip_bicleaner,
        "bicleaner_require_gpu": cfg.bicleaner_require_gpu,
        "bicleaner_mixed_precision": True,
        "source_parquets": [path.name for path in sources],
        "after_bicleaner": sum(shard["rows"] for shard in written_shards),
        "written_shards": written_shards,
        "schema": asdict(schema) if schema is not None else None,
        "counts": counts,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True), encoding="utf-8")


def work_file_stem(cfg: CleanConfig) -> str:
    if cfg.source_range is None:
        return "en-pl"
    start, end = cfg.source_range
    return f"en-pl.{start:05d}-{end:05d}"


def work_file_stem_for_source(source: Path) -> str:
    return f"en-pl.{source_index(source):05d}"


def add_counts(total: dict[str, int], counts: dict[str, int]) -> None:
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value


def seed_seen_from_existing_shards(cfg: CleanConfig, sources: list[Path]) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    if not cfg.skip_existing_shards:
        return seen
    for source in sources:
        shard_path = cfg.out_dir / f"{source_index(source):05d}.{SPLIT}.parquet"
        if not shard_path.exists():
            continue
        table = pq.read_table(shard_path, columns=["pl", "en"])
        data = table.to_pydict()
        for pl, en in zip(data["pl"], data["en"], strict=True):
            seen.add((str(pl).lower(), str(en).lower()))
        print(f"Seeded dedup from existing shard: {shard_path} ({len(data['pl'])} rows)")
    return seen


def main() -> int:
    cfg = parse_args()
    sources = list_local_sources(cfg)
    if not cfg.skip_bicleaner:
        resolve_bicleaner_executable()

    print(f"Source Parquet files: {len(sources)}")
    all_sources = sorted((cfg.raw_dir / CONFIG / SPLIT).glob("*.parquet"))
    if cfg.source_range is not None:
        start, end = cfg.source_range
        all_sources = [path for path in all_sources if start <= source_index(path) <= end]
    seen = seed_seen_from_existing_shards(cfg, all_sources)
    total_counts: dict[str, int] = {"raw_rows": 0, "after_basic_clean": 0, "after_dedup": 0, "after_max_rows": 0}
    written_shards: list[dict[str, Any]] = []
    detected_schema: Schema | None = None
    remaining = cfg.max_rows if cfg.max_rows > 0 else None

    for source in sources:
        if remaining is not None and remaining <= 0:
            break
        source_stem = work_file_stem_for_source(source)
        candidates_path = cfg.work_dir / f"candidates.{source_stem}.tsv"
        scored_path = cfg.work_dir / f"scored.{source_stem}.tsv"
        print(f"Processing source: {source.name}")
        pairs, counts, schema = basic_clean_source(source, cfg, seen, remaining)
        detected_schema = detected_schema or schema
        print(f"Schema: {schema.kind}, EN={schema.en_field}, PL={schema.pl_field}")
        print(f"Rows raw: {counts['raw_rows']}")
        print(f"Rows after basic clean: {counts['after_basic_clean']}")
        print(f"Rows after dedup: {counts['after_dedup']}")
        print(f"Rows after max rows: {counts['after_max_rows']}")
        write_candidates(pairs, candidates_path, overwrite=True)
        print(f"Wrote candidates: {candidates_path}")

        if cfg.skip_bicleaner:
            final_pairs = [Pair(pair.pl, pair.en, 1.0, pair.source_parquet, pair.source_row_index) for pair in pairs]
        else:
            run_bicleaner(candidates_path, scored_path, cfg)
            final_pairs = parse_scored(scored_path, cfg, pairs)
        print(f"Rows final: {len(final_pairs)}")
        written_shards.append(write_shard(final_pairs, cfg, schema, counts, source))
        add_counts(total_counts, counts)
        if remaining is not None:
            remaining -= len(pairs)

    write_run_manifest(cfg, sources, written_shards, total_counts, detected_schema)
    print(f"Wrote outputs: {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
