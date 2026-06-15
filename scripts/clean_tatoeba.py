from __future__ import annotations

import argparse
import json
import os
import random
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_nested, load_config

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


@dataclass(frozen=True)
class CleanConfig:
    project_config: Path
    dataset: str
    config: str
    split: str
    raw_dir: Path
    out_dir: Path
    work_dir: Path
    threshold: float
    max_rows: int
    seed: int
    min_tokens: int
    max_tokens: int
    max_length_ratio: float
    clean_batch_size: int
    bicleaner_model: str
    bicleaner_require_gpu: bool
    bicleaner_mixed_precision: bool
    skip_bicleaner: bool
    reuse_candidates: bool
    overwrite: bool
    unicode_normalization: str
    strip_whitespace: bool
    collapse_whitespace: bool
    remove_control_chars: bool
    protected_processed_dir: Path


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


def parse_args() -> CleanConfig:
    parser = argparse.ArgumentParser(description="Clean Tatoeba eng-pol data for PL -> EN training.")
    parser.add_argument("--project-config", type=Path, default=Path("configs/project_config.yaml"))
    parser.add_argument("--dataset", default="Helsinki-NLP/tatoeba_mt_train")
    parser.add_argument("--config", default="eng-pol")
    parser.add_argument("--split", default="train")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/tatoeba"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("data/work/tatoeba"))
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-length-ratio", type=float, default=2.5)
    parser.add_argument("--clean-batch-size", type=int, default=50_000)
    parser.add_argument("--bicleaner-model", default=DEFAULT_BICLEANER_MODEL)
    parser.add_argument("--bicleaner-require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bicleaner-mixed-precision", action="store_true")
    parser.add_argument("--skip-bicleaner", action="store_true")
    parser.add_argument("--reuse-candidates", action="store_true", help="Skip dataset loading/basic clean and reuse work-dir/candidates.en-pl.tsv.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_cfg = load_config(args.project_config)
    protected_processed_dir = Path(get_nested(project_cfg, "paths.processed_data_dir", "data/processed/en-pl"))
    out_dir = args.out_dir or (protected_processed_dir.parent / "tatoeba-en-pl")
    if out_dir.resolve() == protected_processed_dir.resolve():
        parser.error("--out-dir must not equal current project processed dir. Use data/processed/tatoeba-en-pl.")

    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = int(get_nested(project_cfg, "stage4_dataloader.max_seq_len", 128))
    seed = args.seed
    if seed is None:
        seed = int(get_nested(project_cfg, "project.seed", 42))

    if args.max_rows < 0:
        parser.error("--max-rows must be >= 0")
    if args.min_tokens < 0:
        parser.error("--min-tokens must be >= 0")
    if max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.max_length_ratio < 1.0:
        parser.error("--max-length-ratio must be >= 1.0")
    if args.clean_batch_size <= 0:
        parser.error("--clean-batch-size must be > 0")

    return CleanConfig(
        project_config=args.project_config,
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        raw_dir=args.raw_dir,
        out_dir=out_dir,
        work_dir=args.work_dir,
        threshold=args.threshold,
        max_rows=args.max_rows,
        seed=seed,
        min_tokens=args.min_tokens,
        max_tokens=max_tokens,
        max_length_ratio=args.max_length_ratio,
        clean_batch_size=args.clean_batch_size,
        bicleaner_model=args.bicleaner_model,
        bicleaner_require_gpu=args.bicleaner_require_gpu,
        bicleaner_mixed_precision=args.bicleaner_mixed_precision,
        skip_bicleaner=args.skip_bicleaner,
        reuse_candidates=args.reuse_candidates,
        overwrite=args.overwrite,
        unicode_normalization=str(get_nested(project_cfg, "stage2_cleaning.filters.unicode_normalization", "NFKC")),
        strip_whitespace=bool(get_nested(project_cfg, "stage2_cleaning.filters.strip_whitespace", True)),
        collapse_whitespace=bool(get_nested(project_cfg, "stage2_cleaning.filters.collapse_whitespace", True)),
        remove_control_chars=bool(get_nested(project_cfg, "stage2_cleaning.filters.remove_control_chars", True)),
        protected_processed_dir=protected_processed_dir,
    )


def load_tatoeba(cfg: CleanConfig) -> Dataset:
    local_dir = cfg.raw_dir / cfg.config / cfg.split
    local_parquet = sorted(local_dir.glob("*.parquet")) if local_dir.exists() else []
    if local_parquet:
        dataset = load_dataset("parquet", data_files={cfg.split: [str(path) for path in local_parquet]}, split=cfg.split)
        print(f"Loaded local Parquet dataset: {local_dir}")
    else:
        dataset = load_dataset(cfg.dataset, cfg.config, split=cfg.split)
        print(f"Loaded Hugging Face dataset: {cfg.dataset}/{cfg.config}/{cfg.split}")
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
    out = unicodedata.normalize(cfg.unicode_normalization, text)
    if cfg.remove_control_chars:
        out = CONTROL_CHAR_RE.sub("", out)
    if cfg.collapse_whitespace:
        out = WHITESPACE_RE.sub(" ", out)
    if cfg.strip_whitespace:
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


def basic_clean(dataset: Dataset, schema: Schema, cfg: CleanConfig) -> tuple[list[Pair], dict[str, int]]:
    counts = {"raw_rows": len(dataset), "after_basic_clean": 0, "after_dedup": 0, "after_max_rows": 0}
    dropped: dict[str, int] = {}
    deduped: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    iterator = dataset.iter(batch_size=cfg.clean_batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(dataset) + cfg.clean_batch_size - 1) // cfg.clean_batch_size, desc="Clean Tatoeba", unit="batch")
    for batch in iterator:
        if schema.kind == "translation":
            translations = batch["translation"]
            en_values = [item.get(schema.en_field, "") if isinstance(item, dict) else "" for item in translations]
            pl_values = [item.get(schema.pl_field, "") if isinstance(item, dict) else "" for item in translations]
        else:
            en_values = batch[schema.en_field]
            pl_values = batch[schema.pl_field]

        for en_raw, pl_raw in zip(en_values, pl_values, strict=True):
            pair = Pair(pl=normalize(str(pl_raw), cfg), en=normalize(str(en_raw), cfg))
            reason = reject_reason(pair, cfg)
            if reason:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            counts["after_basic_clean"] += 1
            key = (pair.pl.lower(), pair.en.lower())
            if key in seen:
                dropped["duplicate"] = dropped.get("duplicate", 0) + 1
                continue
            seen.add(key)
            deduped.append(pair)
    counts["after_dedup"] = len(deduped)

    random.Random(cfg.seed).shuffle(deduped)
    if cfg.max_rows > 0:
        deduped = deduped[: cfg.max_rows]
    counts["after_max_rows"] = len(deduped)
    counts.update({f"dropped_{key}": value for key, value in sorted(dropped.items())})
    return deduped, counts


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite without --overwrite: " + ", ".join(str(path) for path in existing))


def write_candidates(pairs: list[Pair], path: Path, overwrite: bool) -> None:
    ensure_writable([path], overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(f"dummy_en\tdummy_pl\t{pair.en}\t{pair.pl}\n")


def read_candidates(path: Path, cfg: CleanConfig) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(f"Cannot recover EN/PL at {path}:{line_number}")
            pairs.append(Pair(pl=normalize(parts[3], cfg), en=normalize(parts[2], cfg), score=1.0))
    return pairs


def resolve_bicleaner_executable() -> Path | str:
    candidates = [
        BICLEANER_VENV_DIR / "bin" / "bicleaner-ai-classify",
        BICLEANER_VENV_DIR / "Scripts" / "bicleaner-ai-classify.exe",
    ]
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
    ensure_writable([scored_path], cfg.overwrite)
    if cfg.overwrite:
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
    if cfg.bicleaner_mixed_precision:
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


def parse_scored(path: Path, cfg: CleanConfig) -> list[Pair]:
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
                kept.append(Pair(pl=normalize(parts[3], cfg), en=normalize(parts[2], cfg), score=score))
    return kept


def write_outputs(pairs: list[Pair], cfg: CleanConfig, schema: Schema, counts: dict[str, int]) -> None:
    out_paths = [cfg.out_dir / "train.pl", cfg.out_dir / "train.en", cfg.out_dir / "train.tsv", cfg.out_dir / "stats.json"]
    ensure_writable(out_paths, cfg.overwrite)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.out_dir / "train.pl").open("w", encoding="utf-8", newline="\n") as pl_file, (cfg.out_dir / "train.en").open("w", encoding="utf-8", newline="\n") as en_file, (cfg.out_dir / "train.tsv").open("w", encoding="utf-8", newline="\n") as tsv_file:
        tsv_file.write("pl\ten\tscore\n")
        for pair in pairs:
            score = 1.0 if pair.score is None else pair.score
            pl_file.write(pair.pl + "\n")
            en_file.write(pair.en + "\n")
            tsv_file.write(f"{pair.pl}\t{pair.en}\t{score:.6f}\n")
    stats = {
        "dataset": cfg.dataset,
        "config": cfg.config,
        "split": cfg.split,
        "raw_dir": str(cfg.raw_dir),
        "out_dir": str(cfg.out_dir),
        "work_dir": str(cfg.work_dir),
        "threshold": cfg.threshold,
        "seed": cfg.seed,
        "min_tokens": cfg.min_tokens,
        "max_tokens": cfg.max_tokens,
        "max_length_ratio": cfg.max_length_ratio,
        "clean_batch_size": cfg.clean_batch_size,
        "skip_bicleaner": cfg.skip_bicleaner,
        "reuse_candidates": cfg.reuse_candidates,
        "bicleaner_require_gpu": cfg.bicleaner_require_gpu,
        "bicleaner_mixed_precision": cfg.bicleaner_mixed_precision,
        "after_bicleaner": len(pairs),
        "schema": asdict(schema),
        "counts": counts,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (cfg.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    cfg = parse_args()
    candidates_path = cfg.work_dir / "candidates.en-pl.tsv"
    scored_path = cfg.work_dir / "scored.tsv"
    out_paths = [cfg.out_dir / "train.pl", cfg.out_dir / "train.en", cfg.out_dir / "train.tsv", cfg.out_dir / "stats.json"]
    if cfg.reuse_candidates and not candidates_path.exists():
        raise FileNotFoundError(f"--reuse-candidates requires existing file: {candidates_path}")

    preflight_paths = [*out_paths]
    if not cfg.reuse_candidates:
        preflight_paths.append(candidates_path)
    if not cfg.skip_bicleaner:
        resolve_bicleaner_executable()
        preflight_paths.append(scored_path)
    ensure_writable(preflight_paths, cfg.overwrite)

    if cfg.reuse_candidates:
        pairs = read_candidates(candidates_path, cfg)
        counts = {
            "raw_rows": len(pairs),
            "after_basic_clean": len(pairs),
            "after_dedup": len(pairs),
            "after_max_rows": len(pairs),
        }
        schema = Schema("candidates", "column_3", "column_4", ["dummy_en", "dummy_pl", "en", "pl"], {})
        print(f"Reused candidates: {candidates_path}")
        print(f"Rows candidates: {len(pairs)}")
    else:
        dataset = load_tatoeba(cfg)
        schema = detect_schema(dataset)
        print(f"Rows raw: {len(dataset)}")
        print(f"Schema: {schema.kind}, EN={schema.en_field}, PL={schema.pl_field}")
        pairs, counts = basic_clean(dataset, schema, cfg)
        print(f"Rows after basic clean: {counts['after_basic_clean']}")
        print(f"Rows dropped by basic clean: {counts['raw_rows'] - counts['after_basic_clean']}")
        print(f"Rows after dedup: {counts['after_dedup']}")
        print(f"Rows after max rows: {counts['after_max_rows']}")

        write_candidates(pairs, candidates_path, cfg.overwrite)
        print(f"Wrote candidates: {candidates_path}")

    if cfg.skip_bicleaner:
        final_pairs = [Pair(pair.pl, pair.en, 1.0) for pair in pairs]
    else:
        run_bicleaner(candidates_path, scored_path, cfg)
        final_pairs = parse_scored(scored_path, cfg)
    print(f"Rows final: {len(final_pairs)}")
    write_outputs(final_pairs, cfg, schema, counts)
    print(f"Wrote outputs: {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
