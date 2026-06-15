from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
ALNUM_RE = re.compile(r"[\wĄąĆćĘęŁłŃńÓóŚśŹźŻż]", re.UNICODE)


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
    bicleaner_model: str
    skip_bicleaner: bool
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
    parser.add_argument("--bicleaner-model", default="bitextor/bicleaner-ai-full-en-pl")
    parser.add_argument("--skip-bicleaner", action="store_true")
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
        bicleaner_model=args.bicleaner_model,
        skip_bicleaner=args.skip_bicleaner,
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
    return len(ALNUM_RE.findall(text)) / non_space < 0.35


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
    pairs: list[Pair] = []
    iterator: Iterable[dict[str, Any]] = dataset
    if tqdm is not None:
        iterator = tqdm(dataset, desc="Clean Tatoeba", unit="rows")
    for row in iterator:
        pair = extract_pair(dict(row), schema, cfg)
        reason = reject_reason(pair, cfg)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        pairs.append(pair)
    counts["after_basic_clean"] = len(pairs)

    seen: set[tuple[str, str]] = set()
    deduped: list[Pair] = []
    for pair in pairs:
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


def run_bicleaner(candidates_path: Path, scored_path: Path, cfg: CleanConfig) -> None:
    executable = shutil.which("bicleaner-ai-classify")
    if executable is None:
        raise RuntimeError("bicleaner-ai-classify not found. On Windows use --skip-bicleaner or run Bicleaner in WSL/Docker.")
    ensure_writable([scored_path], cfg.overwrite)
    command = [executable, str(candidates_path), str(scored_path), cfg.bicleaner_model, "en", "pl"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Bicleaner failed. Command: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


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
        "skip_bicleaner": cfg.skip_bicleaner,
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
    preflight_paths = [candidates_path, *out_paths]
    if not cfg.skip_bicleaner:
        preflight_paths.append(scored_path)
    ensure_writable(preflight_paths, cfg.overwrite)

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
