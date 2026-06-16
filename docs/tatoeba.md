# Tatoeba Data

Tatoeba-specific scripts live under `scripts/tatoeba/`:

- `scripts/tatoeba/download.py` downloads raw Tatoeba.
- `scripts/tatoeba/clean.py` cleans raw Tatoeba and writes cleaned shards.
- `scripts/tatoeba/split.py` deduplicates cleaned shards and writes final train/validation/test splits.
- `scripts/tatoeba/train_tokenizer.py` trains tokenizer with `configs/tatoeba_config.yaml` by default.
- `scripts/tatoeba/train.py` trains with `configs/tatoeba_config.yaml` by default.
- `scripts/tatoeba/evaluate.py` evaluates with `configs/tatoeba_config.yaml` by default.

## Download

```bash
python scripts/tatoeba/download.py
```

Output:

- `data/raw/tatoeba/eng-pol/train`
- `data/raw/tatoeba/eng-pol/train/*.parquet`
- `data/raw/tatoeba/eng-pol/train/metadata.json`

Overwrite local raw copy:

```bash
python scripts/tatoeba/download.py --force
```

Download only first N Parquet parts:

```bash
python scripts/tatoeba/download.py --max-parts 5 --force
```

## Clean

Smoke test without Bicleaner:

```bash
python scripts/tatoeba/clean.py --skip-bicleaner --max-rows 1000 --overwrite
```

Full clean with Bicleaner CLI if installed:

```bash
python scripts/tatoeba/clean.py --max-rows 200000 --overwrite
```

Output defaults to `data/processed/tatoeba-en-pl`, not current `data/processed/en-pl`.

Files:

- `00000.train.parquet`, `00001.train.parquet`, ...
- `00000.train.manifest.json`, `00001.train.manifest.json`, ...
- `manifest.en-pl*.json`

## Split

Create final train/validation/test files:

```bash
python scripts/tatoeba/split.py --overwrite
```

Output defaults to `data/processed/tatoeba-en-pl/splits`.

Files:

- `train-00000-of-00001.parquet`
- `validation-00000-of-00001.parquet`
- `test-00000-of-00001.parquet`
- `split_manifest.json`

Default validation and test sizes are 2000 rows each. Manifest includes rows in, rows after dedup, rows removed, per-file duplicate counts, split sizes, and leakage checks.

## Tokenizer, Train, Eval

Use shared training/eval scripts with Tatoeba config:

```bash
python scripts/tatoeba/train_tokenizer.py --force
python scripts/tatoeba/train.py
python scripts/tatoeba/evaluate.py
```

Equivalent direct commands:

```bash
python scripts/train_tokenizer.py --config configs/tatoeba_config.yaml --force
python scripts/train_model.py --config configs/tatoeba_config.yaml
python scripts/evaluate_model.py --config configs/tatoeba_config.yaml
```

## Filters

- empty pairs
- duplicate `(pl, en)` pairs
- sentences under 5 tokens
- sentences over `stage4_dataloader.max_seq_len` from `configs/project_config.yaml`, default 128
- token length ratio over 2.5
- URL/HTML/mostly punctuation junk

## Bicleaner On Windows

`bicleaner-ai` pip install can fail on Windows because `bicleaner-ai-glove` source package is broken. Manual HF model download does not fix missing `bicleaner-ai-classify` CLI.

Windows-safe path:

```bash
python scripts/tatoeba/clean.py --skip-bicleaner --max-rows 200000 --overwrite
```

For Bicleaner scoring, use WSL/Docker or any env where `bicleaner-ai-classify` works.

## Bicleaner In Docker

Use a separate Bicleaner venv only for cleaning. Torch stays system-wide in Docker for training. `scripts/tatoeba/clean.py` auto-runs `bicleaner-ai-classify` from `.venv-bicleaner` when it exists.

Create and install once:

```bash
bash scripts/setup_bicleaner_venv.sh
```

The setup script creates `.venv-bicleaner` and `.venv-bicleaner/bin/activate-bicleaner-cuda`.

Clean with Bicleaner:

```bash
python scripts/tatoeba/clean.py --max-rows 200000 --overwrite
```

Manual run, if needed:

```bash
. .venv-bicleaner/bin/activate
. .venv-bicleaner/bin/activate-bicleaner-cuda
python scripts/tatoeba/clean.py --max-rows 200000 --overwrite
deactivate
```

`scripts/tatoeba/clean.py` passes `--disable_hardrules` to Bicleaner because basic hard filtering is already handled by the script. Progress tracks written rows in `data/work/tatoeba/scored.*.tsv` and refreshes about every 10 seconds. Auto-run uses venv only for Bicleaner subprocess; parent shell remains unchanged.
