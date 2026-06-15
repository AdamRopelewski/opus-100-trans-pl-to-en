# Tatoeba Data

Two scripts only:

- `scripts/download_tatoeba.py` downloads raw Tatoeba.
- `scripts/clean_tatoeba.py` cleans raw Tatoeba and writes PL -> EN files.

## Download

```bash
python scripts/download_tatoeba.py
```

Output:

- `data/raw/tatoeba/eng-pol/train`
- `data/raw/tatoeba/eng-pol/train/*.parquet`
- `data/raw/tatoeba/eng-pol/train/metadata.json`

Overwrite local raw copy:

```bash
python scripts/download_tatoeba.py --force
```

Download only first N Parquet parts:

```bash
python scripts/download_tatoeba.py --max-parts 5 --force
```

## Clean

Smoke test without Bicleaner:

```bash
python scripts/clean_tatoeba.py --skip-bicleaner --max-rows 1000 --overwrite
```

Full clean with Bicleaner CLI if installed:

```bash
python scripts/clean_tatoeba.py --max-rows 200000 --overwrite
```

Output defaults to `data/processed/tatoeba-en-pl`, not current `data/processed/en-pl`.

Files:

- `train.pl`
- `train.en`
- `train.tsv`
- `stats.json`

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
python scripts/clean_tatoeba.py --skip-bicleaner --max-rows 200000 --overwrite
```

For Bicleaner scoring, use WSL/Docker or any env where `bicleaner-ai-classify` works.
