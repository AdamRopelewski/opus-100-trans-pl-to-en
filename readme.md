# pl-en-transformer: Repo Overview

## What this repository is

`pl-en-transformer` is a training pipeline project for a custom
encoder-decoder Transformer for **Polish -> English** translation on
**OPUS-100 en-pl** data.

Current implemented scope in code:
- Stage 1: data audit
- Stage 2: data cleaning

Stages 3+ (tokenizer, dataloader, model, train, eval) are planned in config
and implementation plan, but not implemented yet.


## Data flow (what is already working)

1. Raw parquet files are read from `data/raw/en-pl`.
2. Stage 1 audits raw splits and writes reports only.
3. Stage 2 cleans raw splits and writes processed parquet files to
   `data/processed/en-pl`.

Important: Stage 1 does **not** modify dataset files.


## Stage 1: Audit

Script:
- `scripts/data_audit.py`

What it does:
- Reads train/validation/test raw parquet splits.
- Computes counts and quality diagnostics:
  - row counts
  - null pairs
  - empty pairs
  - duplicate pairs
  - suspicious patterns (HTML/XML, control chars, weird unicode,
    punctuation-only, very short, numeric mismatch, etc.)
- Computes length stats (chars/words for PL and EN).
- Saves deterministic random samples and suspicious samples.
- Writes outputs:
  - `reports/data_audit.md`
  - `reports/data_audit_manifest.json`

Language ID behavior in Stage 1:
- Always treated as an audit check (never deletes data).
- Ignores:
  - very short sentences
  - rows that are numeric-only or punctuation-only
- Records mismatch counts and row indices for manual verification.

Why Stage 1 can be slow:
- Full pass over 1M+ sentence pairs.
- Per-row Language ID classification is CPU-heavy.


## Stage 2: Cleaning

Script:
- `scripts/clean_data.py`

What it does:
- Reads raw train/validation/test splits.
- Normalizes text (NFKC, whitespace cleanup, optional control-char removal).
- Applies filters (null/empty, min/max words, ratio, identical pair).
- Applies deduplication and anti-leakage protections.
- Writes cleaned parquet files to `data/processed/en-pl`.
- Writes reports:
  - `reports/cleaning_report.md`
  - `reports/cleaning_manifest.json`
  - `reports/removed_examples.jsonl`

Critical behavior:
- Stage 2 **does remove rows** from output dataset.
- It does not only generate JSON suggestions.

Language ID behavior in Stage 2:
- Audit/telemetry only (not a removal reason).
- Mismatch counts and row indices are logged for manual review.
- Very short + numeric/punctuation-only rows are skipped for Language ID.


## Leakage safety and dedup policy (implemented)

Current cleaning supports global leakage-safe behavior:
- `dedup_scope: global`
- `preserve_validation_test_priority: true`
- `remove_train_pairs_present_in_validation_or_test: true`

Meaning:
- Validation/test pairs are protected by priority.
- If the same normalized `(pl, en)` pair appears in train and val/test,
  it is removed from train.
- No rows are moved between splits.


## Do you need Stage 1 before Stage 2?

No, not technically required.

- Stage 2 can run directly on raw data.
- Stage 1 is recommended for visibility and threshold tuning.

Also note:
- Stage 2 does not reuse Stage 1 runtime artifacts for processing.
- Both stages scan raw data independently.


## Progress bars on large datasets

Both stages include `tqdm` progress bars:
- split-level progress
- row-level progress per split

If `tqdm` is not installed, scripts run without bars.


## How to run

From repo root:

```bash
python scripts/data_audit.py --config configs/project_config.yaml
python scripts/clean_data.py --config configs/project_config.yaml
```


## Key paths

- Config: `configs/project_config.yaml`
- Raw data: `data/raw/en-pl`
- Processed data: `data/processed/en-pl`
- Reports: `reports/`
- Implementation plan: `PL_EN_Transformer_Implementation_Plan.md`
