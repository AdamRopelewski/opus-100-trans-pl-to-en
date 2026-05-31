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
- Runs pre-audit cleanup first (null/empty removal, duplicate drop, identical drop,
  min/max words, length-ratio gate).
- Sends rows to local Ollama LLM in batches.
- Classifies each row into:
  - `good`
  - `bad`
  - `uncertain` (manual review)
- Enforces strict JSON response format and ID validation.
- Retries batch on invalid JSON/ID mismatch and when uncertain ratio is too high.
- If uncertain stays high after retries, audit stops and asks whether to rerun with higher retries.
- Writes outputs:
  - `reports/data_audit.md`
  - `reports/data_audit_manifest.json`
  - `reports/llm_audit_labels.jsonl`
  - `reports/llm_audit_batches.jsonl`
  - `reports/llm_bad_sentences.json`
  - per-split artifacts:
    - `reports/llm_audit_labels_validation.jsonl`
    - `reports/llm_audit_labels_test.jsonl`
    - `reports/llm_audit_labels_train.jsonl`
    - `reports/llm_audit_batches_validation.jsonl`
    - `reports/llm_audit_batches_test.jsonl`
    - `reports/llm_audit_batches_train.jsonl`
    - `reports/llm_bad_sentences_validation.json`
    - `reports/llm_bad_sentences_test.json`
    - `reports/llm_bad_sentences_train.json`


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

Stage 2 behavior note:
- No language-ID filter is used.
- Filtering is based on normalization, min/max words, length ratio,
  identical-pair checks, deduplication, and leakage protection.


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

Verbose audit mode (prints batch preview rows, parsed label counts, and running totals):

```bash
python scripts/data_audit.py --config configs/project_config.yaml --verbose
```


## Key paths

- Config: `configs/project_config.yaml`
Ollama requirement:

```bash
ollama pull qwen2.5:7b
```
- Raw data: `data/raw/en-pl`
- Processed data: `data/processed/en-pl`
- Reports: `reports/`
- Implementation plan: `PL_EN_Transformer_Implementation_Plan.md`

Note on config scope:
- `stage3_tokenizer` through `stage7_eval` and `smoke` are placeholders for planned stages.
- Only Stage 1 and Stage 2 are implemented in code right now.
