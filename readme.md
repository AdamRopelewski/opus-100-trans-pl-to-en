# pl-en-transformer: Repo Overview

## What this repository is

`pl-en-transformer` is a training pipeline project for a custom
encoder-decoder Transformer for **Polish -> English** translation on
**OPUS-100 en-pl** data.

Current implemented scope in code:
- Stage 1: data audit
- Stage 2: data cleaning
- Stage 3: tokenizer training
- Stage 4: first PyTorch training pipeline

Stages 5+ are still planned as deeper model/eval iterations, but Stage 4 now
contains the first runnable dataloader, Transformer model, training loop and validation-loss checkpointing.


## Data flow (what is already working)

1. Raw parquet files are read from `data/raw/en-pl`.
2. Stage 1 audits raw splits and writes reports only.
3. Stage 2 cleans raw splits and writes processed parquet files to
   `data/processed/en-pl`.
4. Stage 3 trains the shared SentencePiece tokenizer.
5. Stage 4 trains PL -> EN on processed parquet with PyTorch.

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
- If uncertain stays high after retries, rows remain labeled as `uncertain`.
- Writes outputs:
  - `reports/llm_audit_<timestamp>/data_audit.md`
  - `reports/llm_audit_<timestamp>/data_audit_manifest.json`
  - `reports/llm_audit_<timestamp>/llm_audit_labels.jsonl`
  - `reports/llm_audit_<timestamp>/llm_audit_batches.jsonl`
  - `reports/llm_audit_<timestamp>/llm_bad_sentences.json`
  - per-split artifacts:
    - `reports/llm_audit_<timestamp>/llm_audit_labels_validation.jsonl`
    - `reports/llm_audit_<timestamp>/llm_audit_labels_test.jsonl`
    - `reports/llm_audit_<timestamp>/llm_audit_labels_train.jsonl`
    - `reports/llm_audit_<timestamp>/llm_audit_batches_validation.jsonl`
    - `reports/llm_audit_<timestamp>/llm_audit_batches_test.jsonl`
    - `reports/llm_audit_<timestamp>/llm_audit_batches_train.jsonl`
    - `reports/llm_audit_<timestamp>/llm_bad_sentences_validation.json`
    - `reports/llm_audit_<timestamp>/llm_bad_sentences_test.json`
    - `reports/llm_audit_<timestamp>/llm_bad_sentences_train.json`

Label JSONL format is compact to keep files small:
- global labels file: `{"s":"split","i":row_index,"l":label_id}`
- per-split labels file: `{"i":row_index,"l":label_id}`
- label ids: `0=bad`, `1=uncertain`, `2=good`


## Stage 2: Cleaning

Script:
- `scripts/clean_data.py`

What it does:
- Reads raw train/validation/test splits.
- Optionally consumes Stage 1 labels via `--llm-labels` and keeps only accepted
  LLM rows before writing processed parquet.
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

Stage 2 LLM label filtering:
- `--llm-labels reports/llm_audit_.../llm_audit_labels.jsonl` enables Stage 1
  label filtering.
- Default/`--good-only` keeps only label `2` (`good`).
- `--keep-uncertain` keeps labels `1` and `2` (`uncertain`, `good`).
- Rows missing from labels JSONL are dropped.


## Stage 3: Tokenizer

Script:
- `scripts/train_tokenizer.py`

What it does:
- Reads cleaned train/validation/test splits from `data/processed/en-pl`.
- Builds a shared Polish-English SentencePiece BPE training corpus.
- Trains a shared tokenizer with `vocab_size: 16000`.
- Locks special token ids: `pad=0`, `unk=1`, `bos=2`, `eos=3`.
- Writes outputs:
  - `tokenizers/spm_pl_en.model`
  - `tokenizers/spm_pl_en.vocab`
  - `reports/tokenizer_stats.md`

Tokenizer verification report includes:
- UNK rate per split.
- Subword/word compression ratio.
- Mean and p95 pieces per sentence.
- Long-word split stats by frequency bucket.
- Common long-word examples.


## Stage 4: PyTorch GPU Training

Script:
- `scripts/train_model.py`

What it does:
- Reads processed train/validation/test parquet splits from `data/processed/en-pl`.
- Uses `tokenizers/spm_pl_en.model` with special ids `pad=0`, `unk=1`, `bos=2`, `eos=3`.
- Encodes Polish source as source ids plus EOS.
- Encodes English target as BOS-prefixed decoder input and EOS-suffixed labels.
- Dynamically pads batches, creates source/target padding masks, and creates the decoder causal mask.
- Trains a compact `nn.Transformer` encoder-decoder model using AdamW, inverse-sqrt warmup scheduling, gradient accumulation, gradient clipping, label smoothing, and CUDA AMP.
- Saves checkpoints:
  - `checkpoints/model_last.pt`
  - `checkpoints/model_best.pt`
- Writes JSONL training logs to `logs/model_train.jsonl`.

GPU behavior:
- `stage6_train.require_cuda: true`
- `stage6_train.device: cuda`
- `stage6_train.allow_cpu_fallback: false`


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
python scripts/train_tokenizer.py --config configs/project_config.yaml
python scripts/train_model.py --config configs/project_config.yaml
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
- `stage3_tokenizer` trains a shared SentencePiece BPE tokenizer and writes `reports/tokenizer_stats.md`.
- `stage4_dataloader`, `stage5_model`, and `stage6_train` are used by `scripts/train_model.py`.
- `stage7_eval` remains a config section for later evaluation.
- Stage 1 through Stage 4 are implemented in code right now.
