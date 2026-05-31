# PL to EN Transformer (PyTorch) - Implementation Plan

This document is a step-by-step execution plan for building a Polish to English neural machine translation model with multi-head attention using PyTorch and the OPUS-100 `en-pl` dataset.

---

## 1) Project Goal

Build and train a custom encoder-decoder Transformer for `pl -> en` translation that reaches solid baseline quality on OPUS-100 and can be improved in controlled iterations.

Primary goals:
- End-to-end reproducible training pipeline.
- Stable training on a single `RTX 5070 12GB` GPU.
- Measurable quality with `sacreBLEU` and `chrF++`.

---

## 2) Hardware and Time Assumptions

Target hardware:
- GPU: RTX 5070, 12 GB VRAM.
- CPU/RAM: standard desktop setup (enough to preprocess and stream data).

Expected training duration (baseline model):
- One epoch: roughly 2 to 5 hours (depends mostly on max sequence length and effective batch tokens).
- Useful baseline quality: 8 to 15 epochs.
- Total expected wall time: around 24 to 48 hours (can extend toward 60 to 75 hours if settings are heavy).

---

## 3) Recommended Baseline Model

Use a Transformer-base style model.

Architecture:
- `d_model = 512`
- `num_heads = 8`
- `num_encoder_layers = 6`
- `num_decoder_layers = 6`
- `d_ff = 2048`
- `dropout = 0.1`

Tokenization:
- Shared SentencePiece (or BPE) vocabulary for both languages.
- `vocab_size = 32000`
- Special tokens: `PAD`, `BOS`, `EOS`, `UNK`.

Training settings (starting point):
- Optimizer: AdamW (`betas=(0.9, 0.98)`, `eps=1e-9`, `weight_decay=0.01`).
- Scheduler: warmup + inverse square root decay.
- Warmup steps: `4000` to `8000`.
- Peak LR: around `3e-4`.
- Label smoothing: `0.1`.
- Mixed precision: `fp16` or `bf16`.
- Gradient clipping: `1.0`.

Memory-safe defaults for 12 GB VRAM:
- Start with `max_seq_len = 128`.
- Use small micro-batch and gradient accumulation.
- Enable bucketing by sequence length.

---

## 4) Project Structure (Updated)

```text
project/
  .gitignore
  data/
    raw/
      en-pl/
        train-00000-of-00001.parquet
        validation-00000-of-00001.parquet
        test-00000-of-00001.parquet
    processed/
  tokenizers/
  configs/
    project_config.yaml
  src/
    data/
    model/
    train/
    eval/
    utils/
  scripts/
  checkpoints/
  logs/
  reports/
  venv/
  PL_EN_Transformer_Implementation_Plan.md
```

Notes:
- HF split files are already present in `data/raw/en-pl/`.
- Keep parquet files out of Git via `.gitignore`.
- Use one central config file: `configs/project_config.yaml`.
- All scripts read settings from this file (no per-script standalone configs).

---

## 5) Stage-by-Stage Implementation

Each stage has objective, tasks, outputs, and acceptance checks.

## Stage 0 - Environment Setup

Objective:
- Prepare a reproducible Python/PyTorch environment.

Tasks:
- Create virtual environment.
- Install core dependencies: `torch`, `datasets`, `sentencepiece`, `sacrebleu`, `numpy`, `pandas`, `tqdm`, `pyyaml`.
- Save dependency versions.

Outputs:
- `requirements.txt` or `pyproject.toml`.
- A short setup command list in `README.md`.
- Central config file: `configs/project_config.yaml`.

Acceptance checks:
- `python -c "import torch; print(torch.cuda.is_available())"` returns `True`.
- GPU is visible from PyTorch.

---

## Stage 1 - Data Audit (HF Splits Already Available)

Objective:
- Validate the already downloaded OPUS-100 `en-pl` split files and create a local manifest.

Tasks:
- Read `train`, `validation`, `test` from `data/raw/en-pl/*.parquet`.
- Report split sizes, null/empty pairs, duplicate pairs.
- Report length statistics for chars and words in PL/EN.
- Detect suspicious pairs:
  - identical source and target
  - HTML/XML tags
  - control characters
  - weird unicode classes
  - punctuation-only pairs
  - very short pairs
  - numeric mismatch between source and target
  - optional language-id mismatch
- Save deterministic random samples and suspicious samples using seed from config.
- Save markdown report and JSON manifest.

Outputs:
- Script: `scripts/data_audit.py`.
- Utility module: `src/utils/data_audit.py`.
- Report: `reports/data_audit.md`.
- Manifest: `reports/data_audit_manifest.json`.

Acceptance checks:
- Split counts are known and documented.
- No schema surprises in `translation['pl']` and `translation['en']`.
- Suspicious counts and samples are reported per split.
- Report reproducible for fixed `project.seed`.

Status:
- Dataset split acquisition from HF is done.
- Stage 1 implemented.

---

## Stage 2 - Data Cleaning and Filtering

Objective:
- Remove low-quality pairs safely and prevent train-validation/test leakage.

Tasks:
- Normalize source and target text:
  - Unicode normalization (`NFKC`)
  - strip whitespace
  - collapse whitespace
  - remove control chars
- Filter rows:
  - null pairs
  - empty pairs
  - min/max words
  - source/target length ratio
  - identical source-target
  - optional language-id mismatch
- Deduplicate on normalized `(source, target)` hash.
- Support `dedup_scope=global` with leakage protection:
  - process validation/test before train if `preserve_validation_test_priority=true`
  - remove train pairs present in validation/test when
    `remove_train_pairs_present_in_validation_or_test=true`
  - never move rows between splits
- Track removal reasons per row and per split.
- For rows matching multiple filters, keep all reasons but pick one stable
  primary reason by priority.
- Save removed examples to JSONL with `reason` and `all_reasons`.
- Save cleaned outputs as parquet in `data/processed/en-pl/`.

Outputs:
- Script: `scripts/clean_data.py`.
- Utility module: `src/utils/clean_data.py`.
- Clean dataset files in `data/processed/en-pl/`.
- Cleaning summary report: `reports/cleaning_report.md`.
- Cleaning manifest: `reports/cleaning_manifest.json`.
- Removed examples JSONL: `reports/removed_examples.jsonl`.

Acceptance checks:
- No leaked pair from validation/test remains in train when leakage filter is on.
- Primary removal reasons and totals are documented.
- Language-id behavior explicit: enabled, skipped, or strict error.

Status:
- Stage 2 implemented.

---

## Stage 3 - Tokenizer Training

Implementation note:
- Not implemented in code yet.
- Config stays prepared for SentencePiece training in a future step.

Objective:
- Train a shared subword tokenizer suitable for Polish morphology and English output.

Tasks:
- Build training text from cleaned `pl` and `en`.
- Train SentencePiece/BPE tokenizer with `vocab_size=32000`.
- Define and lock special token IDs.
- Save tokenizer config and model files.

Outputs:
- `tokenizers/spm_pl_en.model`
- `tokenizers/spm_pl_en.vocab`
- `reports/tokenizer_stats.md`

Acceptance checks:
- Tokenization and detokenization are stable.
- Unknown token rate is low on validation split.

Status:
- Planned only (not implemented yet).

---

## Stage 4 - Dataset Pipeline in PyTorch

Implementation note:
- Not implemented in code yet.
- Planned scope includes fp16/bf16-safe padding, masks, and batch smoke tests.

Status:
- Planned only (not implemented yet).

Objective:
- Build efficient train/val/test dataloaders with proper masks.

Tasks:
- Implement dataset class for `(src_ids, tgt_in_ids, tgt_out_ids)`.
- Implement dynamic padding collate function.
- Implement masks:
  - source padding mask
  - target padding mask
  - causal mask for decoder self-attention
- Add length-based bucketing for speed.

Outputs:
- `src/data/dataset.py`
- `src/data/collate.py`
- `src/data/dataloader.py`

Acceptance checks:
- Batches have correct tensor shapes.
- Causal mask blocks future positions.
- No shape mismatch during a forward pass.

---

## Stage 5 - Model Implementation

Implementation note:
- Not implemented in code yet.
- Planned presets: `small` and `base`, with explicit embedding tying flags.

Status:
- Planned only (not implemented yet).

Objective:
- Implement encoder-decoder Transformer with multi-head attention.

Tasks:
- Implement embeddings and positional encoding.
- Implement Transformer model (or wrap `nn.Transformer` cleanly).
- Add output projection to vocab size.
- Optional but recommended: tie target embedding and output projection weights.

Outputs:
- `src/model/transformer_nmt.py`
- Model section inside `configs/project_config.yaml`

Acceptance checks:
- Single forward/backward step works.
- Parameter count is logged.
- Inference on toy batch returns token logits without errors.

---

## Stage 6 - Training Loop and Optimization

Implementation note:
- Not implemented in code yet.
- Planned: scheduler, checkpointing, precision fallback, NaN handling, resume flow.

Status:
- Planned only (not implemented yet).

Objective:
- Train stably on a 12 GB GPU with reproducible checkpoints.

Tasks:
- Implement train loop with mixed precision.
- Add gradient accumulation.
- Add AdamW + LR scheduler + warmup.
- Add label smoothing loss and PAD ignore index.
- Add gradient clipping.
- Save checkpoints and best model by validation BLEU.

Outputs:
- `src/train/train.py`
- `src/train/losses.py`
- `src/train/scheduler.py`
- Training section inside `configs/project_config.yaml`

Acceptance checks:
- Loss decreases during first few thousand steps.
- No NaNs or exploding gradients.
- Training resumes correctly from checkpoint.

---

## Stage 7 - Validation, BLEU, chrF++

Implementation note:
- Not implemented in code yet.
- Planned: validation and test evaluation with sacreBLEU/chrF and translation artifacts.

Status:
- Planned only (not implemented yet).

Objective:
- Measure translation quality and prevent overfitting.

Tasks:
- Implement greedy and beam search decoding.
- Evaluate on validation split every N steps.
- Compute `sacreBLEU` and `chrF++`.
- Track best checkpoints and optionally average top-k checkpoints.

Outputs:
- `src/eval/decode.py`
- `src/eval/metrics.py`
- `reports/validation_curves.md`

Acceptance checks:
- Metrics are deterministic for fixed checkpoint and decode settings.
- Validation trend improves over early epochs.

---

## Stage 8 - Final Test Evaluation

Status:
- Planned only (not implemented yet).

Objective:
- Produce final benchmark numbers and example translations.

Tasks:
- Run final decode on test split.
- Report BLEU and chrF++.
- Provide qualitative examples (good, medium, bad translations).
- Analyze common error categories.

Outputs:
- `reports/final_test_report.md`

Acceptance checks:
- Final metrics and decoding settings are fully documented.
- Report includes reproducibility info (seed, config, checkpoint path).

---

## Stage 9 - Iteration Plan (After Baseline)

Status:
- Planned only (not implemented yet).

Objective:
- Improve quality in controlled experiments.

Priority order:
1. Better data filtering (highest ROI on OPUS-style mixed corpora).
2. Decode tuning (`beam`, `length_penalty`).
3. Increase max sequence length to 192 or 256 if memory allows.
4. Try larger model (`d_model=768`, `8/8`) only if time budget allows.
5. Optional domain fine-tuning.

Outputs:
- `reports/ablation_study.md`

Acceptance checks:
- Each experiment changes one major variable.
- Metrics are compared under identical evaluation settings.

---

## 6) Unified Config Template (12 GB GPU)

```yaml
project:
  name: pl-en-transformer
  seed: 42

paths:
  raw_data_dir: data/raw/en-pl
  processed_data_dir: data/processed/en-pl
  reports_dir: reports
  tokenizer_dir: tokenizers
  checkpoints_dir: checkpoints
  logs_dir: logs

dataset:
  source_lang: pl
  target_lang: en
  splits:
    train_pattern: train-*.parquet
    validation_pattern: validation-*.parquet
    test_pattern: test-*.parquet

stage1_audit:
  samples_per_split: 5
  outputs:
    report_md: reports/data_audit.md
    manifest_json: reports/data_audit_manifest.json

stage2_cleaning:
  filters:
    unicode_normalization: NFKC
    dedup_scope: split
    min_words: 1
    max_words: 200
    max_length_ratio: 3.0
  outputs:
    processed_dir: data/processed/en-pl
    report_md: reports/cleaning_report.md
    manifest_json: reports/cleaning_manifest.json

stage4_dataloader:
  max_seq_len: 128

stage3_tokenizer:
  type: sentencepiece
  vocab_size: 32000
  character_coverage: 0.9995
  model_prefix: tokenizers/spm_pl_en

stage5_model:
  d_model: 512
  nhead: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
  dim_feedforward: 2048
  dropout: 0.1
  tie_embeddings: true

stage6_train:
  precision: fp16
  optimizer: adamw
  lr_peak: 3e-4
  warmup_steps: 4000
  weight_decay: 0.01
  label_smoothing: 0.1
  grad_clip_norm: 1.0
  micro_batch_size: 16
  grad_accum_steps: 8
  num_epochs: 12
  validate_every_steps: 2000
  save_every_steps: 2000

stage7_eval:
  beam_size: 4
  length_penalty: 0.9
```

Note:
- Keep all stage options in single `configs/project_config.yaml`.
- `micro_batch_size` and `grad_accum_steps` are placeholders. Tune by VRAM usage.

---

## 7) Risk List and Mitigations

1. Out-of-memory errors:
- Reduce `max_seq_len`, then micro-batch, then enable gradient checkpointing.

2. Slow training throughput:
- Use bucketing, increase dataloader workers, profile CPU bottlenecks.

3. Quality plateau too early:
- Improve filtering quality, tune LR schedule, and decode settings.

4. Noisy translations from mixed OPUS domains:
- Add stricter cleaning and optional domain fine-tuning.

---

## 8) Definition of Done

Project is complete when:
- End-to-end training is reproducible from a single config file.
- Best checkpoint is selected on validation BLEU/chrF++.
- Final test report contains metrics, configs, and translation examples.
- The full pipeline can be rerun without manual fixes.

---

## 9) Next Immediate Actions


1. Train tokenizer and freeze vocabulary.
2. Implement model and run a tiny overfit test before full training.
