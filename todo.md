- [x] stage1 audit: tune uncertain threshold and retry count on a small subset first]
- [x] stage1: fillter all `-` `\` etc. lets just do whitelisting
- [x] stage2: consume stage1 labels (keep good, review uncertain) and avoid repeating stage1 heuristics
- [x] stage2: share sanitizer with stage1 and keep only lightweight final normalization
- [x] stage 3 - tokenizer
- config hygiene: keep all runtime thresholds centralized in project config
- tests: add unit coverage for llm_audit validation/retry/artifact writing

## Tatoeba multi-parquet pipeline

- [ ] `scripts/tatoeba/download.py`: keep current range download behavior and skip already existing local Parquet files unless `--force` is used.
- [ ] `scripts/tatoeba/download.py`: keep metadata listing all local Parquet files after each run, not only files downloaded in current run.
- [ ] `scripts/tatoeba/clean.py`: load local raw Parquet files one-by-one so each row can keep provenance.
- [ ] `scripts/tatoeba/clean.py`: add `source_parquet` and `source_row_index` to cleaned rows.
- [ ] `scripts/tatoeba/clean.py`: deduplicate globally across all selected raw Parquet files before training.
- [ ] `scripts/tatoeba/clean.py`: write removed duplicate examples with `source_parquet` and `source_row_index`.
- [ ] `scripts/tatoeba/clean.py`: write merged processed Parquet under `data/processed/tatoeba-en-pl` with columns `pl`, `en`, `score`, `source_parquet`, `source_row_index`.
- [ ] `scripts/tatoeba/clean.py`: keep `train.pl`, `train.en`, and `train.tsv` outputs if still useful, but make Parquet primary training artifact.
- [ ] `scripts/tatoeba/clean.py`: update `stats.json` with all source Parquets, rows per source file, total duplicate count, and duplicate count per source file.
- [ ] `scripts/data_audit.py`: decide whether Tatoeba audit runs on merged processed Parquet or raw multi-Parquet with provenance.
- [ ] `src/utils/llm_audit.py`: if auditing raw multi-Parquet, make labels source-aware using `split`, `source_parquet`, and `source_row_index` instead of only `split` and row index.
- [ ] `scripts/clean_data.py`: if shared OPUS stage2 must support multiple raw Parquets per split, replace exact-one split resolver with multi-file resolver.
- [ ] `scripts/train_tokenizer.py`, `scripts/train_model.py`, `scripts/evaluate_model.py`: no change needed if Tatoeba cleaner writes one merged processed Parquet per split; update only if processed output becomes multiple Parquet files.
- [ ] tests: add coverage for raw files `train-00000...` plus `train-00001...` with duplicate across files and provenance preserved in processed output.
