# Data Cleaning Report (OPUS-100 en-pl)

Generated at (UTC): `2026-05-27T19:30:05.243204+00:00`
Raw directory: `data\raw\en-pl`
Processed directory: `data\processed\en-pl`

## Config

- min_words: 1
- max_words: 200
- max_length_ratio: 3.0
- unicode_normalization: NFKC
- dedup_scope: split

## Totals

- Rows in: 1004000
- Rows out: 900648
- Rows removed: 103352
- Retention ratio: 0.89706

## Split: train

- Input file: `train-00000-of-00001.parquet`
- Output file: `train-00000-of-00001.parquet`
- Rows in: 1000000
- Rows out: 896810
- Removed null pairs: 0
- Removed empty pairs: 0
- Removed min words: 0
- Removed max words: 43
- Removed length ratio: 30901
- Removed duplicates: 72246

## Split: validation

- Input file: `validation-00000-of-00001.parquet`
- Output file: `validation-00000-of-00001.parquet`
- Rows in: 2000
- Rows out: 1915
- Removed null pairs: 0
- Removed empty pairs: 0
- Removed min words: 0
- Removed max words: 0
- Removed length ratio: 77
- Removed duplicates: 8

## Split: test

- Input file: `test-00000-of-00001.parquet`
- Output file: `test-00000-of-00001.parquet`
- Rows in: 2000
- Rows out: 1923
- Removed null pairs: 0
- Removed empty pairs: 0
- Removed min words: 0
- Removed max words: 0
- Removed length ratio: 68
- Removed duplicates: 9
