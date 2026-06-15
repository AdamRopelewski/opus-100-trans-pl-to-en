You are working in my local machine translation project.

Goal:
Create a robust data preparation pipeline for Polish -> English MT training using Hugging Face dataset:
Helsinki-NLP/tatoeba_mt_train, config eng-pol, split train.

Then clean it, score sentence pairs with Bicleaner AI model:
bitextor/bicleaner-ai-full-en-xx

Then export parallel files:
data/processed/en-pl/train.pl
data/processed/en-pl/train.en

Important context:
- Direction for my model is PL -> EN.
- HF dataset config is eng-pol, so source may be English and target may be Polish.
- Do not assume exact column names. Inspect dataset.column_names and first row.
- Handle common schemas:
  - sourceString / targetString
  - source / target
  - translation dict
  - src / trg
  - sentence1 / sentence2
- If schema is unknown, fail with a clear error showing column names and first row.
- Use Hugging Face datasets.load_dataset.
- Do not use opustools.
- Bicleaner model is en-xx, so scoring input should be English first, Polish second.
- Final export must be Polish first, English second for PL -> EN training.

Implement:
1. A script:
   scripts/prepare_tatoeba_bicleaner.py

2. CLI args:
   --dataset Helsinki-NLP/tatoeba_mt_train
   --config eng-pol
   --split train
   --out-dir data/processed/en-pl
   --work-dir data/work/tatoeba_bicleaner
   --threshold 0.80
   --max-rows 0
   --seed 42
   --min-chars 15
   --max-chars 700
   --max-length-ratio 2.5
   --bicleaner-model bitextor/bicleaner-ai-full-en-xx
   --skip-bicleaner
   --overwrite

3. Pipeline:
   - create output dirs
   - load HF dataset
   - detect EN and PL fields safely
   - normalize whitespace
   - remove empty pairs
   - remove pairs below min chars
   - remove pairs above max chars
   - remove pairs with length ratio above max-length-ratio
   - remove pairs containing obvious junk:
     URL, HTML tags, &nbsp;, mostly punctuation
   - remove exact duplicates using normalized lowercase key (pl, en)
   - shuffle with seed
   - if max_rows > 0, keep only that many rows after basic cleaning
   - write intermediate TSV for Bicleaner:
     data/work/tatoeba_bicleaner/candidates.en-pl.tsv

4. Bicleaner input format:
   The bicleaner-ai CLI commonly expects at least 4 columns:
   url1<TAB>url2<TAB>source_sentence<TAB>target_sentence
   So write:
   dummy_en<TAB>dummy_pl<TAB>EN<TAB>PL

5. Run Bicleaner from Python via subprocess:
   bicleaner-ai-classify candidates.en-pl.tsv scored.tsv bitextor/bicleaner-ai-full-en-xx en xx

   But make this robust:
   - Check executable availability with shutil.which("bicleaner-ai-classify")
   - If missing, print install hint:
     python -m pip install bicleaner-ai
   - Capture stdout/stderr.
   - If command fails, print full command and stderr.
   - Keep a function run_bicleaner() so I can edit the command easily.

6. Scored output parsing:
   - Do not assume exact number of columns.
   - For every line, split by tab.
   - Score is the last parseable float in the line.
   - EN and PL should be recovered from known candidate columns when possible.
   - If scored output preserves original 4 columns plus score, use col 2 EN and col 3 PL.
   - If format differs, add defensive parsing and clear error.
   - Keep rows with score >= threshold.
   - If --skip-bicleaner is set, export all basic-cleaned rows.

7. Final outputs:
   - train.pl
   - train.en
   - train.tsv with columns: pl, en, score
   - stats.json with:
     dataset
     config
     split
     raw_rows
     after_basic_clean
     after_dedup
     after_max_rows
     after_bicleaner
     threshold
     seed
     min_chars
     max_chars
     max_length_ratio
     bicleaner_model
     timestamp_utc

8. Add a small preview print:
   - raw dataset size
   - detected schema
   - counts after every step
   - 5 sample final pairs with score

9. Add requirements if missing:
   datasets
   tqdm

   Do not add torch unless needed.
   Do not force-install bicleaner-ai into requirements if it causes dependency pain. Add it as optional in comment:
   # optional for filtering:
   # bicleaner-ai

10. Code quality:
   - Python 3.10+
   - type hints
   - dataclass for config
   - no global magic except constants
   - clear functions:
     parse_args()
     load_hf_dataset()
     detect_parallel_fields()
     extract_pair()
     basic_filter()
     write_candidates()
     run_bicleaner()
     parse_scored_file()
     write_final_outputs()
     write_stats()
     main()

11. Also add a README section or a new docs file:
   docs/tatoeba_bicleaner_pipeline.md

Include:
   - what this pipeline does
   - why EN is passed first to Bicleaner
   - why final export is PL -> EN
   - example commands

Example command:
python scripts/prepare_tatoeba_bicleaner.py --threshold 0.80 --max-rows 200000 --overwrite

Example without Bicleaner:
python scripts/prepare_tatoeba_bicleaner.py --skip-bicleaner --max-rows 50000 --overwrite

12. After implementing, run:
   python scripts/prepare_tatoeba_bicleaner.py --skip-bicleaner --max-rows 1000 --overwrite

Fix all errors.

13. Do not touch model training code unless needed.
14. Do not delete existing data.
15. Preserve Windows compatibility.
16. All shell commands in docs must be one-line commands.

Deliver:
- modified/created files list
- exact command to run full pipeline
- exact command to run smoke test