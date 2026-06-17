from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.tatoeba import evaluate


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), str(path), compression="zstd")


def test_tatoeba_evaluate_uses_tokenized_test_ids(
    tmp_path: Path, monkeypatch
) -> None:
    tokenized_dir = tmp_path / "tokenized"
    checkpoint_path = tmp_path / "model_best.pt"
    metrics_path = tmp_path / "reports" / "eval_metrics.json"
    translations_path = tmp_path / "reports" / "translations" / "test.tsv"
    samples_path = tmp_path / "reports" / "translations" / "samples.md"
    checkpoint_path.write_bytes(b"checkpoint")
    _write_parquet(
        tokenized_dir / "test" / "test-tokenized-00001-of-00001.parquet",
        [
            {"src_ids": [4, 5, 3], "tgt_ids": [6, 7]},
            {"src_ids": [8, 3], "tgt_ids": [9]},
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
stage4_dataloader:
  tokenized_splits_dir: {tokenized_dir.as_posix()}
  tokenized_test_pattern: test-tokenized-*.parquet
  tokenized_read_batch_size: 2
stage6_train:
  output_best_checkpoint: {checkpoint_path.as_posix()}
stage7_eval:
  enabled: true
  batch_size: 2
  outputs:
    metrics_json: {metrics_path.as_posix()}
    translations_tsv: {translations_path.as_posix()}
    samples_md: {samples_path.as_posix()}
""".strip(),
        encoding="utf-8",
    )

    class DummyTokenizer:
        def pad_id(self) -> int:
            return 0

        def bos_id(self) -> int:
            return 2

        def eos_id(self) -> int:
            return 3

        def decode(self, ids: list[int]) -> str:
            return " ".join(str(item) for item in ids)

    seen: dict[str, object] = {}

    def fake_translate(source_id_rows: list[list[int]], *_args) -> list[str]:
        seen["source_id_rows"] = source_id_rows
        return [f"translated: {' '.join(str(item) for item in row)}" for row in source_id_rows]

    monkeypatch.setattr(
        evaluate,
        "_load_model_bundle",
        lambda *_args: (DummyTokenizer(), object(), SimpleNamespace(max_seq_len=128)),
    )
    monkeypatch.setattr(evaluate, "_translate_tokenized_batch", fake_translate)
    monkeypatch.setattr(
        evaluate,
        "_compute_metrics",
        lambda hypotheses, references: {"sacrebleu": 12.5, "chrf": 44.0},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--config", str(config_path), "--device", "cpu"],
    )

    assert evaluate.main() == 0
    assert seen["source_id_rows"] == [[4, 5, 3], [8, 3]]

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["rows_evaluated"] == 2
    assert "test-tokenized-00001-of-00001.parquet" in metrics["test_split"]

    lines = translations_path.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "0\t4 5\t6 7\ttranslated: 4 5 3"
