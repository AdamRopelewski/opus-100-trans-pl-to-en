from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts import evaluate_model


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path), compression="zstd")


def test_evaluate_model_writes_metrics_translations_and_samples(
    tmp_path: Path, monkeypatch
) -> None:
    processed_dir = tmp_path / "processed"
    checkpoint_path = tmp_path / "model_best.pt"
    metrics_path = tmp_path / "reports" / "eval_metrics.json"
    translations_path = tmp_path / "reports" / "translations" / "test.tsv"
    samples_path = tmp_path / "reports" / "translations" / "samples.md"
    checkpoint_path.write_bytes(b"checkpoint")
    _write_parquet(
        processed_dir / "test-00000-of-00001.parquet",
        [
            {"pl": "ala ma kota", "en": "alice has a cat"},
            {"pl": "pies spi", "en": "the dog sleeps"},
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
stage2_cleaning:
  outputs:
    processed_dir: {processed_dir.as_posix()}
dataset:
  splits:
    test_pattern: test-*.parquet
stage6_train:
  output_best_checkpoint: {checkpoint_path.as_posix()}
stage7_eval:
  enabled: true
  beam_size: 2
  length_penalty: 0.8
  max_new_tokens: 16
  num_sample_translations: 1
  outputs:
    metrics_json: {metrics_path.as_posix()}
    translations_tsv: {translations_path.as_posix()}
    samples_md: {samples_path.as_posix()}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_model,
        "_load_model_bundle",
        lambda *_args: (object(), object(), SimpleNamespace(max_seq_len=128)),
    )
    monkeypatch.setattr(
        evaluate_model,
        "_translate_batch",
        lambda texts, *_args: [f"translated: {text}" for text in texts],
    )
    monkeypatch.setattr(
        evaluate_model,
        "_compute_metrics",
        lambda hypotheses, references: {"sacrebleu": 12.5, "chrf": 44.0},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_model.py",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--decode",
            "greedy",
        ],
    )

    assert evaluate_model.main() == 0

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["rows_evaluated"] == 2
    assert metrics["decode"] == "greedy"
    assert metrics["beam_size"] == 2
    assert metrics["metrics"] == {"chrf": 44.0, "sacrebleu": 12.5}

    lines = translations_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "index\tsource_pl\treference_en\thypothesis_en"
    assert lines[1] == "0\tala ma kota\talice has a cat\ttranslated: ala ma kota"
    assert lines[2] == "1\tpies spi\tthe dog sleeps\ttranslated: pies spi"

    samples = samples_path.read_text(encoding="utf-8")
    assert "# Sample Translations" in samples
    assert "## Row 0" in samples
    assert "## Row 1" not in samples


def test_compute_metrics_reports_missing_sacrebleu(monkeypatch) -> None:
    monkeypatch.setattr(evaluate_model, "sacrebleu", None)

    try:
        evaluate_model._compute_metrics(["a"], ["a"])
    except RuntimeError as exc:
        assert "install sacrebleu" in str(exc)
    else:
        raise AssertionError("Expected missing sacrebleu RuntimeError")


def test_stage7_disabled_skips(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("stage7_eval:\n  enabled: false\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["evaluate_model.py", "--config", str(config_path)]
    )

    assert evaluate_model.main() == 0
