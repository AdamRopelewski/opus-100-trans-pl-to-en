from __future__ import annotations

from datasets import Dataset

from src.utils.preaudit import PreAuditConfig, normalize_preaudit_text, preaudit_filter_rows, sanitize_preaudit_text


def _clean(text: str) -> str:
    return normalize_preaudit_text(sanitize_preaudit_text(text))


def test_stage1_sanitizer_removes_weird_characters() -> None:
    assert _clean("你好 Ala / ma \\ kota 😊") == "Ala ma kota"


def test_stage1_sanitizer_preserves_pl_en_digits_punctuation_and_quotes() -> None:
    text = "Zażółć gęślą jaźń! Don't \"quote\"; test: ok? 2024."
    assert _clean(text) == text


def test_stage1_sanitizer_removes_leading_dash_markers() -> None:
    assert _clean("   - Hello") == "Hello"
    assert _clean("— Hello") == "Hello"
    assert _clean("-- Hello") == "Hello"


def test_stage1_sanitizer_keeps_dash_after_text_starts() -> None:
    assert _clean("Bielsko-Biała") == "Bielsko-Biała"
    assert _clean("COVID-19") == "COVID-19"
    assert _clean("Hello - world") == "Hello - world"


def test_stage1_preaudit_drops_rows_cleaned_to_empty() -> None:
    dataset = Dataset.from_list(
        [
            {"translation": {"pl": "你好 😊", "en": "valid text"}},
            {"translation": {"pl": "dobry tekst", "en": "valid text"}},
        ]
    )

    rows, dropped = preaudit_filter_rows(
        dataset,
        PreAuditConfig(),
        split_name="test",
        show_progress=False,
    )

    assert dropped == {"empty_pair": 1}
    assert rows == [{"row_index": 1, "pl": "dobry tekst", "en": "valid text"}]
