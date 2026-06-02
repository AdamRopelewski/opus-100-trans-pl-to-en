from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from datasets import Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class PreAuditConfig:
    deduplicate_pairs: bool = True
    remove_identical_pairs: bool = True
    remove_square_bracket_content: bool = True
    min_words: int = 1
    max_words: int = 200
    max_length_ratio: float = 4.0


_ALNUM_PL_EN = "A-Za-z0-9ĄąĆćĘęŁłŃńÓóŚśŹźŻż"
_ALLOWED_PUNCTUATION = r"\?\.,:;'!\"\-"


def normalize_preaudit_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_square_bracket_content(text: str) -> str:
    # Remove subtitle-like cues such as [SIGHS], [MUSIC], [APPLAUSE]
    return re.sub(r"\[[^\]]*\]", " ", text)


def sanitize_preaudit_text(text: str) -> str:
    out = text
    # Normalize dash variants before final dash filtering.
    out = out.replace("–", "-").replace("—", "-").replace("−", "-")

    # Keep ellipsis only at end, preceded by letter/digit.
    out = re.sub(rf"(?<![{_ALNUM_PL_EN}])\.{{3}}", " ", out)
    out = re.sub(rf"\.{{3}}(?!\s*$)", " ", out)

    # Remove all characters except plain text whitelist.
    out = re.sub(rf"[^ {_ALNUM_PL_EN}{_ALLOWED_PUNCTUATION}]", " ", out)

    # Drop dialogue markers; keep later dashes only after text has started.
    out = re.sub(r"^\s*-+\s*", "", out)
    chars: list[str] = []
    seen_text = False
    for char in out:
        if re.match(rf"[{_ALNUM_PL_EN}]", char):
            seen_text = True
        if char == "-" and not seen_text:
            chars.append(" ")
            continue
        chars.append(char)
    out = "".join(chars)

    return out


def pair_key(pl: str, en: str) -> str:
    return f"{pl}\x1f{en}"


def word_count(text: str) -> int:
    return len(text.split())


def preaudit_filter_rows(
    dataset: Dataset,
    cfg: PreAuditConfig,
    split_name: str,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped: Counter[str] = Counter()
    iterator = enumerate(dataset)
    if show_progress and tqdm is not None:
        iterator = enumerate(
            tqdm(
                dataset,
                total=len(dataset),
                desc=f"Preaudit [{split_name}]",
                unit="rows",
                leave=False,
            )
        )
    for idx, row in iterator:
        tr = row.get("translation")
        if not isinstance(tr, dict):
            dropped["null_pair"] += 1
            continue
        pl = str(tr.get("pl", ""))
        en = str(tr.get("en", ""))
        if cfg.remove_square_bracket_content:
            pl = strip_square_bracket_content(pl)
            en = strip_square_bracket_content(en)
        pl = sanitize_preaudit_text(pl)
        en = sanitize_preaudit_text(en)
        pl = normalize_preaudit_text(pl)
        en = normalize_preaudit_text(en)
        if not pl or not en:
            dropped["empty_pair"] += 1
            continue
        if cfg.remove_identical_pairs and pl == en:
            dropped["identical_source_target"] += 1
            continue
        plw = word_count(pl)
        enw = word_count(en)
        if plw < cfg.min_words or enw < cfg.min_words:
            dropped["min_words"] += 1
            continue
        if plw > cfg.max_words or enw > cfg.max_words:
            dropped["max_words"] += 1
            continue
        ratio = max(plw, enw) / max(1, min(plw, enw))
        if ratio > cfg.max_length_ratio:
            dropped["length_ratio"] += 1
            continue
        key = pair_key(pl, en)
        if cfg.deduplicate_pairs and key in seen:
            dropped["duplicate_pair"] += 1
            continue
        seen.add(key)
        rows.append({"row_index": idx, "pl": pl, "en": en})
    return rows, dict(sorted(dropped.items()))
