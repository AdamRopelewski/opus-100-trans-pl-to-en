from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from src.utils.pipeline_constants import (
    DEFAULT_LLM_BATCH_MAX_CHARS,
    DEFAULT_LLM_MAX_BATCH_RETRIES,
    DEFAULT_LLM_MAX_ROWS_PER_BATCH,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_UNCERTAIN_RATIO_RERUN_THRESHOLD,
    DEFAULT_OLLAMA_ENDPOINT,
)


@dataclass
class LlmAuditConfig:
    model: str = DEFAULT_LLM_MODEL
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    batch_max_chars: int = DEFAULT_LLM_BATCH_MAX_CHARS
    max_rows_per_batch: int = DEFAULT_LLM_MAX_ROWS_PER_BATCH
    temperature: float = DEFAULT_LLM_TEMPERATURE
    max_batch_retries: int = DEFAULT_LLM_MAX_BATCH_RETRIES
    uncertain_ratio_rerun_threshold: float = DEFAULT_LLM_UNCERTAIN_RATIO_RERUN_THRESHOLD
    verbose: bool = False
    verbose_preview_rows: int = 10


@dataclass
class PreAuditConfig:
    deduplicate_pairs: bool = True
    remove_identical_pairs: bool = True
    remove_square_bracket_content: bool = True
    min_words: int = 1
    max_words: int = 200
    max_length_ratio: float = 4.0


class UncertainBatchError(RuntimeError):
    pass


LABEL_TO_INT = {"bad": 0, "uncertain": 1, "good": 2}
INT_TO_LABEL = {0: "bad", 1: "uncertain", 2: "good"}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_square_bracket_content(text: str) -> str:
    # Remove subtitle-like cues such as [SIGHS], [MUSIC], [APPLAUSE]
    return re.sub(r"\[[^\]]*\]", " ", text)


_ALNUM_PL_EN = "A-Za-z0-9ĄąĆćĘęŁłŃńÓóŚśŹźŻż"


def _sanitize_preaudit_text(text: str) -> str:
    out = text
    # Normalize leading whitespace and dialogue markers.
    out = out.lstrip()
    out = re.sub(r"^[\-*\s]+", "", out)

    # Keep ellipsis only at end, preceded by letter/digit.
    out = re.sub(rf"(?<![{_ALNUM_PL_EN}])\.{{3}}", " ", out)
    out = re.sub(rf"\.{{3}}(?!\s*$)", " ", out)

    # Remove all characters except plain text whitelist.
    out = re.sub(rf"[^ {_ALNUM_PL_EN}\-\?,\.:;'!]", " ", out)

    # Keep '-' only when between letters/digits.
    out = re.sub(rf"(?<![{_ALNUM_PL_EN}])-(?=[{_ALNUM_PL_EN}]|\s|$)", " ", out)
    out = re.sub(rf"(?<=[{_ALNUM_PL_EN}])-(?![{_ALNUM_PL_EN}])", " ", out)

    return out


def _pair_key(pl: str, en: str) -> str:
    return f"{pl}\x1f{en}"


def _word_count(text: str) -> int:
    return len(text.split())


def _preaudit_filter_rows(
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
            pl = _strip_square_bracket_content(pl)
            en = _strip_square_bracket_content(en)
        pl = _sanitize_preaudit_text(pl)
        en = _sanitize_preaudit_text(en)
        pl = _normalize_text(pl)
        en = _normalize_text(en)
        if not pl or not en:
            dropped["empty_pair"] += 1
            continue
        if cfg.remove_identical_pairs and pl == en:
            dropped["identical_source_target"] += 1
            continue
        plw = _word_count(pl)
        enw = _word_count(en)
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
        key = _pair_key(pl, en)
        if cfg.deduplicate_pairs and key in seen:
            dropped["duplicate_pair"] += 1
            continue
        seen.add(key)
        rows.append({"row_index": idx, "pl": pl, "en": en})
    return rows, dict(sorted(dropped.items()))


def _build_batches(records: list[dict[str, Any]], max_chars: int, max_rows: int) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for rec in records:
        line = f"id:{rec['row_index']} pl:{rec['pl']} en:{rec['en']}"
        line_chars = len(line)
        if current and (len(current) >= max_rows or current_chars + line_chars > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(rec)
        current_chars += line_chars
    if current:
        batches.append(current)
    return batches


def _prompt_for_batch(batch: list[dict[str, Any]]) -> str:
    rows = "\n".join([f"id:{idx} pl:{r['pl']} en:{r['en']}" for idx, r in enumerate(batch)])
    return (
        "You are auditing Polish-English translation pairs.\n"
        "The translation doesnt have to be perfect, you only fillter out the obvious bad ones.\n"
        "Classify each id into exactly one bucket:\n"
        "- good: valid translation pair\n"
        "- bad: wrong/noisy/non-translation/meta/code-like pair\n"
        "- uncertain: cannot decide reliably\n"
        "Return STRICT JSON only with keys good,bad,uncertain and integer arrays.\n"
        "No extra keys, no comments, no markdown.\n"
        "Rows:\n" + rows
    )


def _ollama_generate(cfg: LlmAuditConfig, prompt: str) -> str:
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "context": [],
        "options": {"temperature": cfg.temperature},
    }
    req = urllib.request.Request(
        cfg.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("response", "")).strip()


def _log(msg: str) -> None:
    if tqdm is not None:
        tqdm.write(msg)
        return
    print(msg)


def _compact_label_log(valid: dict[str, list[int]]) -> str:
    def _fmt(name: str) -> str:
        values = valid.get(name, [])
        ids = ",".join(str(x) for x in values)
        return f"{name}({len(values)}):{ids}"

    return " | ".join([_fmt("good"), _fmt("bad"), _fmt("uncertain")])


def _validate_response(raw: str, batch_ids: set[int]) -> dict[str, list[int]]:
    parsed = json.loads(raw)
    for key in ("good", "bad", "uncertain"):
        if key not in parsed or not isinstance(parsed[key], list):
            raise ValueError(f"Missing/invalid key: {key}")
        parsed[key] = [int(x) for x in parsed[key]]
    good = set(parsed["good"])
    bad = set(parsed["bad"])
    uncertain = set(parsed["uncertain"])
    if (good & bad) or (good & uncertain) or (bad & uncertain):
        raise ValueError("IDs overlap between classes")
    returned = good | bad | uncertain
    if not returned.issubset(batch_ids):
        raise ValueError("Response includes IDs outside input batch")
    missing = batch_ids - returned
    if missing:
        parsed["uncertain"].extend(sorted(missing))
    return {"good": sorted(good), "bad": sorted(bad), "uncertain": sorted(set(parsed["uncertain"]))}


def _needs_rerun(valid: dict[str, list[int]], threshold: float) -> bool:
    total = len(valid["good"]) + len(valid["bad"]) + len(valid["uncertain"])
    if total <= 0:
        return True
    uncertain_ratio = len(valid["uncertain"]) / total
    return uncertain_ratio > threshold


def _parse_label_entry(line: str) -> tuple[str, int, str] | None:
    try:
        obj = json.loads(line)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    # Compact format: {"s":"split","i":123,"l":2}
    if "s" in obj and "i" in obj and "l" in obj:
        split = str(obj["s"])
        row_index = int(obj["i"])
        label = INT_TO_LABEL.get(int(obj["l"]))
        if label is None:
            return None
        return split, row_index, label

    # Legacy format compatibility: {"split":"...","row_index":123,"label":"good"}
    if "split" in obj and "row_index" in obj and "label" in obj:
        split = str(obj["split"])
        row_index = int(obj["row_index"])
        label = str(obj["label"])
        if label not in LABEL_TO_INT:
            return None
        return split, row_index, label

    return None


def _parse_batch_entry(line: str) -> tuple[str, int, int] | None:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    try:
        split = str(obj["split"])
        batch_index = int(obj["batch_index"])
        size = int(obj["size"])
    except Exception:
        return None
    return split, batch_index, size


def run_stage1_llm_audit(
    split_files: dict[str, Path],
    cfg: LlmAuditConfig,
    reports_dir: Path,
    preaudit_cfg: PreAuditConfig,
    show_progress: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    labels_path = reports_dir / "llm_audit_labels.jsonl"
    batches_path = reports_dir / "llm_audit_batches.jsonl"
    bad_path = reports_dir / "llm_bad_sentences.json"
    labels_by_split_paths = {s: reports_dir / f"llm_audit_labels_{s}.jsonl" for s in split_files}
    batches_by_split_paths = {s: reports_dir / f"llm_audit_batches_{s}.jsonl" for s in split_files}
    bad_by_split_paths = {s: reports_dir / f"llm_bad_sentences_{s}.json" for s in split_files}

    processed_ids_by_split: dict[str, set[int]] = {s: set() for s in split_files}
    recovered_counts_by_split: dict[str, Counter[str]] = {s: Counter() for s in split_files}
    totals = Counter()
    total_preaudit_dropped: Counter[str] = Counter()
    split_stats: list[dict[str, Any]] = []
    all_bad_records: list[dict[str, Any]] = []
    bad_records_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in split_files}

    resume_from_split: str | None = None
    resume_from_batch_index: int | None = None

    if resume:
        if not labels_path.exists():
            raise FileNotFoundError(f"Resume requested, but labels file not found: {labels_path}")
        if not batches_path.exists():
            raise FileNotFoundError(f"Resume requested, but batches file not found: {batches_path}")

        old_lines = [ln for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        old_batch_lines = [ln for ln in batches_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

        if not old_lines:
            raise ValueError(f"Resume requested, but labels file is empty: {labels_path}")
        if not old_batch_lines:
            raise ValueError(f"Resume requested, but batches file is empty: {batches_path}")

        last_batch = _parse_batch_entry(old_batch_lines[-1])
        if last_batch is None:
            raise ValueError("Resume requested, but last batches entry is invalid JSON schema")
        last_split, last_batch_index, last_batch_size = last_batch

        # Drop last batch log line so this batch is redone.
        old_batch_lines = old_batch_lines[:-1]
        batches_path.write_text(("\n".join(old_batch_lines) + ("\n" if old_batch_lines else "")), encoding="utf-8")

        # Drop last batch_size label lines for matching split from the end.
        removed = 0
        kept_reversed: list[str] = []
        for line in reversed(old_lines):
            parsed = _parse_label_entry(line)
            if parsed is not None and parsed[0] == last_split and removed < last_batch_size:
                removed += 1
                continue
            kept_reversed.append(line)
        old_lines = list(reversed(kept_reversed))

        if removed == 0:
            raise ValueError(
                f"Resume requested, but no labels removed for last batch split={last_split}, batch={last_batch_index}"
            )

        labels_path.write_text(("\n".join(old_lines) + ("\n" if old_lines else "")), encoding="utf-8")
        _log(
            f"[audit][resume] Rewinding split={last_split} batch={last_batch_index} "
            f"by removing {removed} label rows and last batch log entry"
        )
        resume_from_split = last_split
        resume_from_batch_index = last_batch_index

        for line in old_lines:
            parsed = _parse_label_entry(line)
            if parsed is None:
                continue
            split, row_index, label = parsed
            if split not in processed_ids_by_split:
                continue
            if row_index in processed_ids_by_split[split]:
                continue
            processed_ids_by_split[split].add(row_index)
            recovered_counts_by_split[split][label] += 1
            totals[label] += 1

            if label == "bad":
                bad_rec = {
                    "split": split,
                    "row_index": int(row_index),
                    "pl": None,
                    "en": None,
                }
                all_bad_records.append(bad_rec)
                bad_records_by_split[split].append(bad_rec)

        for split_name in split_files:
            # Rebuild per-split compact file from global labels after truncation.
            compact_rows: list[str] = []
            for line in old_lines:
                parsed = _parse_label_entry(line)
                if parsed is None:
                    continue
                line_split, row_index, label = parsed
                if line_split != split_name:
                    continue
                compact_rows.append(
                    json.dumps(
                        {"i": row_index, "l": LABEL_TO_INT[label]},
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                )
            labels_by_split_paths[split_name].write_text(
                "\n".join(compact_rows) + ("\n" if compact_rows else ""),
                encoding="utf-8",
            )
            split_batch_rows: list[str] = []
            for line in old_batch_lines:
                parsed_batch = _parse_batch_entry(line)
                if parsed_batch is None:
                    continue
                if parsed_batch[0] != split_name:
                    continue
                split_batch_rows.append(line)
            batches_by_split_paths[split_name].write_text(
                "\n".join(split_batch_rows) + ("\n" if split_batch_rows else ""),
                encoding="utf-8",
            )
    else:
        for split_name in split_files:
            labels_by_split_paths[split_name].write_text("", encoding="utf-8")
            batches_by_split_paths[split_name].write_text("", encoding="utf-8")

    labels_mode = "a" if resume else "w"
    batches_mode = "a" if resume else "w"
    with labels_path.open(labels_mode, encoding="utf-8") as labels_fp, batches_path.open(batches_mode, encoding="utf-8") as batches_fp:
        split_items = list(split_files.items())
        split_iter = split_items
        if show_progress and tqdm is not None:
            split_iter = tqdm(split_items, desc="Stage1 audit splits", unit="split")

        for split, file_path in split_iter:
            dataset = Dataset.from_parquet(str(file_path))
            rows, preaudit_dropped = _preaudit_filter_rows(
                dataset,
                preaudit_cfg,
                split_name=split,
                show_progress=show_progress,
            )
            total_preaudit_dropped.update(preaudit_dropped)

            batches = _build_batches(rows, max_chars=cfg.batch_max_chars, max_rows=cfg.max_rows_per_batch)
            split_counter = Counter(recovered_counts_by_split[split])
            batch_iter = enumerate(batches)
            if show_progress and tqdm is not None:
                batch_iter = enumerate(
                    tqdm(
                        batches,
                        total=len(batches),
                        desc=f"LLM batches [{split}]",
                        unit="batch",
                        leave=False,
                    )
                )

            start_batch_index = 0
            if resume_from_split is not None and resume_from_batch_index is not None:
                split_order = list(split_files.keys())
                resume_split_pos = split_order.index(resume_from_split)
                current_split_pos = split_order.index(split)
                if current_split_pos < resume_split_pos:
                    continue
                if current_split_pos == resume_split_pos:
                    start_batch_index = resume_from_batch_index

            if start_batch_index > 0 and cfg.verbose:
                _log(f"[audit][{split}] resume start from batch_index={start_batch_index}")

            for bidx, batch in batch_iter:
                if bidx < start_batch_index:
                    continue
                if cfg.verbose:
                    _log(f"[audit][{split}] batch {bidx + 1}/{len(batches)} size={len(batch)}")
                    preview_n = max(0, min(cfg.verbose_preview_rows, len(batch)))
                    if preview_n > 0:
                        _log(f"[audit][{split}] first {preview_n} rows sent to Ollama:")
                        for rec in batch[:preview_n]:
                            _log(f"  id:{rec['row_index']} pl:{rec['pl']} en:{rec['en']}")
                local_to_row_id = {i: int(rec["row_index"]) for i, rec in enumerate(batch)}
                ids = set(local_to_row_id.keys())
                prompt = _prompt_for_batch(batch)
                attempt_logs: list[dict[str, Any]] = []
                valid: dict[str, list[int]] | None = None
                uncertain_triggered = False
                max_attempts = max(1, int(cfg.max_batch_retries) + 1)
                for attempt in range(1, max_attempts + 1):
                    try:
                        raw = _ollama_generate(cfg, prompt)
                        parsed = _validate_response(raw, ids)
                        if cfg.verbose:
                            _log(f"[audit][{split}] parsed: {_compact_label_log(parsed)}")
                        rerun_uncertain = _needs_rerun(parsed, cfg.uncertain_ratio_rerun_threshold)
                        attempt_logs.append(
                            {
                                "attempt": attempt,
                                "ok": True,
                                "uncertain_count": len(parsed["uncertain"]),
                                "batch_size": len(batch),
                                "uncertain_ratio": round(len(parsed["uncertain"]) / max(1, len(batch)), 4),
                                "rerun_due_to_uncertain": rerun_uncertain,
                            }
                        )
                        valid = parsed
                        if not rerun_uncertain:
                            break
                        uncertain_triggered = True
                    except Exception as exc:
                        attempt_logs.append({"attempt": attempt, "ok": False, "error": str(exc)})

                if valid is None:
                    valid = {"good": [], "bad": [], "uncertain": sorted(ids)}

                batches_fp.write(
                    json.dumps(
                        {
                            "split": split,
                            "batch_index": bidx,
                            "size": len(batch),
                            "attempts": attempt_logs,
                            "uncertain_triggered": uncertain_triggered,
                            "result": valid,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                with batches_by_split_paths[split].open("a", encoding="utf-8") as split_batches_fp:
                    split_batches_fp.write(
                        json.dumps(
                            {
                                "split": split,
                                "batch_index": bidx,
                                "size": len(batch),
                                "attempts": attempt_logs,
                                "uncertain_triggered": uncertain_triggered,
                                "result": valid,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

                if uncertain_triggered and _needs_rerun(valid, cfg.uncertain_ratio_rerun_threshold):
                    raise UncertainBatchError(
                        f"High uncertain ratio remained after retries for split={split}, batch={bidx}. "
                        f"Uncertain={len(valid['uncertain'])}/{len(batch)}"
                    )
                per_id = {}
                for label in ("good", "bad", "uncertain"):
                    for local_id in valid[label]:
                        per_id[local_to_row_id[local_id]] = label

                if cfg.verbose:
                    preview_n = max(0, min(cfg.verbose_preview_rows, len(batch)))
                    preview_per_label: dict[str, list[int]] = {"good": [], "bad": [], "uncertain": []}
                    for rec in batch[:preview_n]:
                        row_id = rec["row_index"]
                        label = per_id.get(row_id, "uncertain")
                        preview_per_label[label].append(int(row_id))

                    _log(f"[audit][{split}] first {preview_n} preview by label:")
                    for label in ("good", "bad", "uncertain"):
                        ids = ", ".join(str(x) for x in preview_per_label[label]) or "-"
                        _log(f"  {label}: {len(valid[label])}, {ids}")
                    _log(
                        "[audit][{split}] batch totals -> good: {good}, bad: {bad}, uncertain: {uncertain}".format(
                            split=split,
                            good=len(valid["good"]),
                            bad=len(valid["bad"]),
                            uncertain=len(valid["uncertain"]),
                        )
                    )

                for rec in batch:
                    label = per_id.get(rec["row_index"], "uncertain")
                    split_counter[label] += 1
                    totals[label] += 1
                    out = {"s": split, "i": int(rec["row_index"]), "l": LABEL_TO_INT[label]}
                    labels_fp.write(json.dumps(out, ensure_ascii=True, separators=(",", ":")) + "\n")
                    with labels_by_split_paths[split].open("a", encoding="utf-8") as split_labels_fp:
                        split_labels_fp.write(
                            json.dumps(
                                {"i": int(rec["row_index"]), "l": LABEL_TO_INT[label]},
                                ensure_ascii=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    if label == "bad":
                        bad_rec = {
                            "split": split,
                            "row_index": int(rec["row_index"]),
                            "pl": rec["pl"],
                            "en": rec["en"],
                        }
                        all_bad_records.append(bad_rec)
                        bad_records_by_split[split].append(bad_rec)

                if cfg.verbose:
                    _log(
                        "[audit][{split}] running totals -> good: {good}, bad: {bad}, uncertain: {uncertain}".format(
                            split=split,
                            good=split_counter["good"],
                            bad=split_counter["bad"],
                            uncertain=split_counter["uncertain"],
                        )
                    )

            split_stats.append(
                {
                    "split": split,
                    "file_name": file_path.name,
                    "rows_input": len(dataset),
                    "rows_audited": len(rows),
                    "preaudit_dropped": preaudit_dropped,
                    "good": split_counter["good"],
                    "bad": split_counter["bad"],
                    "uncertain": split_counter["uncertain"],
                    "batches": len(batches),
                }
            )
            if cfg.verbose:
                _log(
                    "[audit][{split}] final totals -> good: {good}, bad: {bad}, uncertain: {uncertain}".format(
                        split=split,
                        good=split_counter["good"],
                        bad=split_counter["bad"],
                        uncertain=split_counter["uncertain"],
                    )
                )

    bad_path.write_text(json.dumps({"count": len(all_bad_records), "records": all_bad_records}, ensure_ascii=True, indent=2), encoding="utf-8")
    for split_name in split_files:
        bad_by_split_paths[split_name].write_text(
            json.dumps(
                {
                    "count": len(bad_records_by_split[split_name]),
                    "records": bad_records_by_split[split_name],
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "llm": {
            "model": cfg.model,
            "endpoint": cfg.endpoint,
            "batch_max_chars": cfg.batch_max_chars,
            "max_rows_per_batch": cfg.max_rows_per_batch,
            "max_batch_retries": cfg.max_batch_retries,
            "uncertain_ratio_rerun_threshold": cfg.uncertain_ratio_rerun_threshold,
        },
        "totals": {
            "preaudit_dropped": dict(sorted(total_preaudit_dropped.items())),
            "good": totals["good"],
            "bad": totals["bad"],
            "uncertain": totals["uncertain"],
        },
        "splits": split_stats,
        "artifacts": {
            "labels_jsonl": str(labels_path),
            "batches_jsonl": str(batches_path),
            "bad_json": str(bad_path),
            "labels_jsonl_by_split": {k: str(v) for k, v in labels_by_split_paths.items()},
            "batches_jsonl_by_split": {k: str(v) for k, v in batches_by_split_paths.items()},
            "bad_json_by_split": {k: str(v) for k, v in bad_by_split_paths.items()},
        },
    }
    if cfg.verbose:
        _log(
            "[audit][all] final totals -> good: {good}, bad: {bad}, uncertain: {uncertain}".format(
                good=totals["good"],
                bad=totals["bad"],
                uncertain=totals["uncertain"],
            )
        )
    return manifest


def write_llm_audit_report(manifest: dict[str, Any], out_md: Path, out_json: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "# Data Audit Report (LLM)",
        "",
        f"Generated at (UTC): `{manifest['generated_at_utc']}`",
        f"Model: `{manifest['llm']['model']}`",
        f"Endpoint: `{manifest['llm']['endpoint']}`",
        f"Batch max chars: `{manifest['llm']['batch_max_chars']}`",
        f"Batch max rows: `{manifest['llm']['max_rows_per_batch']}`",
        f"Max batch retries: `{manifest['llm']['max_batch_retries']}`",
        "Uncertain rerun threshold: "
        f"`{manifest['llm']['uncertain_ratio_rerun_threshold']}`",
        "",
        "## Totals",
        "",
        f"- preaudit dropped: {manifest['totals']['preaudit_dropped']}",
        f"- good: {manifest['totals']['good']}",
        f"- bad: {manifest['totals']['bad']}",
        f"- uncertain (manual review): {manifest['totals']['uncertain']}",
        "",
    ]
    for split in manifest["splits"]:
        lines.extend(
            [
                f"## Split: {split['split']}",
                "",
                f"- File: `{split['file_name']}`",
                f"- Rows input: {split['rows_input']}",
                f"- Rows audited: {split['rows_audited']}",
                f"- Preaudit dropped: {split['preaudit_dropped']}",
                f"- good: {split['good']}",
                f"- bad: {split['bad']}",
                f"- uncertain: {split['uncertain']}",
                f"- batches: {split['batches']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- labels: `{manifest['artifacts']['labels_jsonl']}`",
            f"- batch logs: `{manifest['artifacts']['batches_jsonl']}`",
            f"- bad sentences: `{manifest['artifacts']['bad_json']}`",
            "",
            "### Per-split artifacts",
            "",
        ]
    )
    for split_name in ("validation", "test", "train"):
        labels_map = manifest["artifacts"].get("labels_jsonl_by_split", {})
        batches_map = manifest["artifacts"].get("batches_jsonl_by_split", {})
        bad_map = manifest["artifacts"].get("bad_json_by_split", {})
        if split_name in labels_map:
            lines.extend(
                [
                    f"- {split_name} labels: `{labels_map[split_name]}`",
                    f"- {split_name} batch logs: `{batches_map[split_name]}`",
                    f"- {split_name} bad sentences: `{bad_map[split_name]}`",
                ]
            )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
