#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

TAG_RE = re.compile(r"\[([^\]]+)\]")


def normalize_text(text: str, strip_markers: bool = True) -> str:
    s = unicodedata.normalize("NFC", str(text).strip())
    s = re.sub(r"\s+", " ", s)
    if strip_markers:
        s = s.lstrip("*?")
        s = s.rstrip("-")
    return s.strip()


def split_graphemes(s: str) -> List[str]:
    try:
        import regex as re2  # type: ignore

        return [g for g in re2.findall(r"\X", s) if g and not g.isspace()]
    except Exception:
        return [ch for ch in s if not ch.isspace()]


def tokenize_form(form: str) -> List[str]:
    tokens: List[str] = []
    for part in form.split():
        tokens.extend(split_graphemes(part))
    return tokens


def parse_input_field(
    text: str,
    dedupe_strategy: str,
    strip_markers: bool,
) -> Tuple[Dict[str, List[str]], Counter]:
    stats = Counter()
    parsed: Dict[str, List[str]] = {}

    s = str(text)
    matches = list(TAG_RE.finditer(s))
    for i, m in enumerate(matches):
        lang = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        raw_form = s[start:end].strip()
        if not raw_form:
            stats["empty_segments"] += 1
            continue

        form = normalize_text(raw_form, strip_markers=strip_markers)
        segs = tokenize_form(form)
        if not segs:
            stats["empty_tokenized_segments"] += 1
            continue

        if lang in parsed:
            stats["duplicate_lang_segments"] += 1
            if dedupe_strategy == "first":
                continue
            if dedupe_strategy == "longest":
                if len(segs) > len(parsed[lang]):
                    parsed[lang] = segs
                continue
            raise ValueError(f"Unknown dedupe_strategy: {dedupe_strategy}")
        parsed[lang] = segs

    return parsed, stats


def parse_row(
    row: pd.Series,
    row_id: str,
    proto_lang: str,
    dedupe_strategy: str,
    strip_markers: bool,
    min_input_langs: int,
) -> Tuple[dict | None, Counter]:
    stats = Counter()
    inputs, input_stats = parse_input_field(
        text=str(row["input"]),
        dedupe_strategy=dedupe_strategy,
        strip_markers=strip_markers,
    )
    stats.update(input_stats)

    target_raw = normalize_text(str(row["output"]), strip_markers=strip_markers)
    target_tokens = tokenize_form(target_raw)
    if not target_tokens:
        stats["rows_skipped_empty_target"] += 1
        return None, stats
    if len(inputs) < min_input_langs:
        stats["rows_skipped_too_few_inputs"] += 1
        return None, stats

    entry = {
        "id": row_id,
        "inputs": inputs,
        "target": {proto_lang: target_tokens},
    }
    return entry, stats


def parse_split(
    csv_path: Path,
    split_name: str,
    proto_lang: str,
    dedupe_strategy: str,
    strip_markers: bool,
    min_input_langs: int,
) -> Tuple[List[dict], Counter]:
    df = pd.read_csv(csv_path)
    if "input" not in df.columns or "output" not in df.columns:
        raise ValueError(f"{csv_path} must contain input/output columns")

    entries: List[dict] = []
    stats = Counter()
    for i, row in enumerate(df.itertuples(index=False), 1):
        row_series = pd.Series(row._asdict())
        row_id = f"{split_name}-{i}"
        entry, row_stats = parse_row(
            row=row_series,
            row_id=row_id,
            proto_lang=proto_lang,
            dedupe_strategy=dedupe_strategy,
            strip_markers=strip_markers,
            min_input_langs=min_input_langs,
        )
        stats.update(row_stats)
        if entry is not None:
            entries.append(entry)
    stats["rows_in_csv"] = int(len(df))
    stats["rows_kept"] = int(len(entries))
    return entries, stats


def collect_languages(entries: Iterable[dict]) -> List[str]:
    langs = set()
    for e in entries:
        langs.update(e["inputs"].keys())
        langs.update(e["target"].keys())
    return sorted(langs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PIE split CSVs for FeVeT-style proto training.")
    parser.add_argument(
        "--train-csv",
        default="dataset/splits/iecor_kaikki_koebler_normalized/train.csv",
    )
    parser.add_argument(
        "--val-csv",
        default="dataset/splits/iecor_kaikki_koebler_normalized/val.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="dataset/splits/iecor_kaikki_koebler_normalized/test.csv",
    )
    parser.add_argument(
        "--output-json",
        default="fevet_pie/data/pie_iecor_prepared.json",
    )
    parser.add_argument("--proto-lang", default="protopie")
    parser.add_argument(
        "--dedupe-strategy",
        choices=["first", "longest"],
        default="first",
        help="How to resolve duplicate [lang] forms in one input row.",
    )
    parser.add_argument(
        "--no-strip-markers",
        action="store_true",
        help="Keep leading *, ? and trailing - in forms.",
    )
    parser.add_argument(
        "--min-input-langs",
        type=int,
        default=1,
        help="Drop rows with fewer input languages.",
    )
    args = parser.parse_args()

    train_csv = Path(args.train_csv).resolve()
    val_csv = Path(args.val_csv).resolve()
    test_csv = Path(args.test_csv).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    strip_markers = not args.no_strip_markers

    train_entries, train_stats = parse_split(
        csv_path=train_csv,
        split_name="train",
        proto_lang=args.proto_lang,
        dedupe_strategy=args.dedupe_strategy,
        strip_markers=strip_markers,
        min_input_langs=args.min_input_langs,
    )
    val_entries, val_stats = parse_split(
        csv_path=val_csv,
        split_name="val",
        proto_lang=args.proto_lang,
        dedupe_strategy=args.dedupe_strategy,
        strip_markers=strip_markers,
        min_input_langs=args.min_input_langs,
    )
    test_entries, test_stats = parse_split(
        csv_path=test_csv,
        split_name="test",
        proto_lang=args.proto_lang,
        dedupe_strategy=args.dedupe_strategy,
        strip_markers=strip_markers,
        min_input_langs=args.min_input_langs,
    )

    all_entries = train_entries + val_entries + test_entries
    langs = collect_languages(all_entries)

    payload = {
        "meta": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "proto_lang": args.proto_lang,
            "dedupe_strategy": args.dedupe_strategy,
            "strip_markers": strip_markers,
            "min_input_langs": args.min_input_langs,
        },
        "languages": langs,
        "splits": {
            "train": train_entries,
            "val": val_entries,
            "test": test_entries,
        },
        "stats": {
            "train": dict(train_stats),
            "val": dict(val_stats),
            "test": dict(test_stats),
            "languages_total": len(langs),
        },
    }

    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Prepared dataset written to:")
    print(output_json)
    print("Stats:")
    print(json.dumps(payload["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
