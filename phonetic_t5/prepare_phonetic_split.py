#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import epitran
import pandas as pd
import pycountry
from epitran import meta

try:
    from phonemizer.backend import EspeakBackend

    PHONEMIZER_AVAILABLE = True
except Exception:
    EspeakBackend = None
    PHONEMIZER_AVAILABLE = False

try:
    from uroman import Uroman

    UROMAN_AVAILABLE = True
except Exception:
    Uroman = None
    UROMAN_AVAILABLE = False

TAG_RE = re.compile(r"\[([^\]]+)\]")

# Manual language-tag overrides for frequent tags in mixed-tag datasets.
# Values are Epitran codes.
EPITRAN_OVERRIDES: Dict[str, str] = {
    "fr": "fra-Latn",
    "it": "ita-Latn",
    "pt": "por-Latn",
    "ro": "ron-Latn",
    "nl": "nld-Latn",
    "de": "deu-Latn",
    "ca": "cat-Latn",
    "cs": "ces-Latn",
    "pl": "pol-Latn",
    "ru": "rus-Cyrl",
    "fa": "fas-Arab",
    "hi": "hin-Deva",
    # Some tags already appear as ISO-639-3 in the data.
    "fra": "fra-Latn",
    "ita": "ita-Latn",
    "por": "por-Latn",
    "ron": "ron-Latn",
    "nld": "nld-Latn",
    "deu": "deu-Latn",
    "cat": "cat-Latn",
    "ces": "ces-Latn",
    "pol": "pol-Latn",
    "rus": "rus-Cyrl",
    "fas": "fas-Arab",
    "hin": "hin-Deva",
}

# Manual eSpeak fallback overrides.
ESPEAK_OVERRIDES: Dict[str, str] = {
    "en": "en-us",
    "eng": "en-us",
    "fr": "fr-fr",
    "fra": "fr-fr",
    "la": "la",
    "lat": "la",
    "el": "el",
    "ell": "el",
    "grc": "grc",
    "nb": "nb",
    "nob": "nb",
    "nn": "nb",
    "nno": "nb",
    "old": "nb",
}


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phonetic_split")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(out_dir / "phoneticization_run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def resolve_iso3(tag: str) -> Optional[str]:
    tag = normalize_tag(tag)
    tag = tag.split("-")[0].split("_")[0]

    if len(tag) == 2:
        lang = pycountry.languages.get(alpha_2=tag)
        return getattr(lang, "alpha_3", None) if lang else None

    if len(tag) == 3:
        lang = pycountry.languages.get(alpha_3=tag)
        return tag if lang else None

    return None


def resolve_epitran_code(tag: str) -> Optional[str]:
    t = normalize_tag(tag)
    if t in EPITRAN_OVERRIDES:
        return EPITRAN_OVERRIDES[t]

    iso3 = resolve_iso3(t)
    if not iso3:
        return None

    if meta.supported_lang(iso3):
        return meta.get_default_mode(iso3)
    return None


def get_espeak_voices() -> set[str]:
    try:
        out = subprocess.check_output(
            ["espeak-ng", "--voices"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return set()

    voices = set()
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            voices.add(parts[1].strip().lower())
    return voices


def resolve_espeak_code(tag: str, voices: set[str]) -> Optional[str]:
    t = normalize_tag(tag)
    base = t.split("-")[0].split("_")[0]

    # Direct/manual mappings first.
    if t in ESPEAK_OVERRIDES and ESPEAK_OVERRIDES[t] in voices:
        return ESPEAK_OVERRIDES[t]
    if base in ESPEAK_OVERRIDES and ESPEAK_OVERRIDES[base] in voices:
        return ESPEAK_OVERRIDES[base]

    # Exact tag match.
    if t in voices:
        return t
    if base in voices:
        return base

    # ISO resolution route.
    iso3 = resolve_iso3(base)
    if iso3 and iso3 in voices:
        return iso3

    if iso3:
        lang = pycountry.languages.get(alpha_3=iso3)
        iso2 = getattr(lang, "alpha_2", None) if lang else None
        if iso2 and iso2 in voices:
            return iso2

    # Try alpha2 -> alpha3 direction.
    if len(base) == 2:
        lang = pycountry.languages.get(alpha_2=base)
        iso3 = getattr(lang, "alpha_3", None) if lang else None
        if iso3 and iso3 in voices:
            return iso3

    return None


def safe_transliterate(epi: epitran.Epitran, form: str) -> str:
    # Token-wise transliteration to preserve token boundaries in multi-word forms.
    tokens = form.split()
    if not tokens:
        return form
    out = []
    for tok in tokens:
        try:
            out.append(epi.transliterate(tok))
        except Exception:
            out.append(tok)
    return " ".join(out)


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).split())


def safe_uroman(uro: Any, form: str) -> str:
    tokens = form.split()
    if not tokens:
        return form

    out = []
    for tok in tokens:
        try:
            out.append(uro.romanize_string(tok))
        except Exception:
            out.append(tok)
    return normalize_spaces(" ".join(out))


def parse_tag_set(value: str) -> set[str]:
    if not value:
        return set()
    return {normalize_tag(x) for x in value.split(",") if x.strip()}


def convert_input(
    text: str,
    epi_cache: Dict[str, epitran.Epitran],
    espeak_cache: Dict[str, EspeakBackend],
    espeak_voices: set[str],
    uroman_engine: Any,
    tag_stats: Dict[str, Counter],
    reason_stats: Counter,
    use_espeak_fallback: bool,
    use_uroman_fallback: bool,
    espeak_allow_tags: set[str],
    espeak_block_tags: set[str],
    espeak_logger: logging.Logger,
) -> Tuple[str, bool]:
    s = str(text)
    matches = list(TAG_RE.finditer(s))
    if not matches:
        return s, False

    parts = []
    any_changed = False

    for i, m in enumerate(matches):
        tag = m.group(1)
        tag_n = normalize_tag(tag)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        raw_form = s[start:end].strip()

        tag_stats[tag_n]["segments_total"] += 1

        converted_form = raw_form
        handled = False

        epi_code = resolve_epitran_code(tag_n)
        if epi_code:
            if epi_code not in epi_cache:
                try:
                    epi_cache[epi_code] = epitran.Epitran(epi_code)
                except Exception:
                    epi_cache[epi_code] = None

            epi = epi_cache.get(epi_code)
            if epi is not None:
                handled = True
                tag_stats[tag_n]["segments_with_epitran_model"] += 1
                tag_stats[tag_n]["segments_with_any_model"] += 1
                tag_stats[tag_n]["segments_with_model"] += 1
                converted_form = safe_transliterate(epi, raw_form)
                if converted_form != raw_form:
                    tag_stats[tag_n]["segments_changed"] += 1
                    any_changed = True
            else:
                reason_stats["epitran_init_failed"] += 1
        else:
            reason_stats["no_epitran_mapping"] += 1

        # Fallback to eSpeak if requested and Epitran path did not handle this segment.
        if (not handled) and use_espeak_fallback and PHONEMIZER_AVAILABLE and espeak_voices:
            base_tag = tag_n.split("-")[0].split("_")[0]
            if espeak_allow_tags and (tag_n not in espeak_allow_tags) and (base_tag not in espeak_allow_tags):
                reason_stats["espeak_not_allowlisted"] += 1
                espeak_code = None
            elif (tag_n in espeak_block_tags) or (base_tag in espeak_block_tags):
                reason_stats["espeak_blocked_tag"] += 1
                espeak_code = None
            else:
                espeak_code = resolve_espeak_code(tag_n, espeak_voices)

            if espeak_code:
                if espeak_code not in espeak_cache:
                    try:
                        espeak_cache[espeak_code] = EspeakBackend(
                            language=espeak_code,
                            preserve_punctuation=True,
                            with_stress=True,
                            language_switch="remove-flags",
                            logger=espeak_logger,
                        )
                    except Exception:
                        espeak_cache[espeak_code] = None

                backend = espeak_cache.get(espeak_code)
                if backend is not None:
                    handled = True
                    tag_stats[tag_n]["segments_with_espeak_model"] += 1
                    tag_stats[tag_n]["segments_with_any_model"] += 1
                    tag_stats[tag_n]["segments_with_model"] += 1
                    try:
                        converted_form = backend.phonemize([raw_form], strip=True, njobs=1)[0]
                    except Exception:
                        converted_form = raw_form
                    if converted_form != raw_form:
                        tag_stats[tag_n]["segments_changed"] += 1
                        any_changed = True
                else:
                    reason_stats["espeak_init_failed"] += 1
            else:
                reason_stats["no_espeak_mapping"] += 1

        # Last-resort fallback: uroman for unsupported scripts/tags.
        if (not handled) and use_uroman_fallback and UROMAN_AVAILABLE and uroman_engine is not None:
            handled = True
            tag_stats[tag_n]["segments_with_uroman_model"] += 1
            tag_stats[tag_n]["segments_with_any_model"] += 1
            tag_stats[tag_n]["segments_with_model"] += 1
            converted_form = safe_uroman(uroman_engine, raw_form)
            if converted_form != raw_form:
                tag_stats[tag_n]["segments_changed"] += 1
                any_changed = True

        if not handled:
            tag_stats[tag_n]["segments_no_model"] += 1

        part = f"[{tag}]"
        if converted_form:
            part += f" {converted_form}"
        parts.append(part)

    return " ".join(parts), any_changed


def process_split_file(
    split_name: str,
    src_path: Path,
    dst_path: Path,
    epi_cache: Dict[str, epitran.Epitran],
    espeak_cache: Dict[str, EspeakBackend],
    espeak_voices: set[str],
    uroman_engine: Any,
    tag_stats: Dict[str, Counter],
    reason_stats: Counter,
    keep_raw_column: bool,
    use_espeak_fallback: bool,
    use_uroman_fallback: bool,
    espeak_allow_tags: set[str],
    espeak_block_tags: set[str],
    espeak_logger: logging.Logger,
    logger: logging.Logger,
    log_every: int,
) -> Dict[str, int]:
    t0 = time.perf_counter()
    logger.info(f"[split:{split_name}] reading {src_path}")
    df = pd.read_csv(src_path)
    if "input" not in df.columns:
        raise ValueError(f"Missing 'input' column in {src_path}")
    logger.info(f"[split:{split_name}] rows_in_csv={len(df)}")

    rows_changed = 0
    converted_inputs = []

    for idx, val in enumerate(df["input"].astype(str), start=1):
        new_val, changed = convert_input(
            text=val,
            epi_cache=epi_cache,
            espeak_cache=espeak_cache,
            espeak_voices=espeak_voices,
            uroman_engine=uroman_engine,
            tag_stats=tag_stats,
            reason_stats=reason_stats,
            use_espeak_fallback=use_espeak_fallback,
            use_uroman_fallback=use_uroman_fallback,
            espeak_allow_tags=espeak_allow_tags,
            espeak_block_tags=espeak_block_tags,
            espeak_logger=espeak_logger,
        )
        converted_inputs.append(new_val)
        if changed:
            rows_changed += 1
        if log_every > 0 and idx % log_every == 0:
            logger.info(f"[split:{split_name}] processed={idx}/{len(df)} changed_so_far={rows_changed}")

    if keep_raw_column and "input_raw" not in df.columns:
        df.insert(df.columns.get_loc("input") + 1, "input_raw", df["input"])

    df["input"] = converted_inputs

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_path, index=False)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"[split:{split_name}] wrote {dst_path} rows={len(df)} rows_changed={rows_changed} "
        f"changed_ratio={(rows_changed / len(df)) if len(df) else 0.0:.4f} elapsed_sec={elapsed:.2f}"
    )

    return {
        "rows": int(len(df)),
        "rows_changed": int(rows_changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build phoneticized split files from [lang] form inputs")
    parser.add_argument(
        "--source-dir",
        default="/root/PIE-reconstruction/dataset/splits/iecor_kaikki_koebler_normalized",
        help="Directory with train.csv/val.csv/test.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/PIE-reconstruction/dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input",
        help="Output directory for phoneticized splits",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split names to process",
    )
    parser.add_argument(
        "--no-input-raw",
        action="store_true",
        help="Do not add input_raw backup column",
    )
    parser.add_argument(
        "--use-espeak-fallback",
        action="store_true",
        help="Enable eSpeak fallback (via phonemizer) when Epitran has no model for a tag",
    )
    parser.add_argument(
        "--use-uroman-fallback",
        action="store_true",
        help="Enable uroman fallback when neither Epitran nor eSpeak are available for a segment",
    )
    parser.add_argument(
        "--espeak-allow-tags",
        default="",
        help="Comma-separated tag allowlist for eSpeak fallback (empty = all tags)",
    )
    parser.add_argument(
        "--espeak-block-tags",
        default="",
        help="Comma-separated tag blocklist for eSpeak fallback",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N rows within a split (0 disables progress logs)",
    )
    args = parser.parse_args()

    src_dir = Path(args.source_dir)
    out_dir = Path(args.output_dir)
    logger = setup_logger(out_dir)
    t_all = time.perf_counter()
    logger.info("=== phonetic split build started ===")
    logger.info(f"source_dir={src_dir}")
    logger.info(f"output_dir={out_dir}")
    logger.info(f"splits={args.splits}")
    logger.info(f"use_espeak_fallback={args.use_espeak_fallback} use_uroman_fallback={args.use_uroman_fallback}")
    if sys.platform.startswith("win") and not sys.flags.utf8_mode:
        logger.warning("Windows UTF-8 mode is disabled. Run with PYTHONUTF8=1 to avoid Epitran decode issues.")

    epi_cache: Dict[str, epitran.Epitran] = {}
    espeak_cache: Dict[str, EspeakBackend] = {}
    espeak_voices = get_espeak_voices() if args.use_espeak_fallback else set()
    logger.info(f"espeak_voices_discovered={len(espeak_voices)}")
    uroman_engine = Uroman() if args.use_uroman_fallback and UROMAN_AVAILABLE else None
    if args.use_uroman_fallback and not UROMAN_AVAILABLE:
        logger.warning("--use-uroman-fallback requested but uroman is not installed; continuing without uroman.")
    espeak_allow_tags = parse_tag_set(args.espeak_allow_tags)
    espeak_block_tags = parse_tag_set(args.espeak_block_tags)
    espeak_logger = logging.getLogger("phonemizer_fallback")
    espeak_logger.setLevel(logging.ERROR)
    tag_stats: Dict[str, Counter] = defaultdict(Counter)
    reason_stats: Counter = Counter()
    split_report: Dict[str, Dict[str, int]] = {}

    for split in args.splits:
        src_path = src_dir / f"{split}.csv"
        dst_path = out_dir / f"{split}.csv"
        if not src_path.exists():
            raise FileNotFoundError(f"Missing split file: {src_path}")

        rep = process_split_file(
            split_name=split,
            src_path=src_path,
            dst_path=dst_path,
            epi_cache=epi_cache,
            espeak_cache=espeak_cache,
            espeak_voices=espeak_voices,
            uroman_engine=uroman_engine,
            tag_stats=tag_stats,
            reason_stats=reason_stats,
            keep_raw_column=not args.no_input_raw,
            use_espeak_fallback=args.use_espeak_fallback,
            use_uroman_fallback=args.use_uroman_fallback,
            espeak_allow_tags=espeak_allow_tags,
            espeak_block_tags=espeak_block_tags,
            espeak_logger=espeak_logger,
            logger=logger,
            log_every=args.log_every,
        )
        split_report[split] = rep
        logger.info(f"[split:{split}] rows={rep['rows']} rows_changed={rep['rows_changed']}")

    # Aggregate stats
    total_segments = sum(v["segments_total"] for v in tag_stats.values())
    total_with_epitran_model = sum(v["segments_with_epitran_model"] for v in tag_stats.values())
    total_with_espeak_model = sum(v["segments_with_espeak_model"] for v in tag_stats.values())
    total_with_uroman_model = sum(v["segments_with_uroman_model"] for v in tag_stats.values())
    total_with_any_model = sum(v["segments_with_any_model"] for v in tag_stats.values())
    total_with_model = sum(v["segments_with_model"] for v in tag_stats.values())
    total_changed = sum(v["segments_changed"] for v in tag_stats.values())

    report = {
        "source_dir": str(src_dir),
        "output_dir": str(out_dir),
        "splits": split_report,
        "summary": {
            "unique_tags": len(tag_stats),
            "total_segments": int(total_segments),
            "segments_with_epitran_model": int(total_with_epitran_model),
            "segments_with_espeak_model": int(total_with_espeak_model),
            "segments_with_uroman_model": int(total_with_uroman_model),
            "segments_with_any_model": int(total_with_any_model),
            "segments_with_model": int(total_with_model),
            "segments_changed": int(total_changed),
            "segments_with_any_model_ratio": float(total_with_any_model / total_segments) if total_segments else 0.0,
            "segments_with_model_ratio": float(total_with_model / total_segments) if total_segments else 0.0,
            "segments_changed_ratio": float(total_changed / total_segments) if total_segments else 0.0,
            "epitran_models_initialized": len([k for k, v in epi_cache.items() if v is not None]),
            "espeak_voices_discovered": int(len(espeak_voices)),
            "espeak_models_initialized": len([k for k, v in espeak_cache.items() if v is not None]),
            "espeak_allow_tags_count": int(len(espeak_allow_tags)),
            "espeak_block_tags_count": int(len(espeak_block_tags)),
            "uroman_enabled": bool(args.use_uroman_fallback and UROMAN_AVAILABLE),
        },
        "reason_stats": dict(reason_stats),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phoneticization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"wrote report: {out_dir / 'phoneticization_report.json'}")

    # Per-tag table
    rows = []
    for tag, c in sorted(tag_stats.items(), key=lambda x: x[1]["segments_total"], reverse=True):
        rows.append(
            {
                "tag": tag,
                "segments_total": int(c["segments_total"]),
                "segments_with_epitran_model": int(c["segments_with_epitran_model"]),
                "segments_with_espeak_model": int(c["segments_with_espeak_model"]),
                "segments_with_uroman_model": int(c["segments_with_uroman_model"]),
                "segments_with_any_model": int(c["segments_with_any_model"]),
                "segments_with_model": int(c["segments_with_model"]),
                "segments_changed": int(c["segments_changed"]),
                "segments_no_model": int(c["segments_no_model"]),
                "with_any_model_ratio": (c["segments_with_any_model"] / c["segments_total"]) if c["segments_total"] else 0.0,
                "with_model_ratio": (c["segments_with_model"] / c["segments_total"]) if c["segments_total"] else 0.0,
                "changed_ratio": (c["segments_changed"] / c["segments_total"]) if c["segments_total"] else 0.0,
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "phoneticization_tag_stats.csv", index=False)
    logger.info(f"wrote tag stats: {out_dir / 'phoneticization_tag_stats.csv'}")
    logger.info("top reason_stats:")
    for k, v in reason_stats.most_common(10):
        logger.info(f"  {k}: {v}")
    logger.info(f"total_elapsed_sec={time.perf_counter() - t_all:.2f}")
    logger.info("=== phonetic split build finished ===")

    print("Done.")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
