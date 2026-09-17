"""Build redistributable dataset variants from the internal paper splits.

The paper benchmark contains Koebler-derived rows whose redistribution terms
could not be verified. This script removes those rows, labels EtymologyDB rows
that had an empty source field in the research files, and writes manifests with
row counts and SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


OPEN_BASE_SOURCES = {"iecor", "kaikki"}
OPEN_EXPANDED_SOURCES = OPEN_BASE_SOURCES | {"etymology_db"}
KNOWN_SOURCES = OPEN_EXPANDED_SOURCES | {"koebler", "starling"}


def read_rows(path: Path, blank_source: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

    for row in rows:
        source = (row.get("source") or "").strip().lower()
        row["source"] = source or blank_source
        if row["source"] not in KNOWN_SOURCES:
            raise ValueError(f"Unknown source {row['source']!r} in {path}")
    return rows


def filter_sources(
    rows: Iterable[dict[str, str]], allowed_sources: set[str]
) -> list[dict[str, str]]:
    return [row for row in rows if row["source"] in allowed_sources]


def add_release_ids(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    occurrences: defaultdict[str, int] = defaultdict(int)
    released: list[dict[str, str]] = []

    for row in rows:
        identity = "\x1f".join(
            row.get(column, "")
            for column in ("source", "cognate_set_id", "input", "output")
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        occurrences[digest] += 1
        released.append(
            {
                "release_id": f"{split}-{digest}-{occurrences[digest]:02d}",
                **row,
            }
        )
    return released


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty split: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_variant(
    output_root: Path,
    name: str,
    splits: dict[str, list[dict[str, str]]],
    description: str,
) -> None:
    variant_dir = output_root / name
    files: dict[str, dict[str, object]] = {}

    for split, rows in splits.items():
        released = add_release_ids(rows, split)
        path = variant_dir / f"{split}.csv"
        write_csv(path, released)
        files[split] = {
            "path": path.name,
            "rows": len(released),
            "sources": dict(sorted(Counter(r["source"] for r in released).items())),
            "sha256": sha256(path),
        }

    manifest = {
        "name": name,
        "description": description,
        "paper_exact_split": False,
        "exclusions": ["koebler", "starling"],
        "files": files,
    }
    manifest_path = variant_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_release(
    input_dir: Path,
    output_dir: Path,
    validated_train: Path | None,
    blank_source: str,
) -> None:
    canonical = {
        split: read_rows(input_dir / f"{split}.csv", blank_source)
        for split in ("train", "val", "test")
    }

    write_variant(
        output_dir,
        "open_base",
        {
            split: filter_sources(rows, OPEN_BASE_SOURCES)
            for split, rows in canonical.items()
        },
        "IE-CoR and Kaikki rows. Koebler and Starling rows are excluded.",
    )

    write_variant(
        output_dir,
        "open_expanded",
        {
            split: filter_sources(rows, OPEN_EXPANDED_SOURCES)
            for split, rows in canonical.items()
        },
        "Open base plus EtymologyDB-derived training rows.",
    )

    if validated_train is not None:
        validated_rows = filter_sources(
            read_rows(validated_train, blank_source), OPEN_BASE_SOURCES
        )
        etymology_rows = filter_sources(
            canonical["train"], {"etymology_db"}
        )
        write_variant(
            output_dir,
            "open_validated_synthetic_expanded",
            {
                "train": validated_rows + etymology_rows,
                "val": filter_sources(canonical["val"], OPEN_BASE_SOURCES),
                "test": filter_sources(canonical["test"], OPEN_BASE_SOURCES),
            },
            "Validated synthetic base plus EtymologyDB rows; public analogue, not the exact paper V11 split.",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the internal train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/splits"),
        help="Directory for generated open variants.",
    )
    parser.add_argument(
        "--validated-train",
        type=Path,
        help="Optional validated synthetic training CSV.",
    )
    parser.add_argument(
        "--blank-source",
        default="etymology_db",
        choices=sorted(KNOWN_SOURCES),
        help="Source assigned to blank source fields in the internal files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_release(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        validated_train=args.validated_train,
        blank_source=args.blank_source,
    )


if __name__ == "__main__":
    main()
