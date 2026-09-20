#!/usr/bin/env python3
"""Validate classification front matter against the Paper Hub taxonomy."""
import argparse
import sys
from pathlib import Path

from classify_papers import ROOT, load_taxonomy, split_front_matter, validate_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Markdown files or directories (default: papers)")
    parser.add_argument("--require-classification", action="store_true", help="Report Markdown files without a classification block")
    args = parser.parse_args()
    inputs = args.paths or [ROOT / "papers"]
    files = sorted({file for item in inputs for file in ((item.rglob("*.md")) if item.is_dir() else [item]) if file.is_file() and file.suffix.lower() == ".md"})
    allowed, rules, _ = load_taxonomy()
    failures = 0
    classified = 0
    for path in files:
        metadata, _ = split_front_matter(path.read_text(encoding="utf-8"))
        classification = metadata.get("classification")
        if not classification:
            if args.require_classification:
                print(f"ERROR {path}: no classification")
                failures += 1
            continue
        classified += 1
        if not isinstance(classification, dict):
            print(f"ERROR {path}: classification must be a mapping")
            failures += 1
            continue
        errors = validate_result(classification, allowed, rules)
        if errors:
            print(f"ERROR {path}: {'; '.join(errors)}")
            failures += 1
    print(f"Validated {classified} classified paper(s); {failures} error(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
