#!/usr/bin/env python3
"""Convert a Markdown file to PDF using md2pdf."""

import argparse
import sys
from pathlib import Path


def convert(md_path: Path, pdf_path: Path | None = None) -> Path:
    """Convert md_path to PDF. Returns the output path."""
    try:
        from md2pdf.core import md2pdf
    except ImportError:
        print('Install md2pdf first: pip install "md2pdf[cli]"', file=sys.stderr)
        sys.exit(1)
    out = pdf_path or md_path.with_suffix(".pdf")
    md2pdf(str(out), md_file_path=str(md_path))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Markdown to PDF.")
    parser.add_argument("input", nargs="?", default="report.md", help="Input .md file")
    parser.add_argument("-o", "--output", help="Output .pdf file (default: same name as input)")
    args = parser.parse_args()

    md_path = Path(args.input)
    if not md_path.is_file():
        print(f"Error: {md_path} not found.", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(args.output) if args.output else None
    out = convert(md_path, pdf_path)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
