#!/usr/bin/env python3
"""Convert VIDEO_DESCRIPTION_REPORT.md to PDF. Run from project root."""
import os
import sys

INPUT_MD = os.path.join(os.path.dirname(__file__), "VIDEO_DESCRIPTION_REPORT.md")
OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "VIDEO_DESCRIPTION_REPORT.pdf")


def main() -> None:
    if not os.path.isfile(INPUT_MD):
        print(f"Error: {INPUT_MD} not found.", file=sys.stderr)
        sys.exit(1)
    try:
        from md2pdf.core import md2pdf
    except ImportError:
        print("Install md2pdf first: pip install \"md2pdf[cli]\"", file=sys.stderr)
        sys.exit(1)
    md2pdf(OUTPUT_PDF, md=INPUT_MD)
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
