"""
One-off: turn handbook.pdf into data/handbook.md.

About half the handbook's pages were exported from Illustrator with fonts that
don't map to real characters, so plain text extraction returns junk for them.
For each page we try the embedded text first and fall back to OCR.

Needs: poppler-utils (pdftotext, pdftoppm) and tesseract on PATH.
Usage: python tools/extract_handbook.py handbook.pdf data/handbook.md

The output is committed to the repo, so you can hand-fix any OCR mistakes and
the next refresh will pick them up. You only need to re-run this if ITU
publishes a new handbook.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MIN_WORDS = 40  # below this, the embedded text is treated as unusable


def page_count(pdf: str) -> int:
    out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", out).group(1))


def embedded_text(pdf: str, page: int) -> str:
    return subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), "-layout", pdf, "-"],
        capture_output=True, text=True,
    ).stdout


def ocr_text(pdf: str, page: int, tmp: Path) -> str:
    prefix = tmp / f"p{page}"
    subprocess.run(
        ["pdftoppm", "-f", str(page), "-l", str(page), "-r", "300", "-gray", "-png", pdf, str(prefix)],
        check=True,
    )
    img = next(tmp.glob(f"p{page}*.png"))
    out = subprocess.run(["tesseract", str(img), "-", "--psm", "3"], capture_output=True, text=True).stdout
    img.unlink()
    return out


def real_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", text))


def clean(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t­]+", " ", line).strip()
        # drop OCR noise lines (stray symbols, single letters)
        if not line or real_words(line) == 0:
            continue
        lines.append(line)
    text = "\n".join(lines)
    # join lines that were wrapped mid-sentence
    text = re.sub(r"(?<![.:;!?)\]])\n(?=[a-z(])", " ", text)
    return text.strip()


def main(pdf: str, out_path: str):
    pages = page_count(pdf)
    parts = ["# ITU Student Handbook\n"]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for p in range(1, pages + 1):
            text = embedded_text(pdf, p)
            method = "text"
            if real_words(text) < MIN_WORDS:
                text = ocr_text(pdf, p, tmp)
                method = "ocr"
            text = clean(text)
            n = real_words(text)
            print(f"page {p:>2}: {method:<4} {n} words", file=sys.stderr)
            if n < 5:
                continue  # cover / photo-only page
            parts.append(f"\n## Page {p}\n\n{text}\n")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("handbook.pdf", "data/handbook.md")))
