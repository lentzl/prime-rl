#!/usr/bin/env python3
"""Extract pinned, complete public-domain TRAIN chapters without paraphrasing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

BOOKS = (
    (
        35,
        "time-machine",
        "The Time Machine",
        "H. G. Wells",
        "1895",
        "1946",
        "2892e919000e17c83e1dac51b30f4675db50536b644d7579fe8a89bb399a9bdc",
        r"^ [IVX]+\.\n [^\n]+\n",
        tuple(range(1, 5)),
    ),
    (
        120,
        "treasure-island",
        "Treasure Island",
        "Robert Louis Stevenson",
        "1883",
        "1894",
        "5bc08275eacea8640b1aa185c645283bb8c564d7b23df6b03437d229b4fc6ebd",
        r"^[IVX]+\n[^\n]+\n",
        tuple(range(1, 5)),
    ),
    (
        97,
        "flatland",
        "Flatland",
        "Edwin Abbott Abbott",
        "1884",
        "1926",
        "eb84ae1a164f9d54c7bd831c847f99046f6a3199ba78b0bd7c98c09d108a3a9a",
        r"^§ \d+ [^\n]+(?:\n(?!\n)[^\n]+)*\n",
        tuple(range(13, 17)),
    ),
    (
        37423,
        "how-we-think",
        "How We Think",
        "John Dewey",
        "1910",
        "1952",
        "679313717a590a285c10303a3ddd88a990041e415df74c6c678ac12c6c7b8896",
        r"^CHAPTER [A-Z]+\n\n[^\n]+(?:\n(?!\n)[^\n]+)*\n",
        tuple(range(1, 9)),
    ),
)

BOOK_OF_TEA = (
    769, "book-of-tea", "The Book of Tea", "Kakuzo Okakura", "1906", "1913",
    "a7464287f9b431a55ff50d05e48dec31db050eb4a88554425961c6699ea6fcd7",
    r"^[IVX]+\. [^\n]+\n", (1, 2),
)


def prepare(raw_dir: Path, output_dir: Path, *, additional_chapters_per_book: int = 0,
            include_book_of_tea: bool = False, additional_book_of_tea_chapters: int = 0) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if additional_chapters_per_book < 0 or additional_book_of_tea_chapters < 0:
        raise ValueError("additional chapter count must be nonnegative")
    if additional_book_of_tea_chapters and not include_book_of_tea:
        raise ValueError("additional Book of Tea chapters require including that book")
    chapters, books = [], []
    selected_books = (*BOOKS, BOOK_OF_TEA) if include_book_of_tea else BOOKS
    for ebook, slug, title, author, year, died, expected_hash, pattern, selected in selected_books:
        raw = (raw_dir / f"pg{ebook}.txt").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_hash:
            raise ValueError(f"source changed: {slug}: {digest}")
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
        boundaries = list(re.finditer(pattern, text, re.MULTILINE))
        expected_count = {35: 16, 120: 34, 97: 22, 37423: 16, 769: 7}[ebook]
        if len(boundaries) != expected_count:
            raise ValueError(f"unexpected chapter boundaries: {slug}: {len(boundaries)}")
        additional = additional_book_of_tea_chapters if ebook == 769 else additional_chapters_per_book
        selected = (*selected, *range(max(selected) + 1, max(selected) + 1 + additional))
        if max(selected) >= len(boundaries):
            raise ValueError(f"selected chapter lacks a following boundary: {slug}")
        for number in selected:
            start, end = boundaries[number - 1], boundaries[number]
            body = text[start.end() : end.start()].strip()
            body = re.sub(r"\n+PART [A-Z]+(?:: |--)[^\n]+\s*$", "", body)
            paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body) if p.strip()]
            source = "\n\n".join(paragraphs) + "\n"
            if not 400 <= len(source.split()) <= 6500 or "*** END OF" in source:
                raise ValueError(f"unexpected chapter extent: {slug} {number}")
            chapters.append(
                {
                    "slug": f"{slug}-ch{number:02}",
                    "book": slug,
                    "chapter": number,
                    "heading": " ".join(start.group().split()),
                    "next_heading": " ".join(end.group().split()),
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "source_words": len(source.split()),
                    "paragraphs": len(paragraphs),
                    "source": source,
                }
            )
        books.append(
            {
                "slug": slug,
                "title": title,
                "author": author,
                "publication_year": year,
                "author_death_year": died,
                "ebook": ebook,
                "url": f"https://www.gutenberg.org/ebooks/{ebook}",
                "download_url": f"https://www.gutenberg.org/cache/epub/{ebook}/pg{ebook}.txt",
                "raw_sha256": digest,
                "chapters": selected,
                "copyright_catalog": "Public domain in the USA",
            }
        )
    output_dir.mkdir(parents=True)
    for row in chapters:
        (output_dir / f"{row['slug']}.md").write_text(row.pop("source"), encoding="utf-8")
    manifest = {
        "schema_version": "document-summary-public-train-sources/v1",
        "retrieved": "2026-09-08",
        "split": "TRAIN",
        "summary_supervision": "not included; requires separately authored and reviewed labels",
        "normalization": "CRLF to LF; chapter headings omitted; paragraph-internal whitespace unwrapped; inter-part heading omitted; prose and sidenotes retained",
        "source_license": "https://www.gutenberg.org/policy/license.html",
        "raw_files_include_full_license": True,
        "excluded_eval_books": [11, 2274, 37134],
        "pretraining_contamination_possible": True,
        "books": books,
        "chapters": chapters,
    }
    (output_dir / "SOURCES.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--additional-chapters-per-book", type=int, default=0)
    parser.add_argument("--include-book-of-tea", action="store_true", help="Add complete TRAIN chapters I and II.")
    parser.add_argument("--additional-book-of-tea-chapters", type=int, default=0,
                        help="Extend Book of Tea beyond chapter II without changing the other books' selection.")
    args = parser.parse_args()
    result = prepare(args.raw_dir, args.output_dir, additional_chapters_per_book=args.additional_chapters_per_book,
                     include_book_of_tea=args.include_book_of_tea,
                     additional_book_of_tea_chapters=args.additional_book_of_tea_chapters)
    print(
        json.dumps({"chapters": len(result["chapters"]), "words": sum(c["source_words"] for c in result["chapters"])})
    )
