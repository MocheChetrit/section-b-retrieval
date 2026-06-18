"""Preprocessing and chunking (simplified version)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


# Short windows: better factual matching + more precise max-pooling at query time.
# We stay below MiniLM's effective limit (256 wordpieces) once the title is added.
WINDOW_TOKENS = 160
STRIDE_TOKENS = 80
MIN_CHUNK_TOKENS = 20


def _clean_space(text: str) -> str:
    return " ".join(str(text).split())


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    title = _clean_space(record.get("title", ""))
    content = _clean_space(record.get("content", ""))

    # The title is prefixed to EVERY chunk: it anchors the entity for MiniLM
    # (natural text is encoded better than a "Title: ... Content: ..." template).
    prefix = f"{title}. " if title else ""

    words = content.split()
    chunks: List[Chunk] = []

    # Page with no content: the title is enough to represent the page.
    if not words:
        chunks.append(Chunk(page_id=page_id, chunk_id=0, text=title))
        return chunks

    chunk_id = 0
    for start in range(0, len(words), STRIDE_TOKENS):
        part = words[start : start + WINDOW_TOKENS]

        # We always keep the 1st chunk; then we truncate if the window is too short.
        if chunk_id > 0 and len(part) < MIN_CHUNK_TOKENS:
            break

        text = f"{prefix}{' '.join(part)}".strip()
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=text))
        chunk_id += 1

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
