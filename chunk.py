"""Préprocessing et découpage en chunks (version simplifiée)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


# Fenêtres courtes : meilleur matching factuel + max-pooling plus précis au query-time.
# On reste sous la limite effective de MiniLM (256 wordpieces) une fois le titre ajouté.
WINDOW_TOKENS = 160
STRIDE_TOKENS = 80
MIN_CHUNK_TOKENS = 20


def _clean_space(text: str) -> str:
    return " ".join(str(text).split())


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    title = _clean_space(record.get("title", ""))
    content = _clean_space(record.get("content", ""))

    # Le titre est préfixé sur CHAQUE chunk : il ancre l'entité pour MiniLM
    # (un texte naturel s'encode mieux qu'un template "Title: ... Content: ...").
    prefix = f"{title}. " if title else ""

    words = content.split()
    chunks: List[Chunk] = []

    # Page sans contenu : le titre suffit à représenter la page.
    if not words:
        chunks.append(Chunk(page_id=page_id, chunk_id=0, text=title))
        return chunks

    chunk_id = 0
    for start in range(0, len(words), STRIDE_TOKENS):
        part = words[start : start + WINDOW_TOKENS]

        # On garde toujours le 1er chunk ; on coupe ensuite si la fenêtre est trop courte.
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
