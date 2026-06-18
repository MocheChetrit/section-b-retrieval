"""
Section B entry point.

The autograder calls run(queries) once with all queries (batch of 50).
Query embedding + retrieval must fit within the time limit (GPU available).
"""
from __future__ import annotations

from typing import List

from index import build_index
from retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """Returns, for each query, the ordered list of page_ids (most relevant first)."""
    return search_batch(queries)


def build_offline_index() -> None:
    """Run once locally to create artifacts/ (not timed during grading)."""
    build_index()

    # We also pre-build the BM25 cache so that `run()` does not rebuild it
    # during grading (this counts toward the GitHub rubric's "seamless run" criterion).
    from retrieve import _get_lexical_index

    _get_lexical_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index construit dans artifacts/. Lance : python scripts/eval_public.py")
