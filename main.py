"""
Section B entry point.

L'autograder appelle run(queries) une fois avec toutes les requêtes (batch de 50).
L'embedding des requêtes + la recherche doivent tenir dans la limite de temps (GPU dispo).
"""
from __future__ import annotations

from typing import List

from index import build_index
from retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """Renvoie, pour chaque requête, la liste ordonnée des page_id (plus pertinent d'abord)."""
    return search_batch(queries)


def build_offline_index() -> None:
    """À lancer une fois en local pour créer artifacts/ (non chronométré au grading)."""
    build_index()

    # On pré-construit aussi le cache BM25 pour que `run()` ne le reconstruise pas
    # au grading (compte pour le critère "seamless run" du barème GitHub).
    from retrieve import _get_lexical_index

    _get_lexical_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index construit dans artifacts/. Lance : python scripts/eval_public.py")
