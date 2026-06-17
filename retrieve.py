"""Recuperation au query-time : dense mean-pool + BM25 chunk-level vectorise, fusion RRF.

Optimisations de taille (pour un chargement disque rapide a froid) :
  - vecteurs stockes en float16, reconvertis en float32 au chargement ;
  - index BM25 stocke compresse (savez_compressed), poids/idf en float16.
Le calcul reste mathematiquement identique -> score inchange.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from chunk import chunk_corpus
from embed import embed_queries
from index import load_index
from utils import ARTIFACTS_DIR, K_EVAL, iter_entries


_WORD_RE = re.compile(r"[a-z0-9]+")
_LEX_CACHE = None
_BM25_NPZ = "bm25_index.npz"
_BM25_VOCAB = "bm25_vocab.pkl"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
    "by", "from", "at", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "which", "who", "what", "when", "where", "about",
    "into", "during", "including", "between", "among", "its", "it",
    "did", "does", "do", "has", "have", "had",
}

N_POOL = 200
K_RRF = 60
W_DENSE = 3.0
W_BM25 = 2.0
BM25_K1 = 1.5
BM25_B = 0.75


def _tokens(text: str) -> List[str]:
    raw = _WORD_RE.findall(text.lower())
    out: List[str] = []
    for t in raw:
        if len(t) < 2 or t in STOPWORDS:
            continue
        out.append(t)
        if len(t) > 4 and t.endswith("s"):
            out.append(t[:-1])
        if len(t) == 4 and t.isdigit():
            year = int(t)
            if 1000 <= year <= 2099:
                out.append(f"{(year // 10) * 10}s")
    return out


def _build_lexical_index():
    records = list(iter_entries())
    chunks = chunk_corpus(records)
    chunk_tokens = [_tokens(c.text) for c in chunks]
    n_chunks = len(chunk_tokens)

    dl_arr = np.zeros(n_chunks, dtype=np.float32)
    vocab = {}
    postings = defaultdict(list)
    for ci, toks in enumerate(chunk_tokens):
        c = Counter(toks)
        dl_arr[ci] = sum(c.values()) or 1
        for tok, tf in c.items():
            tid = vocab.get(tok)
            if tid is None:
                tid = len(vocab); vocab[tok] = tid
            postings[tid].append((ci, tf))

    n_terms = len(vocab)
    avgdl = float(dl_arr.sum() / max(n_chunks, 1))

    term_off = np.zeros(n_terms + 1, dtype=np.int64)
    pc_list = []; tf_list = []
    for tid in range(n_terms):
        for ci, tf in postings[tid]:
            pc_list.append(ci); tf_list.append(tf)
        term_off[tid + 1] = len(pc_list)

    post_chunk = np.asarray(pc_list, dtype=np.int32)
    post_tf = np.asarray(tf_list, dtype=np.float32)
    dl_post = dl_arr[post_chunk]
    post_w = (post_tf * (BM25_K1 + 1.0) /
              (post_tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl_post / avgdl))).astype(np.float16)

    df = np.diff(term_off).astype(np.float32)
    idf = np.log(1.0 + (n_chunks - df + 0.5) / (df + 0.5)).astype(np.float16)

    out_dir = ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / _BM25_NPZ,
                        post_chunk=post_chunk, post_w=post_w, term_off=term_off,
                        idf=idf, n_chunks=np.int64(n_chunks))
    with (out_dir / _BM25_VOCAB).open("wb") as f:
        pickle.dump(vocab, f)

    return {"post_chunk": post_chunk, "post_w": post_w.astype(np.float32),
            "term_off": term_off, "idf": idf.astype(np.float32),
            "n_chunks": n_chunks, "vocab": vocab}


def _get_lexical_index():
    global _LEX_CACHE
    if _LEX_CACHE is not None:
        return _LEX_CACHE

    npz_path = ARTIFACTS_DIR / _BM25_NPZ
    vocab_path = ARTIFACTS_DIR / _BM25_VOCAB
    if npz_path.exists() and vocab_path.exists():
        data = np.load(npz_path)
        with vocab_path.open("rb") as f:
            vocab = pickle.load(f)
        _LEX_CACHE = {
            "post_chunk": data["post_chunk"],
            "post_w": data["post_w"].astype(np.float32),
            "term_off": data["term_off"],
            "idf": data["idf"].astype(np.float32),
            "n_chunks": int(data["n_chunks"]),
            "vocab": vocab,
        }
        return _LEX_CACHE

    _LEX_CACHE = _build_lexical_index()
    return _LEX_CACHE


def _bm25_chunk_scores(query: str, lex, n_chunks: int) -> np.ndarray:
    vocab = lex["vocab"]; off = lex["term_off"]
    pc = lex["post_chunk"]; pw = lex["post_w"]; idf = lex["idf"]

    parts_c = []; parts_v = []
    for tok in set(_tokens(query)):
        tid = vocab.get(tok)
        if tid is None:
            continue
        s, e = int(off[tid]), int(off[tid + 1])
        if e > s:
            parts_c.append(pc[s:e])
            parts_v.append(pw[s:e] * idf[tid])

    if not parts_c:
        return np.zeros(n_chunks, dtype=np.float32)

    allc = np.concatenate(parts_c)
    allv = np.concatenate(parts_v)
    cs = np.bincount(allc, weights=allv, minlength=n_chunks)[:n_chunks]
    return cs.astype(np.float32)


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(len(scores))
    return r


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    corpus_vectors, chunk_page_ids = load_index(artifacts_dir)
    corpus_vectors = np.asarray(corpus_vectors, dtype=np.float32)  # float16 -> float32
    query_vectors = embed_queries(queries, batch_size=64)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    lex = _get_lexical_index()

    unique_pages = list(dict.fromkeys(chunk_page_ids))
    page_pos = {pid: i for i, pid in enumerate(unique_pages)}
    num_pages = len(unique_pages)
    n_chunks = len(chunk_page_ids)
    chunk_pos = np.asarray([page_pos[p] for p in chunk_page_ids], dtype=np.int64)

    contiguous = bool(np.all(np.diff(chunk_pos) >= 0))
    if contiguous:
        starts = np.concatenate(([0], np.flatnonzero(np.diff(chunk_pos)) + 1))
        counts = np.diff(np.concatenate((starts, [n_chunks]))).astype(np.float32)
    else:
        counts = np.zeros(num_pages, dtype=np.float32)
        np.add.at(counts, chunk_pos, 1.0)
        counts = np.maximum(counts, 1.0)

    dense_matrix = query_vectors @ corpus_vectors.T

    if contiguous:
        page_dense_all = np.add.reduceat(dense_matrix, starts, axis=1) / counts[None, :]
    else:
        page_dense_all = np.zeros((len(queries), num_pages), dtype=np.float32)
        for q_idx in range(len(queries)):
            pd = np.zeros(num_pages, dtype=np.float32)
            np.add.at(pd, chunk_pos, dense_matrix[q_idx])
            page_dense_all[q_idx] = pd / counts

    results: List[List[int]] = []
    for q_idx, query in enumerate(queries):
        page_dense = page_dense_all[q_idx]
        cs = _bm25_chunk_scores(query, lex, n_chunks)
        if contiguous:
            page_bm25 = np.maximum.reduceat(cs, starts)
        else:
            page_bm25 = np.zeros(num_pages, dtype=np.float32)
            np.maximum.at(page_bm25, chunk_pos, cs)

        rd = _ranks(page_dense)
        rb = _ranks(page_bm25)
        rrf = (
            np.where(rd < N_POOL, W_DENSE / (K_RRF + rd), 0.0)
            + np.where(rb < N_POOL, W_BM25 / (K_RRF + rb), 0.0)
        )
        top = np.argsort(-rrf, kind="stable")[:top_k]
        results.append([int(unique_pages[i]) for i in top])

    return results