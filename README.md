# Section B — Wikipedia Page Retrieval

Hybrid retrieval pipeline (dense + lexical) over a Wikipedia-style corpus.
For each query, `run(queries)` returns a ranked list of `page_id`, evaluated with NDCG@10.

**Public NDCG@10: 0.4458** — query phase ~7 s (well under the 60 s limit).

**Video presentation:** https://drive.google.com/file/d/1MdgtaVFzcAzIdQ1xSU9MdVr0t0RO7l9w/view?usp=sharing

Authors:  Moche Chetrit (<340872084>), Naomi Chauvart (<337917843>)

---

## Method (short)

- **Chunking** — overlapping windows of 160 tokens, stride 80; the page title is
  prepended to every chunk; very short chunks are dropped.
- **Embeddings** — `all-MiniLM-L6-v2`, **fine-tuned** on the public query–page pairs
  with a contrastive in-batch-negatives objective. Vectors are L2-normalized.
- **Dense retrieval** — query/chunk dot product, aggregated to page level by
  **mean-pooling** (mean-pooling beat max-pooling, which favored long pages).
- **Lexical retrieval** — chunk-level BM25, aggregated per page by max.
- **Fusion** — Reciprocal Rank Fusion of the two rankings (dense weight 3.0,
  BM25 weight 2.0).

| Variant | NDCG@10 |
|---|---|
| Dense only (base, max-pool) | 0.165 |
| BM25 only | 0.230 |
| Hybrid RRF (base encoder) | 0.218 |
| + mean-pooling | 0.271 |
| + fine-tuned encoder (final) | **0.4458** |

The fine-tuning gain was validated on held-out queries (35 train / 15 validation)
before training the final model on all 50, to confirm it generalizes (not memorization).

---

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions needed (CUDA-12.1 torch compatible with
the VM GPU, `sentence-transformers==3.1.1`, `numpy<2`, `transformers==4.44.2`).

---

## Run the evaluation

The artifacts are already committed (see below), so no rebuild is needed:

```bash
python scripts/eval_public.py
```

Prints mean NDCG@10 on the 50 public queries.

---

## Artifacts (`artifacts/`, committed via Git LFS)

| File | Description |
|---|---|
| `index_vectors.npy` | Chunk embeddings, stored as float16 for fast loading |
| `index_meta.json` | Chunk-to-page metadata |
| `bm25_index.npz` | Precomputed chunk-level BM25 index (compressed) |
| `bm25_vocab.pkl` | BM25 vocabulary (token to id) |
| `finetuned_minilm/` | The fine-tuned MiniLM checkpoint used at query time |

`run()` loads these directly and does **not** rebuild the index at grading time.

---

## Rebuilding the index (optional, offline only)

To rebuild from scratch on your own machine:

```bash
python scripts/build_index.py     # embeds the corpus into artifacts/
```

The fine-tuned model is loaded automatically by `embed.py` when
`artifacts/finetuned_minilm/` is present.

---

## File overview

- `main.py` — entry point, exposes `run(queries)`
- `chunk.py` — corpus chunking
- `embed.py` — embedding (loads the fine-tuned model if present)
- `index.py` — offline index build / load
- `retrieve.py` — dense + BM25 retrieval and RRF fusion
- `utils.py` — shared paths and helpers
- `eval.py`, `scripts/` — provided evaluation utilities (unchanged)
