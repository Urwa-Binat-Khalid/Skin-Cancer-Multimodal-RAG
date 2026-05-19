"""
retrieval.py — Phase 6: Hybrid Retrieval System
Covers:
  - BM25 sparse retrieval (rank-bm25)
  - Dense vector retrieval (Qdrant)
  - Knowledge-graph neighbourhood retrieval
  - Query expansion with synonyms
  - Cross-encoder reranking
  - hybrid_retrieve() — fuses all three, deduplicates, reranks
  - retrieve_images() — CLIP-space image similarity search
  - retrieve_for_rag() — top-level call used by llm.py
"""

import re
import numpy as np
import pandas as pd
import torch
from collections import Counter

import networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    DEVICE, SYNONYMS, RERANKER_MODEL,
    TEXT_COLLECTION, IMAGE_COLLECTION,
    VECTOR_TOP_K, BM25_TOP_K, GRAPH_TOP_K,
    RETRIEVAL_TOP_K, IMAGE_TOP_K,
)
from vector_store import qdrant_search, make_label_filter, make_source_filter
from knowledge_graph import graph_context_for_query
from embeddings import embed_query_text, embed_query_image


# ── Module-level state (populated by init_retrieval) ──────────────────────────

_text_model: SentenceTransformer | None = None
_reranker:   CrossEncoder | None        = None
_bm25:       BM25Okapi | None           = None
_df_chunks:  pd.DataFrame | None        = None
_kg:         nx.DiGraph | None          = None
_mm_model                               = None   # CLIP model
_mm_tokenizer                           = None


def init_retrieval(df_chunks:   pd.DataFrame,
                   kg:          nx.DiGraph,
                   mm_model,
                   mm_tokenizer,
                   device: str = DEVICE) -> None:
    """
    Initialise all retrieval components.
    Call once after all data and models are loaded.
    """
    global _text_model, _reranker, _bm25, _df_chunks, _kg
    global _mm_model, _mm_tokenizer

    print("Loading text retrieval model ...")
    _text_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

    print("Loading cross-encoder reranker ...")
    _reranker = CrossEncoder(RERANKER_MODEL, device=device)

    print("Building BM25 index ...")
    _bm25 = BM25Okapi([_tokenize(t) for t in df_chunks["text"].tolist()])

    _df_chunks    = df_chunks
    _kg           = kg
    _mm_model     = mm_model
    _mm_tokenizer = mm_tokenizer
    print("Retrieval system ready.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\b[\w\-\.]+\b", text.lower()) if len(t) > 1]


def _expand_query(query: str) -> str:
    ql    = query.lower()
    extra = [syn for term, syns in SYNONYMS.items()
             if term in ql for syn in syns]
    return query + " " + " ".join(extra) if extra else query


# ── Individual search strategies ───────────────────────────────────────────────

def vector_search(query: str,
                  top_k: int = VECTOR_TOP_K,
                  source_filter: str | None = None) -> list[dict]:
    q_emb   = embed_query_text(query, _text_model)
    qfilter = make_source_filter(source_filter) if source_filter else None
    hits    = qdrant_search(TEXT_COLLECTION, q_emb, top_k, qfilter)
    return [
        {
            "id":          h.id,
            "text":        h.payload.get("text",        ""),
            "score":       float(h.score),
            "section":     h.payload.get("section",     ""),
            "source_type": h.payload.get("source_type", ""),
            "pmid":        h.payload.get("pmid",        ""),
            "pmcid":       h.payload.get("pmcid",       ""),
            "title":       h.payload.get("title",       ""),
            "method":      "vector",
        }
        for h in hits
    ]


def bm25_search(query: str, top_k: int = BM25_TOP_K) -> list[dict]:
    scores  = _bm25.get_scores(_tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if scores[idx] < 0.01:
            continue
        row = _df_chunks.iloc[idx]
        results.append({
            "id":          int(idx),
            "text":        str(row.get("text",        "")),
            "score":       float(scores[idx]),
            "section":     str(row.get("section",     "")),
            "source_type": str(row.get("source_type", "")),
            "pmid":        str(row.get("pmid",        "")),
            "pmcid":       str(row.get("pmcid",       "")),
            "title":       str(row.get("title",       "")),
            "method":      "bm25",
        })
    return results


def graph_search(query: str, top_k: int = GRAPH_TOP_K) -> tuple[list[dict], str]:
    """Returns (results, graph_context_string)."""
    q_lower  = query.lower()
    entities = [n for n in _kg.nodes() if len(n) > 3 and n in q_lower]
    if not entities:
        return [], ""

    ctx_lines = []
    neighbors = set()
    for ent in entities[:3]:
        for _, nbr, data in _kg.out_edges(ent, data=True):
            neighbors.add(nbr)
            ctx_lines.append(f"{ent} --[{data.get('relation', '')}]--> {nbr}")
        for pred, _, data in _kg.in_edges(ent, data=True):
            neighbors.add(pred)
            ctx_lines.append(f"{pred} --[{data.get('relation', '')}]--> {ent}")

    results, seen = [], set()
    for nbr in list(neighbors)[:15]:
        mask = _df_chunks["text"].str.lower().str.contains(
            re.escape(nbr), na=False
        )
        for idx in _df_chunks[mask].index[:2]:
            if idx in seen:
                continue
            seen.add(idx)
            row = _df_chunks.iloc[idx]
            results.append({
                "id":           int(idx),
                "text":         str(row.get("text",        "")),
                "score":        0.7,
                "section":      str(row.get("section",     "")),
                "source_type":  str(row.get("source_type", "")),
                "pmid":         str(row.get("pmid",        "")),
                "pmcid":        str(row.get("pmcid",       "")),
                "title":        str(row.get("title",       "")),
                "method":       "graph",
                "graph_context": "\n".join(ctx_lines[:10]),
            })
            if len(results) >= top_k:
                break

    return results, "\n".join(ctx_lines[:10])


# ── Reranking ──────────────────────────────────────────────────────────────────

def do_rerank(query: str,
              candidates: list[dict],
              top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    if not candidates:
        return []
    pairs  = [(query, c["text"][:512]) for c in candidates]
    scores = _reranker.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ── Hybrid retrieval ───────────────────────────────────────────────────────────

def hybrid_retrieve(query: str,
                    top_k: int = RETRIEVAL_TOP_K,
                    source_filter: str | None = None) -> dict:
    """
    Fuse vector + BM25 + graph results, deduplicate, rerank.

    Returns:
        {
            "results":   list of top-k reranked dicts,
            "graph_ctx": str,
            "stats":     {"vector": N, "bm25": N, "graph": N, "merged": N},
        }
    """
    expanded  = _expand_query(query)
    vec_res   = vector_search(expanded, top_k=VECTOR_TOP_K,
                               source_filter=source_filter)
    bm25_res  = bm25_search(expanded, top_k=BM25_TOP_K)
    graph_res, graph_ctx = graph_search(query, top_k=GRAPH_TOP_K)

    seen, merged = set(), []
    for r in vec_res + bm25_res + graph_res:
        key = r["text"][:80]
        if key not in seen:
            seen.add(key)
            merged.append(r)

    reranked = do_rerank(query, merged, top_k=top_k)
    return {
        "results":   reranked,
        "graph_ctx": graph_ctx,
        "stats": {
            "vector": len(vec_res),
            "bm25":   len(bm25_res),
            "graph":  len(graph_res),
            "merged": len(merged),
        },
    }


# ── Image retrieval ────────────────────────────────────────────────────────────

def retrieve_images(query: str,
                    top_k: int = IMAGE_TOP_K,
                    label_filter: str | None = None) -> list[dict]:
    """
    Retrieve visually similar dermoscopy images using CLIP text encoder.
    Uses the same embedding space as the image collection (512-dim).
    """
    tokens = _mm_tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        q_emb = _mm_model.encode_text(tokens)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
    q_emb = q_emb.cpu().numpy()[0].astype(np.float32)

    qfilter = make_label_filter(label_filter) if label_filter else None
    hits    = qdrant_search(IMAGE_COLLECTION, q_emb, top_k, qfilter)
    return [
        {
            "id":          h.id,
            "score":       float(h.score),
            "label":       h.payload.get("label",                  ""),
            "dataset":     h.payload.get("dataset",                ""),
            "image_path":  h.payload.get("image_path",             ""),
            "description": h.payload.get("image_text_description", ""),
        }
        for h in hits
    ]


# ── Top-level retrieval for RAG ────────────────────────────────────────────────

def retrieve_for_rag(query: str,
                     top_k: int          = RETRIEVAL_TOP_K,
                     include_images: bool = True,
                     image_top_k: int    = IMAGE_TOP_K) -> dict:
    """
    Run hybrid text retrieval + optional image retrieval.
    Returns structured dict consumed by llm.ask().
    """
    hybrid  = hybrid_retrieve(query, top_k=top_k)
    sources = [
        {
            "rank":         i + 1,
            "pmid":         r.get("pmid",         ""),
            "pmcid":        r.get("pmcid",         ""),
            "title":        r.get("title",         ""),
            "section":      r.get("section",       ""),
            "method":       r.get("method",        ""),
            "rerank_score": r.get("rerank_score",  0.0),
            "text":         r.get("text",          ""),
        }
        for i, r in enumerate(hybrid["results"])
    ]

    # Auto-detect disease label for image filter
    label   = None
    q_lower = query.lower()
    for disease in [
        "melanoma", "basal cell carcinoma", "squamous cell carcinoma",
        "nevus", "actinic keratosis", "dermatofibroma",
    ]:
        if disease in q_lower:
            label = disease
            break

    images = (
        retrieve_images(query, top_k=image_top_k, label_filter=label)
        if include_images else []
    )

    return {
        "query":     query,
        "sources":   sources,
        "graph_ctx": hybrid["graph_ctx"],
        "images":    images,
        "stats":     hybrid["stats"],
    }
