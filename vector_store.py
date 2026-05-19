"""
vector_store.py — Phase 4: Qdrant Vector Database
Covers:
  - Qdrant client initialisation (on-disk, in-process)
  - Create / reset text and image collections
  - Batch upload of text chunks and image metadata
  - Generic search helper (used by retrieval.py)
"""

import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

from config import (
    QDRANT_PATH, TEXT_COLLECTION, IMAGE_COLLECTION,
    TEXT_EMBED_DIM, IMAGE_EMBED_DIM, UPLOAD_BATCH_SIZE,
)


# ── Client ─────────────────────────────────────────────────────────────────────

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return (or create) the global Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
    gc.collect()


# ── Collections ────────────────────────────────────────────────────────────────

def create_collections(
    text_dim:  int = TEXT_EMBED_DIM,
    image_dim: int = IMAGE_EMBED_DIM,
) -> None:
    """Drop existing collections (if any) and recreate them fresh."""
    client   = get_client()
    existing = [c.name for c in client.get_collections().collections]

    for name, dim in [
        (TEXT_COLLECTION,  text_dim),
        (IMAGE_COLLECTION, image_dim),
    ]:
        if name in existing:
            client.delete_collection(name)
            print(f"Dropped collection: {name}")
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Created collection : {name}  (dim={dim})")


def collection_counts() -> dict[str, int]:
    client = get_client()
    counts = {}
    for name in [TEXT_COLLECTION, IMAGE_COLLECTION]:
        info  = client.get_collection(name)
        count = (
            info.points_count
            if hasattr(info, "points_count")
            else info.vectors_count
        )
        counts[name] = count
    return counts


# ── Upload ──────────────────────────────────────────────────────────────────────

def _upload_points(collection: str,
                   embeddings: np.ndarray,
                   payloads:   list[dict],
                   batch_size: int = UPLOAD_BATCH_SIZE) -> None:
    client = get_client()
    for start in tqdm(range(0, len(payloads), batch_size),
                      desc=f"Uploading {collection}"):
        end    = min(start + batch_size, len(payloads))
        points = [
            PointStruct(
                id      = start + i,
                vector  = embeddings[start + i].tolist(),
                payload = payloads[start + i],
            )
            for i in range(end - start)
        ]
        client.upsert(collection_name=collection, points=points)


def upload_text_chunks(df_chunks: pd.DataFrame,
                       embeddings: np.ndarray) -> None:
    """Upload text chunk vectors + metadata to Qdrant."""
    payloads = [
        {
            "chunk_id":    str(r.get("chunk_id",    "")),
            "pmid":        str(r.get("pmid",        "")),
            "pmcid":       str(r.get("pmcid",       "")),
            "title":       str(r.get("title",       ""))[:500],
            "section":     str(r.get("section",     "")),
            "source_type": str(r.get("source_type", "")),
            "text":        str(r.get("text",        ""))[:2000],
        }
        for _, r in df_chunks.iterrows()
    ]
    _upload_points(TEXT_COLLECTION, embeddings, payloads)
    print(f"Text vectors uploaded: {len(payloads)}")


def upload_image_metadata(df_images: pd.DataFrame,
                          embeddings: np.ndarray) -> None:
    """Upload image vectors + metadata to Qdrant."""
    payloads = [
        {
            "image_id":               str(r.get("image_id",               "")),
            "image_path":             str(r.get("image_path",             "")),
            "label":                  str(r.get("label",                  "")),
            "dataset":                str(r.get("dataset",                "")),
            "split":                  str(r.get("split",                  "")),
            "image_text_description": str(r.get("image_text_description", ""))[:500],
            "embed_source":           str(r.get("embed_source",           "")),
        }
        for _, r in df_images.iterrows()
    ]
    _upload_points(IMAGE_COLLECTION, embeddings, payloads)
    print(f"Image vectors uploaded: {len(payloads)}")


# ── Search ──────────────────────────────────────────────────────────────────────

def qdrant_search(collection: str,
                  vector:     np.ndarray,
                  limit:      int,
                  q_filter:   Filter | None = None) -> list:
    """
    Unified Qdrant search that works with both old and new client versions.
    Returns a list of ScoredPoint objects.
    """
    client = get_client()
    try:
        return client.query_points(
            collection_name=collection,
            query=vector.tolist(),
            query_filter=q_filter,
            limit=limit,
        ).points
    except AttributeError:
        return client.search(
            collection_name=collection,
            query_vector=vector.tolist(),
            query_filter=q_filter,
            limit=limit,
        )


def make_label_filter(label: str) -> Filter:
    return Filter(must=[FieldCondition(key="label", match=MatchValue(value=label))])


def make_source_filter(source_type: str) -> Filter:
    return Filter(must=[FieldCondition(key="source_type",
                                       match=MatchValue(value=source_type))])
