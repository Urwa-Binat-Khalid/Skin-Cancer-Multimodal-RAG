"""
multimodal.py — Phase 10: Multimodal Image Query
Covers:
  - embed_image_query()  — embed a local image file with BiomedCLIP
  - retrieve_similar_images() — Qdrant image-space similarity search
  - multimodal_ask() — image + optional text → similar cases + research summary

SAFETY: Visual similarity is NOT clinical diagnosis.
        Every result is framed as "similar-image label suggestion".
"""

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from config import DEVICE, IMAGE_COLLECTION, MEDICAL_DISCLAIMER
from vector_store import qdrant_search, make_label_filter
from llm import ask, call_llm_with_retry, RAG_SYSTEM_PROMPT
from retrieval import retrieve_for_rag


# ── Module-level CLIP references (set by pipeline.py after model load) ─────────

_mm_model     = None
_mm_preprocess = None
_mm_tokenizer  = None


def init_multimodal(model, preprocess, tokenizer) -> None:
    """Call once after BiomedCLIP is loaded (in pipeline.py or embeddings.py)."""
    global _mm_model, _mm_preprocess, _mm_tokenizer
    _mm_model      = model
    _mm_preprocess = preprocess
    _mm_tokenizer  = tokenizer
    print("Multimodal module initialised.")


# ── Image embedding ────────────────────────────────────────────────────────────

def embed_image_query(image_path: str,
                      device: str = DEVICE) -> np.ndarray:
    """
    Embed a single image file using BiomedCLIP.
    Returns normalised float32 array (IMAGE_EMBED_DIM,).
    """
    from PIL import Image as PILImage
    img    = PILImage.open(image_path).convert("RGB")
    tensor = _mm_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = _mm_model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype(np.float32)


# ── Similar image search ───────────────────────────────────────────────────────

def retrieve_similar_images(image_path: str,
                             top_k: int = 5,
                             device: str = DEVICE) -> tuple[list[dict], str]:
    """
    Find visually similar dermoscopy images in Qdrant.

    Returns:
        (similar_images, similar_label_suggestion)
        similar_images: list of {"id", "score", "label", "dataset", "image_path"}
        similar_label_suggestion: majority-vote label from top-3 results
    """
    img_emb  = embed_image_query(image_path, device)
    img_hits = qdrant_search(IMAGE_COLLECTION, img_emb, top_k)

    similar_images = [
        {
            "id":         h.id,
            "score":      float(h.score),
            "label":      h.payload.get("label",      ""),
            "dataset":    h.payload.get("dataset",    ""),
            "image_path": h.payload.get("image_path", ""),
        }
        for h in img_hits
    ]

    if similar_images:
        label_counts = Counter(r["label"] for r in similar_images[:3])
        suggestion   = label_counts.most_common(1)[0][0]
    else:
        suggestion = "unknown"

    return similar_images, suggestion


# ── Multimodal prompt builder ──────────────────────────────────────────────────

def _build_mm_prompt(text_query: str,
                     similar_label: str,
                     similar_case_labels: list[str],
                     top_score: float,
                     research_context: str) -> str:
    return f"""
You are a medical research-support assistant for dermatology and skin cancer literature.

{RAG_SYSTEM_PROMPT}

MEDICAL DISCLAIMER:
{MEDICAL_DISCLAIMER}

Image Similarity Result:
- Similar-image label suggestion: {similar_label}
- Top similar case labels: {similar_case_labels}
- Top visual similarity score: {top_score:.3f}

Retrieved Research Context:
{research_context}

User text query:
{text_query}

Write the response in this format:

ANSWER:
[Brief research-support answer grounded only in the retrieved context.]

IMAGE SIMILARITY NOTE:
[Explain that the label is based on retrieved visually similar images, not diagnosis.]

RESEARCH FINDINGS:
[Summarize only findings supported by retrieved sources with citations.]

SAFETY NOTE:
This is not medical advice and not a diagnosis. A dermatologist should evaluate any concerning lesion.

CONFIDENCE:
[HIGH / MEDIUM / LOW based only on retrieved context quality.]

CITATIONS:
[List Source numbers used.]
"""


# ── Top-level multimodal ask ────────────────────────────────────────────────────

def multimodal_ask(image_path: str | None = None,
                   text_query: str        = "",
                   top_k: int             = 5,
                   device: str            = DEVICE) -> dict:
    """
    Accept an image file + optional text query.
    1. Find visually similar dermoscopy cases (image space).
    2. Retrieve related research papers (text space).
    3. Generate a guarded research-support summary via Groq.

    Returns dict with keys:
        image_path, text_query, similar_images, similar_label_suggestion,
        predicted_label (alias), mm_answer, text_answer, sources,
        rag_query, medical_disclaimer
    """
    results = {
        "image_path":        image_path,
        "text_query":        text_query,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }

    # Step 1: Visual similarity
    similar_images, similar_label = [], "unknown"
    if image_path and Path(image_path).exists():
        similar_images, similar_label = retrieve_similar_images(image_path, top_k, device)

    results["similar_images"]            = similar_images
    results["similar_label_suggestion"]  = similar_label
    results["predicted_label"]           = similar_label   # backward-compat alias

    # Step 2: Text retrieval
    rag_query = text_query if text_query else (
        f"{similar_label} dermoscopy visual similarity "
        f"skin lesion research deep learning"
    )
    rag = ask(rag_query, top_k=top_k, include_images=False)
    results["text_answer"] = rag["answer"]
    results["sources"]     = rag["sources"]
    results["rag_query"]   = rag_query

    # Step 3: Multimodal synthesis
    top_score = similar_images[0]["score"] if similar_images else 0.0
    similar_case_labels = [r["label"] for r in similar_images[:3]]

    research_context = "\n".join([
        f"[Source {i + 1}] PMID:{s.get('pmid', '')} | "
        f"PMCID:{s.get('pmcid', '')}\n"
        f"Title: {s.get('title', '')}\n"
        f"Text: {s.get('text', '')[:700]}"
        for i, s in enumerate(rag["sources"][:3])
    ])

    mm_prompt = _build_mm_prompt(
        text_query=text_query,
        similar_label=similar_label,
        similar_case_labels=similar_case_labels,
        top_score=top_score,
        research_context=research_context,
    )
    results["mm_answer"] = call_llm_with_retry(mm_prompt)

    return results
