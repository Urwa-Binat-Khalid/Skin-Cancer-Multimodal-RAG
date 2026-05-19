"""
embeddings.py — Phase 3: Text & Image Embeddings
Covers:
  - Text embedding with BAAI/bge-large-en-v1.5 (1024-dim, normalized)
  - Image embedding with BiomedCLIP (512-dim); text-description fallback
  - Save/load helpers for .npy files
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import open_clip

from config import (
    DEVICE, TEXT_MODEL_NAME, IMAGE_MODEL_NAME, IMAGE_FALLBACK_MODEL,
    TEXT_EMBED_DIM, IMAGE_EMBED_DIM,
    EMBED_BATCH_SIZE, IMG_BATCH_SIZE, DIRS,
)


# ── Resolve device ─────────────────────────────────────────────────────────────

def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    return device


# ── Text Embeddings ────────────────────────────────────────────────────────────

def load_text_model(device: str = DEVICE) -> SentenceTransformer:
    print(f"Loading {TEXT_MODEL_NAME} ...")
    return SentenceTransformer(TEXT_MODEL_NAME, device=device)


def embed_texts(texts: list[str],
                model: SentenceTransformer,
                batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """
    Encode a list of strings with BGE-large.
    Returns float32 array of shape (N, TEXT_EMBED_DIM), L2-normalised.
    """
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Text embeddings: {embeddings.shape} | "
          f"norm range {norms.min():.4f}–{norms.max():.4f}")
    return embeddings


def save_text_embeddings(embeddings: np.ndarray) -> Path:
    path = DIRS["embeddings"] / "text_embeddings.npy"
    np.save(path, embeddings)
    print(f"Saved text embeddings → {path}")
    return path


def load_text_embeddings() -> np.ndarray:
    path = DIRS["embeddings"] / "text_embeddings.npy"
    arr = np.load(path)
    print(f"Loaded text embeddings: {arr.shape}")
    return arr


# ── Image Embeddings ───────────────────────────────────────────────────────────

def load_image_model(device: str = DEVICE):
    """
    Load BiomedCLIP; fall back to CLIP ViT-B/32 if unavailable.
    Returns (model, preprocess, tokenizer, model_name_str).
    """
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(IMAGE_MODEL_NAME)
        tokenizer = open_clip.get_tokenizer(IMAGE_MODEL_NAME)
        name = "BiomedCLIP"
        print("BiomedCLIP loaded.")
    except Exception as e:
        print(f"BiomedCLIP unavailable ({e}) — falling back to CLIP ViT-B/32")
        model, _, preprocess = open_clip.create_model_and_transforms(
            IMAGE_FALLBACK_MODEL, pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer(IMAGE_FALLBACK_MODEL)
        name = "CLIP-ViT-B/32"

    model = model.to(device)
    model.eval()
    print(f"Image model: {name}, dim={_get_image_dim(model, device)}")
    return model, preprocess, tokenizer, name


def _get_image_dim(model, device: str) -> int:
    if hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
        return model.visual.output_dim
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        return model.encode_image(dummy).shape[-1]


def embed_images(df,
                 model,
                 preprocess,
                 tokenizer,
                 device: str = DEVICE,
                 img_batch:      int = IMG_BATCH_SIZE,
                 fallback_batch: int = EMBED_BATCH_SIZE,
                 ) -> tuple[np.ndarray, list[str]]:
    """
    Embed dermoscopy images with BiomedCLIP.
    Falls back to text-description embedding for unreadable images.

    Returns:
        embeddings: float32 array (N, IMAGE_EMBED_DIM)
        sources:    list of 'image' or 'text_fallback' per row
    """
    all_embs       = []
    img_batch_data = []
    img_idx        = []

    def flush():
        if not img_batch_data:
            return
        tensor = torch.stack(img_batch_data).to(device)
        with torch.no_grad():
            embs = model.encode_image(tensor)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        for idx, emb in zip(img_idx, embs.cpu().numpy()):
            all_embs.append((idx, emb.astype(np.float32), "image"))
        img_batch_data.clear()
        img_idx.clear()
        if device == "cuda":
            torch.cuda.empty_cache()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Embedding images"):
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img_batch_data.append(preprocess(img))
            img_idx.append(i)
            if len(img_batch_data) >= img_batch:
                flush()
        except Exception:
            all_embs.append((i, None, "text_fallback"))
    flush()

    # Text-description fallback for failed images
    fallback_idx = [i for i, e, _ in all_embs if e is None]
    if fallback_idx:
        print(f"  Text fallback for {len(fallback_idx)} images ...")
        texts = df.loc[fallback_idx, "image_text_description"].tolist()
        f_embs = []
        for b in tqdm(range(0, len(texts), fallback_batch), desc="  Fallback"):
            tokens = tokenizer(texts[b:b + fallback_batch]).to(device)
            with torch.no_grad():
                e = model.encode_text(tokens)
                e = e / e.norm(dim=-1, keepdim=True)
            f_embs.append(e.cpu().numpy().astype(np.float32))
            if device == "cuda":
                torch.cuda.empty_cache()
        f_embs_np = np.vstack(f_embs)
        f_map = dict(zip(fallback_idx, f_embs_np))
        all_embs = [
            (i, f_map[i] if e is None else e, s)
            for i, e, s in all_embs
        ]

    all_embs.sort(key=lambda x: x[0])
    embeddings = np.vstack([e for _, e, _ in all_embs])
    sources    = [s for _, _, s in all_embs]
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Image embeddings: {embeddings.shape} | "
          f"norm range {norms.min():.4f}–{norms.max():.4f}")
    return embeddings, sources


def save_image_embeddings(embeddings: np.ndarray) -> Path:
    path = DIRS["embeddings"] / "image_embeddings.npy"
    np.save(path, embeddings)
    print(f"Saved image embeddings → {path}")
    return path


def load_image_embeddings() -> np.ndarray:
    path = DIRS["embeddings"] / "image_embeddings.npy"
    arr = np.load(path)
    print(f"Loaded image embeddings: {arr.shape}")
    return arr


# ── Query embedding (used by retrieval.py) ─────────────────────────────────────

def embed_query_text(query: str,
                     text_model: SentenceTransformer,
                     device: str = DEVICE) -> np.ndarray:
    """Embed a single query string for dense retrieval."""
    with torch.no_grad():
        return text_model.encode(
            [f"Represent this sentence for retrieval: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)


def embed_query_image(image_path: str,
                      model,
                      preprocess,
                      device: str = DEVICE) -> np.ndarray:
    """Embed a query image file for visual similarity search."""
    img    = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype(np.float32)
