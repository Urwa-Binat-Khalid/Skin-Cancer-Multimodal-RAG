"""
config.py — Central configuration for DermIQ: Multimodal Medical RAG
All constants, paths, model names, API keys, and hyperparameters live here.
"""

import os
from pathlib import Path

# ── API Keys ───────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
ENTREZ_EMAIL  = os.getenv("ENTREZ_EMAIL", "your_email@gmail.com")

# ── Device ─────────────────────────────────────────────────────────────────────
# Auto-detected at runtime in embeddings.py; override here if needed
DEVICE = "cuda"   # or "cpu"

# ── Base Directories ───────────────────────────────────────────────────────────
BASE_DIR = Path("data")

DIRS = {
    "pubmed":        BASE_DIR / "pubmed",
    "pmc_fulltext":  BASE_DIR / "pmc_fulltext",
    "images":        BASE_DIR / "images" / "metadata",
    "chunks":        BASE_DIR / "chunks",
    "embeddings":    BASE_DIR / "embeddings",
    "qdrant":        BASE_DIR / "qdrant_storage",
    "kg":            BASE_DIR / "knowledge_graph",
    "evaluation":    BASE_DIR / "evaluation",
    "deployment":    BASE_DIR / "deployment",
    "logs":          BASE_DIR / "logs",
}

# ── Dataset Paths (Kaggle) ─────────────────────────────────────────────────────
HAM_DIR  = Path("/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000")
ISIC_DIR = Path(
    "/kaggle/input/datasets/nodoubttome/skin-cancer9-classesisic/"
    "Skin cancer ISIC The International Skin Imaging Collaboration"
)

# ── PubMed Search ──────────────────────────────────────────────────────────────
PUBMED_MAX_RESULTS   = 500
FULLTEXT_MAX_PAPERS  = 150

SKIN_CANCER_QUERY = """
(
    melanoma OR "skin cancer" OR "basal cell carcinoma"
    OR "squamous cell carcinoma" OR "actinic keratosis"
    OR "melanocytic nevus" OR "skin lesion"
)
AND (
    dermoscopy OR dermatoscopy OR "skin lesion classification"
    OR diagnosis OR segmentation OR classification
)
AND (
    "deep learning" OR CNN OR transformer OR "vision transformer"
    OR "machine learning" OR "artificial intelligence"
)
"""

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_MAX_WORDS   = 250
CHUNK_OVERLAP     = 40
ABSTRACT_MAX_WORDS = 200
MIN_CHUNK_CHARS   = 50

# ── Embedding Models ───────────────────────────────────────────────────────────
TEXT_MODEL_NAME  = "BAAI/bge-large-en-v1.5"
IMAGE_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
IMAGE_FALLBACK_MODEL = "ViT-B-32"
TEXT_EMBED_DIM   = 1024   # BGE-large output dim
IMAGE_EMBED_DIM  = 512    # BiomedCLIP output dim
EMBED_BATCH_SIZE = 64
IMG_BATCH_SIZE   = 32

# ── Qdrant Vector DB ───────────────────────────────────────────────────────────
QDRANT_PATH       = str(DIRS["qdrant"])
TEXT_COLLECTION   = "skin_cancer_text"
IMAGE_COLLECTION  = "skin_cancer_images"
UPLOAD_BATCH_SIZE = 256

# ── Reranker ───────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM (Groq) ─────────────────────────────────────────────────────────────────
LLM_MODEL      = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 1000
LLM_TEMPERATURE = 0.1
LLM_MAX_RETRIES = 3
LLM_RETRY_WAIT  = 10   # seconds

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K  = 5
VECTOR_TOP_K     = 20
BM25_TOP_K       = 20
GRAPH_TOP_K      = 10
IMAGE_TOP_K      = 3

SYNONYMS = {
    "melanoma":             ["cutaneous melanoma", "malignant melanoma"],
    "basal cell carcinoma": ["bcc", "basal cell cancer"],
    "dermoscopy":           ["dermatoscopy", "dermoscope"],
    "cnn":                  ["convolutional neural network"],
    "vit":                  ["vision transformer"],
    "efficientnet":         ["efficientnet-b4", "efficientnetb4"],
    "ham10000":             ["ham 10000"],
    "auc":                  ["area under curve", "auroc"],
}

# ── Knowledge Graph ─────────────────────────────────────────────────────────────
KG_SAVE_PATH = DIRS["kg"] / "skin_cancer_kg.graphml"

ENTITY_LEXICON = {
    "DISEASE_SKIN": [
        "melanoma", "basal cell carcinoma", "squamous cell carcinoma",
        "actinic keratosis", "nevus", "skin cancer", "skin lesion",
        "melanocytic nevus", "dermatofibroma", "bcc", "scc",
        "benign keratosis", "vascular lesion",
    ],
    "TECHNIQUE": [
        "dermoscopy", "dermatoscopy", "biopsy", "whole slide imaging",
        "reflectance confocal microscopy",
    ],
    "MODEL": [
        "cnn", "convolutional neural network", "resnet", "vgg", "inception",
        "efficientnet", "densenet", "vision transformer", "vit", "swin",
        "unet", "u-net", "transformer", "yolo", "deep learning",
        "machine learning", "artificial intelligence", "random forest", "svm",
    ],
    "METRIC": [
        "accuracy", "sensitivity", "specificity", "auc", "roc",
        "f1", "f1 score", "precision", "recall", "iou", "dice coefficient",
    ],
    "DATASET": ["ham10000", "isic", "isic 2018", "isic 2019", "ph2", "derm7pt"],
    "BODY_SITE": ["skin", "dermis", "epidermis", "melanocyte", "keratinocyte"],
}

RELATION_PATTERNS = [
    (
        r"([\w\s\-]+?)\s+(?:detects?|classif(?:ies|y)|diagnos(?:es|e)|segments?)"
        r"\s+([\w\s\-]+)", "DETECTS"
    ),
    (
        r"([\w\s\-]+?)\s+(?:achieved?|obtained?|reached?)\s+(?:an?\s+)?"
        r"(accuracy|auc|sensitivity|specificity|f1)\s+of\s+([\d\.]+\s*%?)",
        "ACHIEVES"
    ),
    (
        r"([\w\s\-]+?)\s+(?:trained?|evaluated?|tested?)\s+(?:on|using)"
        r"\s+(?:the\s+)?(ham10000|isic[\s\w]*|ph2)", "TRAINED_ON"
    ),
    (
        r"([\w\s\-]+?)\s+(?:improves?|outperforms?|enhances?)\s+([\w\s\-]+)",
        "IMPROVES"
    ),
]

# ── HAM10000 label map ─────────────────────────────────────────────────────────
HAM_LABEL_MAP = {
    "akiec": "actinic keratosis",
    "bcc":   "basal cell carcinoma",
    "bkl":   "benign keratosis",
    "df":    "dermatofibroma",
    "mel":   "melanoma",
    "nv":    "nevus",
    "vasc":  "vascular lesion",
}

# ── Medical Disclaimer ─────────────────────────────────────────────────────────
MEDICAL_DISCLAIMER = (
    "This system is for research and educational support only. "
    "It is NOT a clinical diagnostic tool and must not replace "
    "a qualified dermatologist or medical professional."
)
