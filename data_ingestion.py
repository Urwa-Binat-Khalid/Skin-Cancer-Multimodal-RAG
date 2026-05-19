"""
data_ingestion.py — Phase 1: Data Collection
Covers:
  - PubMed search and abstract fetching (Biopython Entrez)
  - PMC full-text XML download (EuropePMC / BioC / OAI strategies)
  - HAM10000 + ISIC image manifest building
"""

import re
import time
import requests
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from Bio import Entrez, Medline
from tqdm import tqdm

from config import (
    ENTREZ_EMAIL, SKIN_CANCER_QUERY, PUBMED_MAX_RESULTS,
    FULLTEXT_MAX_PAPERS, DIRS, HAM_DIR, ISIC_DIR, HAM_LABEL_MAP,
    CHUNK_MAX_WORDS, CHUNK_OVERLAP,
)


# ── PubMed Search ──────────────────────────────────────────────────────────────

def search_pubmed(query: str = SKIN_CANCER_QUERY,
                  max_results: int = PUBMED_MAX_RESULTS) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    Entrez.email = ENTREZ_EMAIL
    all_ids = []
    for retstart in range(0, max_results, 100):
        try:
            handle = Entrez.esearch(
                db="pubmed", term=query,
                retmax=100, retstart=retstart, sort="relevance"
            )
            record = Entrez.read(handle)
            ids = record["IdList"]
            all_ids.extend(ids)
            time.sleep(0.4)
            if len(ids) < 100:
                break
        except Exception as e:
            print(f"PubMed search error at offset {retstart}: {e}")
            time.sleep(2)
    return all_ids[:max_results]


def fetch_abstracts(paper_ids: list[str], batch_size: int = 50) -> pd.DataFrame:
    """Fetch Medline records for a list of PMIDs; returns DataFrame."""
    Entrez.email = ENTREZ_EMAIL
    records_list = []
    for i in tqdm(range(0, len(paper_ids), batch_size), desc="Fetching abstracts"):
        batch = paper_ids[i:i + batch_size]
        try:
            handle = Entrez.efetch(
                db="pubmed", id=",".join(batch),
                rettype="medline", retmode="text"
            )
            for r in Medline.parse(handle):
                records_list.append({
                    "pmid":             r.get("PMID", ""),
                    "title":            r.get("TI",   ""),
                    "abstract":         r.get("AB",   ""),
                    "journal":          r.get("JT",   ""),
                    "publication_date": r.get("DP",   ""),
                    "authors":          ", ".join(r.get("AU", [])[:10]),
                    "mesh_terms":       "; ".join(r.get("MH", [])),
                    "keywords":         "; ".join(
                        r.get("OT", []) if isinstance(r.get("OT", []), list)
                        else [str(r.get("OT", ""))]
                    ),
                    "source": "PubMed",
                })
            time.sleep(0.4)
        except Exception as e:
            print(f"Abstract fetch error at batch {i}: {e}")
            time.sleep(2)
    return pd.DataFrame(records_list)


# ── PMC Full-Text ──────────────────────────────────────────────────────────────

def get_pmc_ids(paper_ids: list[str], batch_size: int = 150) -> pd.DataFrame:
    """Convert PMIDs to PMCIDs via NCBI ID converter API."""
    BASE_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
    pmc_records = []
    for i in tqdm(range(0, len(paper_ids), batch_size), desc="PMC ID lookup"):
        batch = paper_ids[i:i + batch_size]
        for attempt in range(3):
            try:
                r = requests.get(BASE_URL, params={
                    "ids": ",".join(batch), "idtype": "pmid",
                    "format": "xml", "tool": "dermiq_rag",
                    "email": ENTREZ_EMAIL,
                }, timeout=30)
                if r.status_code != 200:
                    raise ValueError(f"HTTP {r.status_code}")
                root = ET.fromstring(r.content)
                for rec in root.findall(".//record"):
                    pmid  = rec.get("pmid",  "")
                    pmcid = rec.get("pmcid", "")
                    if pmcid:
                        pmc_records.append({
                            "pmid":  pmid,
                            "pmcid": pmcid,
                            "doi":   rec.get("doi", ""),
                        })
                time.sleep(0.5)
                break
            except Exception:
                time.sleep(2)
    if not pmc_records:
        return pd.DataFrame(columns=["pmid", "pmcid", "doi"])
    return pd.DataFrame(pmc_records)


def download_fulltext(pmcid: str, save_dir: Path) -> tuple[str | None, str]:
    """
    Try three strategies to download full-text XML for a given PMCID.
    Returns (file_path, strategy_used) or (None, 'failed').
    """
    # Strategy 1: Europe PMC
    try:
        r = requests.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
            timeout=30, headers={"Accept": "application/xml"}
        )
        if (r.status_code == 200 and len(r.text) > 5000
                and ("<?xml" in r.text[:100] or "<!DOCTYPE article" in r.text[:100])):
            path = save_dir / f"{pmcid}_europepmc.xml"
            path.write_text(r.text, encoding="utf-8")
            return str(path), "europepmc"
    except Exception:
        pass

    # Strategy 2: BioC API
    try:
        numeric = pmcid.replace("PMC", "")
        r = requests.get(
            f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
            f"pmcoa.cgi/BioC_xml/{numeric}/unicode", timeout=30
        )
        if r.status_code == 200 and "<collection>" in r.text and len(r.text) > 3000:
            path = save_dir / f"{pmcid}_bioc.xml"
            path.write_text(r.text, encoding="utf-8")
            return str(path), "bioc"
    except Exception:
        pass

    # Strategy 3: PMC OAI
    try:
        r = requests.get(
            f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
            f"?verb=GetRecord"
            f"&identifier=oai:pubmedcentral.nih.gov:{pmcid.replace('PMC', '')}"
            f"&metadataPrefix=pmc",
            timeout=30,
        )
        if r.status_code == 200 and "<article" in r.text and len(r.text) > 5000:
            path = save_dir / f"{pmcid}_oai.xml"
            path.write_text(r.text, encoding="utf-8")
            return str(path), "oai"
    except Exception:
        pass

    return None, "failed"


def download_all_fulltexts(df_pmc: pd.DataFrame,
                            max_papers: int = FULLTEXT_MAX_PAPERS
                            ) -> pd.DataFrame:
    """Batch-download full texts; returns manifest DataFrame."""
    downloaded, failed = [], []
    for _, row in tqdm(df_pmc.head(max_papers).iterrows(),
                       total=max_papers, desc="Full text download"):
        path, method = download_fulltext(row["pmcid"], DIRS["pmc_fulltext"])
        if path:
            downloaded.append({
                "pmid":   row["pmid"],
                "pmcid":  row["pmcid"],
                "method": method,
                "path":   path,
            })
        else:
            failed.append(row["pmcid"])
        time.sleep(0.4)
    print(f"Downloaded: {len(downloaded)} | Failed: {len(failed)}")
    return pd.DataFrame(downloaded)


# ── Image Manifest ─────────────────────────────────────────────────────────────

def build_image_manifest() -> pd.DataFrame:
    """
    Index HAM10000 + ISIC images into a manifest DataFrame.
    Each row: image_id, image_path, dataset, split, label, image_text_description.
    """
    all_records = []

    # HAM10000
    ham_meta_path = HAM_DIR / "HAM10000_metadata.csv"
    if ham_meta_path.exists():
        ham_meta = pd.read_csv(ham_meta_path)
        ham_meta["label"] = ham_meta["dx"].map(HAM_LABEL_MAP)
        label_map = dict(zip(ham_meta["image_id"], ham_meta["label"]))

        for folder in [HAM_DIR / "HAM10000_images_part_1",
                       HAM_DIR / "HAM10000_images_part_2"]:
            if not folder.exists():
                print(f"Missing HAM10000 folder: {folder}")
                continue
            for img in folder.glob("*.jpg"):
                all_records.append({
                    "image_id":   img.stem,
                    "image_path": str(img),
                    "dataset":    "HAM10000",
                    "split":      "unknown",
                    "label":      label_map.get(img.stem, "unknown"),
                })
        print(f"HAM10000: {len(all_records)} images indexed")
    else:
        print(f"HAM10000 metadata not found at {ham_meta_path}")

    # ISIC — auto-discover root
    isic_root = None
    for candidate in Path("/kaggle/input").rglob("Train"):
        if candidate.is_dir() and "isic" in str(candidate).lower():
            isic_root = candidate.parent
            break
    if isic_root is None:
        for candidate in Path("/kaggle/input").rglob("Train"):
            if candidate.is_dir():
                subdirs = [d for d in candidate.iterdir() if d.is_dir()]
                if len(subdirs) > 2:
                    isic_root = candidate.parent
                    break

    isic_count = 0
    if isic_root:
        for split in ["Train", "Test"]:
            split_path = isic_root / split
            if not split_path.exists():
                continue
            for cls in split_path.iterdir():
                if not cls.is_dir():
                    continue
                for img in cls.glob("*.jpg"):
                    all_records.append({
                        "image_id":   img.stem,
                        "image_path": str(img),
                        "dataset":    "ISIC",
                        "split":      split,
                        "label":      cls.name.lower().strip(),
                    })
                    isic_count += 1
        print(f"ISIC: {isic_count} images indexed")
    else:
        print("ISIC dataset not found.")

    if not all_records:
        print("WARNING: No images found.")
        return pd.DataFrame(columns=[
            "image_id", "image_path", "dataset", "split",
            "label", "image_text_description",
        ])

    df = pd.DataFrame(all_records).drop_duplicates(subset=["image_path"])
    df["image_text_description"] = (
        "Dermoscopy image from " + df["dataset"] + " dataset. "
        + "Diagnostic label: " + df["label"] + ". "
        + "Split: " + df["split"] + "."
    )
    print(f"Total images indexed: {len(df)}")
    return df


# ── Quick chunk helper (used by pipeline before document_processor is called) ──

def chunk_text_simple(text: str,
                       max_words: int = CHUNK_MAX_WORDS,
                       overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple word-count chunker with overlap."""
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [text]
    step, chunks, start = max_words - overlap, [], 0
    while start < len(words):
        end   = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += step
    return chunks
