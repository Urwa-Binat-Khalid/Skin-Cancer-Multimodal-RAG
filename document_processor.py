"""
document_processor.py — Phase 2: Document Processing
Covers:
  - JATS XML and BioC XML parsing (parse_xml_proper)
  - Text cleaning (strip tags, remove citations, normalise whitespace)
  - Sentence-boundary-aware chunking (chunk_text)
  - Building the full RAG chunk DataFrame from downloaded XML + abstract fallback
"""

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import (
    DIRS, CHUNK_MAX_WORDS, CHUNK_OVERLAP,
    ABSTRACT_MAX_WORDS, MIN_CHUNK_CHARS,
)

# Tags whose content we always discard
SKIP_TAGS = {
    "xref", "ref", "ref-list", "table", "fig",
    "supplementary-material", "media", "ext-link",
}


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip XML tags, citation markers, figure refs, and extra whitespace."""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\s*[\d,\s\-]+\s*\]", "", text)                          # [1], [2,3]
    text = re.sub(r"\(\s*(Fig|Figure|Table)\.?\s*\d+\s*\)", "",
                  text, flags=re.IGNORECASE)
    return text.strip()


# ── XML Parser ─────────────────────────────────────────────────────────────────

def parse_xml_proper(xml_path: Path) -> dict:
    """
    Parse a PMC XML file (JATS or BioC format).
    Returns:
        {
            "title":    str,
            "abstract": str,
            "sections": [{"section_title": str, "text": str}, ...],
            "figures":  [{"figure_label": str, "caption": str}, ...],
        }
    Returns empty structure for HTML files or unreadable content.
    """
    raw = xml_path.read_text(encoding="utf-8", errors="ignore")

    # Skip HTML files (broken downloads)
    if raw.lstrip().startswith("<!DOCTYPE html") or "<html" in raw[:200]:
        return {"title": "", "abstract": "", "sections": [], "figures": []}

    soup = BeautifulSoup(raw, "lxml-xml")

    # ── BioC format ───────────────────────────────────────────────────────────
    if soup.find("collection"):
        passages = []
        for p in soup.find_all("passage"):
            infon = ""
            for i in p.find_all("infon"):
                if i.get("key") == "type":
                    infon = (i.text or "").lower()
            if any(x in infon for x in ["table", "fig", "ref", "caption"]):
                continue
            t = p.find("text")
            if t and t.text and len(t.text.strip()) > 20:
                passages.append({
                    "section_title": infon or "Body",
                    "text": clean_text(t.text),
                })
        sections: dict = {}
        for p in passages:
            sections.setdefault(p["section_title"], []).append(p["text"])
        return {
            "title":    "",
            "abstract": "",
            "sections": [
                {"section_title": k, "text": " ".join(v)}
                for k, v in sections.items()
            ],
            "figures": [],
        }

    # ── JATS format ───────────────────────────────────────────────────────────
    title = ""
    t_tag = soup.find("article-title")
    if t_tag:
        title = clean_text(t_tag.get_text(" "))

    abstract = ""
    a_tag = soup.find("abstract")
    if a_tag:
        abstract = clean_text(a_tag.get_text(" "))

    sections = []
    body = soup.find("body")
    if body:
        for sec in body.find_all("sec", recursive=True):
            h       = sec.find("title", recursive=False)
            heading = clean_text(h.get_text(" ")) if h else "Section"
            paras   = []
            for p in sec.find_all("p", recursive=False):
                txt = clean_text(p.get_text(" "))
                if len(txt) > 30:
                    paras.append(txt)
            if paras:
                sections.append({
                    "section_title": heading,
                    "text": " ".join(paras),
                })

    figures = []
    for fig in soup.find_all("fig"):
        label   = fig.find("label")
        caption = fig.find("caption")
        cap_txt = clean_text(caption.get_text(" ")) if caption else ""
        lbl_txt = clean_text(label.get_text(" "))   if label   else ""
        if cap_txt and len(cap_txt) > 30:
            figures.append({"figure_label": lbl_txt, "caption": cap_txt})

    return {
        "title":    title,
        "abstract": abstract,
        "sections": sections,
        "figures":  figures,
    }


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str,
               max_words: int = CHUNK_MAX_WORDS,
               overlap: int   = CHUNK_OVERLAP) -> list[str]:
    """
    Word-count chunker with sentence-boundary heuristic at 80% of chunk length.
    Returns a list of non-empty string chunks.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [text]
    step, chunks, start = max_words - overlap, [], 0
    while start < len(words):
        end   = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        if end < len(words):
            boundary = int(len(chunk) * 0.80)
            lp = max(chunk.rfind(". ", boundary), chunk.rfind(".\n", boundary))
            if lp > 0:
                chunk = chunk[:lp + 1]
        chunks.append(chunk.strip())
        start += step
    return chunks


# ── Build RAG Chunk DataFrame ──────────────────────────────────────────────────

def build_chunks(df_downloaded: pd.DataFrame,
                 df_abstracts: pd.DataFrame) -> pd.DataFrame:
    """
    Parse all downloaded XML files and build the full chunk DataFrame.
    Falls back to PubMed abstracts for papers without full text.

    Args:
        df_downloaded: manifest from data_ingestion.download_all_fulltexts()
        df_abstracts:  DataFrame from data_ingestion.fetch_abstracts()

    Returns:
        DataFrame with columns:
            chunk_id, pmcid, pmid, source_type, section, text,
            word_count, title, journal
    """
    xml_files = list(DIRS["pmc_fulltext"].glob("*.xml"))
    print(f"Parsing {len(xml_files)} XML files...")

    all_chunks     = []
    fulltext_pmcids = set()

    for xf in tqdm(xml_files, desc="Parsing XML"):
        pmcid = xf.stem.split("_")[0]
        parsed = parse_xml_proper(xf)

        pmid_match = df_downloaded.loc[
            df_downloaded["pmcid"] == pmcid, "pmid"
        ].values
        pmid = str(pmid_match[0]) if len(pmid_match) > 0 else ""

        has_content = (
            len(parsed.get("abstract", "")) > 50
            or len(parsed.get("sections", [])) > 0
        )
        if not has_content:
            continue

        fulltext_pmcids.add(pmcid)

        # Abstract chunk
        if parsed["abstract"]:
            for i, chunk in enumerate(
                chunk_text(parsed["abstract"], max_words=ABSTRACT_MAX_WORDS)
            ):
                all_chunks.append({
                    "chunk_id":    f"{pmcid}_abstract_{i}",
                    "pmcid":       pmcid,
                    "pmid":        pmid,
                    "source_type": "PMC XML",
                    "section":     "Abstract",
                    "text":        chunk,
                    "word_count":  len(chunk.split()),
                    "title":       parsed["title"],
                    "journal":     "",
                })

        # Section chunks
        for sec in parsed["sections"]:
            heading = sec["section_title"]
            text    = sec["text"]
            if not text or len(text) < 30:
                continue
            safe_heading = re.sub(r"[^a-zA-Z0-9]", "_", heading)
            for i, chunk in enumerate(chunk_text(text)):
                all_chunks.append({
                    "chunk_id":    f"{pmcid}_{safe_heading}_{i}",
                    "pmcid":       pmcid,
                    "pmid":        pmid,
                    "source_type": "PMC XML",
                    "section":     heading,
                    "text":        chunk,
                    "word_count":  len(chunk.split()),
                    "title":       parsed["title"],
                    "journal":     "",
                })

        # Figure captions
        for fi, fig in enumerate(parsed["figures"]):
            cap  = fig["caption"]
            lbl  = fig.get("figure_label", f"Figure {fi + 1}")
            text = clean_text(f"{lbl}. {cap}")
            if len(text) > 30:
                all_chunks.append({
                    "chunk_id":    f"{pmcid}_fig_{fi}",
                    "pmcid":       pmcid,
                    "pmid":        pmid,
                    "source_type": "PMC XML",
                    "section":     "Figure Caption",
                    "text":        text,
                    "word_count":  len(text.split()),
                    "title":       parsed["title"],
                    "journal":     "",
                })

    # Abstract-only fallback for paywalled papers
    downloaded_pmids = set(df_downloaded["pmid"].astype(str))
    for _, row in df_abstracts.iterrows():
        pmid = str(row["pmid"])
        if pmid in downloaded_pmids:
            continue
        ab = str(row.get("abstract", "")).strip()
        if len(ab) < 50:
            continue
        for i, chunk in enumerate(
            chunk_text(ab, max_words=ABSTRACT_MAX_WORDS)
        ):
            all_chunks.append({
                "chunk_id":    f"{pmid}_abs_c{i:03d}",
                "pmcid":       "",
                "pmid":        pmid,
                "source_type": "PubMed Abstract",
                "section":     "Abstract",
                "text":        chunk,
                "word_count":  len(chunk.split()),
                "title":       str(row.get("title",   "")),
                "journal":     str(row.get("journal", "")),
            })

    df = pd.DataFrame(all_chunks)
    df = df[df["text"].str.strip().str.len() > MIN_CHUNK_CHARS].reset_index(drop=True)

    # Fill missing columns
    for col in ["journal", "title", "section", "pmid", "pmcid"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    print(f"Total chunks built : {len(df)}")
    print(df["source_type"].value_counts().to_string())
    return df
