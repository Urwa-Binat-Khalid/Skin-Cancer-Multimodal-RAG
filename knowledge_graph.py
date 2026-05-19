"""
knowledge_graph.py — Phase 5: Knowledge Graph
Covers:
  - Custom medical entity lexicon + scispaCy NER
  - Pattern-based relation extraction (DETECTS, ACHIEVES, TRAINED_ON, IMPROVES)
  - Co-occurrence relation extraction (sentence-level)
  - NetworkX DiGraph construction and serialisation
"""

import re
import itertools
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import spacy
from tqdm import tqdm

from config import ENTITY_LEXICON, RELATION_PATTERNS, KG_SAVE_PATH


# ── NLP model ──────────────────────────────────────────────────────────────────

def load_nlp():
    """Load scispaCy NER model; fall back to en_core_web_sm."""
    try:
        nlp = spacy.load("en_ner_bc5cdr_md")
        print("scispaCy en_ner_bc5cdr_md loaded.")
    except OSError:
        nlp = spacy.load("en_core_web_sm")
        print("Fallback: en_core_web_sm loaded.")
    return nlp


# ── Entity lexicon map ─────────────────────────────────────────────────────────

def build_lexicon_map(lexicon: dict = ENTITY_LEXICON) -> dict[str, str]:
    """Flat map of term → entity_type from the lexicon dict."""
    return {
        term.lower(): etype
        for etype, terms in lexicon.items()
        for term in terms
    }


# ── Entity extraction ──────────────────────────────────────────────────────────

def get_entities(text: str, nlp, lexicon_map: dict) -> list[dict]:
    """
    Extract entities from text using:
      1. Lexicon substring match (fast, high-precision for domain terms)
      2. scispaCy NER (broader coverage)
    Returns list of {"text": str, "label": str}.
    """
    found: dict[str, dict] = {}
    text_lower = text.lower()

    for phrase, etype in lexicon_map.items():
        if phrase in text_lower:
            found[phrase] = {"text": phrase, "label": etype}

    try:
        doc = nlp(text[:10000])
        for ent in doc.ents:
            key = ent.text.lower().strip()
            if len(key) > 2 and key not in found:
                found[key] = {"text": key, "label": ent.label_}
    except Exception:
        pass

    return list(found.values())


# ── Graph construction ─────────────────────────────────────────────────────────

def build_knowledge_graph(df_chunks: pd.DataFrame, nlp) -> nx.DiGraph:
    """
    Build a directed knowledge graph from all text chunks.

    Node attributes : label, entity_type, frequency
    Edge attributes : relation, weight, evidence (chunk_id list)

    Returns a NetworkX DiGraph.
    """
    lexicon_map  = build_lexicon_map()
    G            = nx.DiGraph()
    entity_freq: dict[str, int]       = defaultdict(int)
    entity_types: dict[str, str]      = {}
    edge_weights: dict[tuple, int]    = defaultdict(int)
    edge_evidence: dict[tuple, list]  = defaultdict(list)

    for _, row in tqdm(df_chunks.iterrows(),
                       total=len(df_chunks), desc="Building knowledge graph"):
        text     = str(row["text"])
        cid      = str(row["chunk_id"])
        entities = get_entities(text, nlp, lexicon_map)

        # Accumulate entity frequencies
        for e in entities:
            k = e["text"]
            entity_freq[k]  += 1
            entity_types[k]  = e["label"]

        # Pattern-based relations
        for pattern, rel_type in RELATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = [g.strip().lower() for g in match.groups() if g]
                if len(groups) >= 2:
                    subj = groups[0][-60:]
                    obj  = groups[1][:60]
                    if len(subj) > 2 and len(obj) > 2 and subj != obj:
                        key = (subj, rel_type, obj)
                        edge_weights[key]  += 1
                        edge_evidence[key].append(cid)

        # Co-occurrence (sentence level)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            sl    = sent.lower()
            found = [e for e in entities if e["text"] in sl]
            for e1, e2 in itertools.combinations(found, 2):
                if e1["label"] == e2["label"] == "METRIC":
                    continue
                key = (e1["text"], "CO_OCCURS", e2["text"])
                edge_weights[key]  += 1
                edge_evidence[key].append(cid)

    # Add nodes
    for ent, freq in entity_freq.items():
        G.add_node(
            ent,
            label=ent,
            entity_type=entity_types.get(ent, "UNKNOWN"),
            frequency=freq,
        )

    # Add edges
    for (subj, rel, obj), weight in edge_weights.items():
        if not subj or not obj or subj == obj:
            continue
        for n in [subj, obj]:
            if n not in G:
                G.add_node(n, label=n, entity_type="UNKNOWN", frequency=1)
        evidence = "; ".join(list(set(edge_evidence[(subj, rel, obj)]))[:5])
        G.add_edge(subj, obj, relation=rel, weight=weight, evidence=evidence)

    print(f"Knowledge graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


# ── Serialisation ──────────────────────────────────────────────────────────────

def save_graph(G: nx.DiGraph, path: Path = KG_SAVE_PATH) -> None:
    nx.write_graphml(G, str(path))
    print(f"Knowledge graph saved → {path}")


def load_graph(path: Path = KG_SAVE_PATH) -> nx.DiGraph:
    G = nx.read_graphml(str(path))
    print(f"Knowledge graph loaded: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


# ── Graph-based context extraction (used by retrieval.py) ─────────────────────

def graph_context_for_query(query: str,
                             G: nx.DiGraph,
                             max_lines: int = 10) -> str:
    """
    Find entities in the query, walk their neighbourhood,
    return a short string of triples for injection into the LLM prompt.
    """
    q_lower  = query.lower()
    entities = [n for n in G.nodes() if len(n) > 3 and n in q_lower]
    lines    = []
    for ent in entities[:3]:
        for _, nbr, data in G.out_edges(ent, data=True):
            lines.append(f"{ent} --[{data.get('relation', '')}]--> {nbr}")
        for pred, _, data in G.in_edges(ent, data=True):
            lines.append(f"{pred} --[{data.get('relation', '')}]--> {ent}")
    return "\n".join(lines[:max_lines])
