"""
llm.py — Phase 7-8-9: LLM Reasoning + Citations
Covers:
  - Groq client with rate-limit retry
  - Medical safety system prompt (hallucination prevention + safety rules)
  - build_prompt() — structures retrieved context + graph triples for the LLM
  - ask() — end-to-end: retrieve → prompt → LLM → structured response
"""

import time
from groq import Groq

from config import (
    GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    LLM_MAX_RETRIES, LLM_RETRY_WAIT, MEDICAL_DISCLAIMER,
)
from retrieval import retrieve_for_rag


# ── Groq client ────────────────────────────────────────────────────────────────

_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── System prompt ──────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a medical research-support assistant for skin cancer and dermatology.

IMPORTANT MEDICAL SAFETY RULES:
1. This system is for research and educational support only.
2. You are NOT a doctor and must NOT provide a clinical diagnosis.
3. You must NOT recommend a final treatment plan.
4. You must NOT tell the user they have or do not have cancer.
5. You must advise consulting a qualified dermatologist for diagnosis or treatment decisions.
6. If the user describes urgent symptoms such as rapid growth, bleeding, severe pain, infection, \
or major change in a lesion, advise urgent medical evaluation.

HALLUCINATION PREVENTION RULES:
1. Answer ONLY using the retrieved context.
2. Every important claim must cite a source using [Source N].
3. If the retrieved context does not contain enough information, say:
   "The retrieved context is insufficient to answer this confidently."
4. Do NOT use outside medical knowledge unless it is explicitly present in the retrieved context.
5. Do NOT invent model names, datasets, metrics, AUC values, treatment details, or clinical claims.
6. If sources disagree, mention the disagreement.
7. Separate research findings from clinical interpretation.
8. Use cautious language: "may", "reported", "associated with", "the retrieved studies suggest".

IMAGE SAFETY RULES:
1. For images, describe retrieved visual similarity only.
2. Do NOT call visual similarity a diagnosis.
3. Do NOT output "100% confidence" for medical labels.
4. Use "similar-image label suggestion" instead of "predicted diagnosis".

Format your response as:
ANSWER: [grounded answer using citations]
MEDICAL SAFETY NOTE: This is not medical advice and not a diagnosis. \
Please consult a qualified dermatologist.
CONFIDENCE: [HIGH/MEDIUM/LOW based only on retrieved context]
REASONING: [why the answer is supported or uncertain]
CITATIONS: [Source numbers used]
"""


# ── LLM call with retry ────────────────────────────────────────────────────────

def call_llm_with_retry(prompt: str,
                         max_retries: int = LLM_MAX_RETRIES,
                         retry_wait: int  = LLM_RETRY_WAIT) -> str:
    """Call Groq API; retry on rate-limit (429) up to max_retries times."""
    client = get_groq_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries - 1:
                wait = retry_wait * (attempt + 1)
                print(f"  Rate limited — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                return f"LLM error: {e}"
    return "LLM error: max retries exceeded"


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(query: str,
                 sources: list[dict],
                 graph_context: str = "") -> str:
    """
    Assemble the full prompt sent to the LLM.
    Each source is rendered as a numbered block with PMID/PMCID header.
    """
    source_block = "\n\n".join([
        f"[Source {i + 1}] ({s.get('section', '')}) — {s.get('title', '')[:60]}\n"
        f"PMID:{s.get('pmid', '')} | PMCID:{s.get('pmcid', '')}\n"
        f"{s.get('text', '')}"
        for i, s in enumerate(sources)
    ])

    graph_block = (
        f"\nKNOWLEDGE GRAPH RELATIONSHIPS:\n{graph_context}"
        if graph_context else ""
    )

    return (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"RETRIEVED CONTEXT:\n{source_block}"
        f"{graph_block}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"Provide a cautious research-support answer with citations.\n"
        f"Do not provide a diagnosis.\n"
    )


# ── Main ask function ──────────────────────────────────────────────────────────

def ask(query: str,
        top_k: int          = 5,
        include_images: bool = True) -> dict:
    """
    End-to-end RAG: hybrid retrieval → prompt construction → LLM → response dict.

    Returns:
        {
            "query":     str,
            "answer":    str,
            "sources":   list[dict],
            "images":    list[dict],
            "graph_ctx": str,
            "stats":     dict,
        }
    """
    retrieval = retrieve_for_rag(
        query,
        top_k=top_k,
        include_images=include_images,
    )

    prompt = build_prompt(
        query=query,
        sources=retrieval["sources"],
        graph_context=retrieval["graph_ctx"],
    )

    answer = call_llm_with_retry(prompt)

    return {
        "query":     query,
        "answer":    answer,
        "sources":   retrieval["sources"],
        "images":    retrieval["images"],
        "graph_ctx": retrieval["graph_ctx"],
        "stats":     retrieval["stats"],
    }
