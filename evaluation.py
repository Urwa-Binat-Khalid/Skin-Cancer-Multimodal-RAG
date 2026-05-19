"""
evaluation.py — Phase 11: Evaluation
Covers:
  - Offline retrieval metrics (no LLM needed):
      top/avg rerank score, citation coverage, keyword hit rate,
      retrieval diversity, graph context usage
  - RAGAS metrics (needs Groq quota):
      Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
  - Retrieval strategy comparison: BM25 only vs Dense only vs Hybrid
  - Bar-chart visualisation of comparison results
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import DIRS
from retrieval import hybrid_retrieve, bm25_search, vector_search, do_rerank
from llm import ask


EVAL_DIR = DIRS["evaluation"]

TEST_QUERIES = [
    {
        "query":        "What CNN architectures are used for melanoma classification?",
        "ground_truth": (
            "CNN architectures such as ResNet, EfficientNet, VGG, DenseNet, "
            "and Inception are used for melanoma classification from dermoscopy images."
        ),
    },
    {
        "query":        "What is the accuracy of deep learning for skin cancer detection?",
        "ground_truth": (
            "Deep learning models report high accuracy for skin cancer detection, "
            "varying by dataset, lesion type, and validation method."
        ),
    },
    {
        "query":        "How is dermoscopy used in skin lesion diagnosis?",
        "ground_truth": (
            "Dermoscopy is a non-invasive imaging technique used to visualize "
            "subsurface skin structures and support assessment of skin lesions."
        ),
    },
    {
        "query":        "What datasets are commonly used for skin cancer classification?",
        "ground_truth": (
            "Common datasets include HAM10000, ISIC Archive, PH2, and Derm7pt."
        ),
    },
    {
        "query":        "What is the difference between melanoma and nevus in dermoscopy?",
        "ground_truth": (
            "Melanoma shows asymmetric structure, irregular borders, multiple colors, "
            "while benign nevi are more symmetric and regular."
        ),
    },
]

_STOP_WORDS = {
    "what", "which", "where", "when", "does", "used",
    "skin", "the", "and", "for", "how", "with",
}


# ── Part A: Offline retrieval metrics ─────────────────────────────────────────

def run_offline_eval(test_queries: list[dict] = TEST_QUERIES) -> pd.DataFrame:
    """
    Evaluate retrieval quality without calling the LLM.
    Metrics: top/avg rerank score, citation coverage, context coverage,
             keyword hit rate, retrieval diversity, graph context flag.
    """
    print("\n── Offline Retrieval Metrics ────────────────────────────")
    rows = []

    for item in tqdm(test_queries, desc="Offline eval"):
        query  = item["query"]
        hybrid = hybrid_retrieve(query, top_k=5)
        results = hybrid["results"]

        rerank_scores = [r.get("rerank_score", 0) for r in results]
        methods_used  = set(r.get("method", "") for r in results)
        cited = [r for r in results if r.get("pmid", "") or r.get("pmcid", "")]
        with_text = [r for r in results if len(r.get("text", "")) > 50]

        keywords   = [
            w for w in query.lower().split()
            if len(w) > 4 and w not in _STOP_WORDS
        ]
        top_text   = results[0].get("text", "").lower() if results else ""
        kw_hits    = sum(1 for kw in keywords if kw in top_text)
        kw_rate    = round(kw_hits / max(len(keywords), 1), 3)

        rows.append({
            "query":               query[:60],
            "top_rerank_score":    round(max(rerank_scores), 3) if rerank_scores else 0,
            "avg_rerank_score":    round(
                sum(rerank_scores) / max(len(rerank_scores), 1), 3
            ),
            "citation_coverage":   round(len(cited) / max(len(results), 1), 3),
            "context_coverage":    round(len(with_text) / max(len(results), 1), 3),
            "keyword_hit_rate":    kw_rate,
            "retrieval_diversity": len(methods_used),
            "methods_used":        "+".join(sorted(methods_used)),
            "has_graph_context":   bool(hybrid.get("graph_ctx", "")),
            "vector_hits":         hybrid["stats"]["vector"],
            "bm25_hits":           hybrid["stats"]["bm25"],
            "graph_hits":          hybrid["stats"]["graph"],
            "total_candidates":    hybrid["stats"]["merged"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "offline_eval.csv", index=False)
    _print_offline_summary(df)
    return df


def _print_offline_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print("OFFLINE RETRIEVAL EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"  Queries evaluated       : {len(df)}")
    print(f"  Avg top rerank score    : {df['top_rerank_score'].mean():.3f}")
    print(f"  Avg mean rerank score   : {df['avg_rerank_score'].mean():.3f}")
    print(f"  Avg citation coverage   : {df['citation_coverage'].mean():.3f}")
    print(f"  Avg keyword hit rate    : {df['keyword_hit_rate'].mean():.3f}")
    print(f"  Avg retrieval diversity : {df['retrieval_diversity'].mean():.1f} methods")
    print(f"  Graph context used      : "
          f"{df['has_graph_context'].sum()}/{len(df)} queries")


# ── Part B: Generate LLM answers ───────────────────────────────────────────────

def generate_llm_answers(test_queries: list[dict] = TEST_QUERIES,
                          inter_query_wait: int = 8) -> list[dict]:
    """Generate answers for RAGAS; skips failed/rate-limited calls."""
    print("\n── Generating LLM Answers ───────────────────────────────")
    ragas_rows = []

    for item in tqdm(test_queries, desc="LLM answers"):
        answer = ""
        for attempt in range(3):
            result = ask(item["query"], top_k=5, include_images=False)
            answer = result.get("answer", "")
            if "LLM error" not in answer and len(answer) > 50:
                break
            wait = 20 * (attempt + 1)
            print(f"  Rate-limited — waiting {wait}s (attempt {attempt+1}/3)")
            time.sleep(wait)

        if "LLM error" in answer or len(answer) < 50:
            print(f"  Skipped: {item['query'][:50]}")
            continue

        contexts = [
            s.get("text", "")
            for s in result.get("sources", [])
            if s.get("text", "")
        ]
        ragas_rows.append({
            "user_input":         item["query"],
            "response":           answer,
            "retrieved_contexts": contexts,
            "reference":          item["ground_truth"],
        })
        time.sleep(inter_query_wait)

    print(f"Generated {len(ragas_rows)}/{len(test_queries)} answers.")
    return ragas_rows


# ── Part C: RAGAS evaluation ───────────────────────────────────────────────────

def run_ragas_eval(ragas_rows: list[dict],
                   groq_api_key: str,
                   device: str = "cpu") -> pd.DataFrame | None:
    """Run RAGAS metrics. Returns DataFrame or None on failure."""
    if not ragas_rows:
        print("No RAGAS rows — skipping.")
        return None

    print(f"\n── RAGAS Metrics on {len(ragas_rows)} answers ──────────")
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            Faithfulness, AnswerRelevancy,
            LLMContextPrecisionWithoutReference, LLMContextRecall,
        )
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings

        ragas_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=groq_api_key,
        )
        ragas_emb = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        df_input = pd.DataFrame(ragas_rows)
        df_input.to_csv(EVAL_DIR / "ragas_input.csv", index=False)
        dataset  = Dataset.from_pandas(df_input)

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                LLMContextPrecisionWithoutReference(),
                LLMContextRecall(),
            ],
            llm=ragas_llm,
            embeddings=ragas_emb,
        )

        df_ragas = result.to_pandas()
        df_ragas.to_csv(EVAL_DIR / "ragas_results.csv", index=False)
        _print_ragas_summary(df_ragas)
        return df_ragas

    except Exception as e:
        print(f"RAGAS error: {e}")
        return None


def _print_ragas_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print("RAGAS EVALUATION SUMMARY")
    print(f"{'='*55}")
    for col, label in [
        ("faithfulness",                            "Faithfulness       "),
        ("answer_relevancy",                        "Answer Relevancy   "),
        ("llm_context_precision_without_reference", "Context Precision  "),
        ("context_recall",                          "Context Recall     "),
    ]:
        val = df[col].mean() if col in df.columns else float("nan")
        if not pd.isna(val):
            print(f"  ✓ {label}: {val:.3f}")
        else:
            print(f"  ✗ {label}: N/A")


# ── Part D: Retrieval strategy comparison ─────────────────────────────────────

COMPARE_QUERIES = [
    "CNN architecture melanoma classification dermoscopy",
    "basal cell carcinoma deep learning accuracy",
    "dermoscopy skin lesion diagnosis transformer",
    "HAM10000 ISIC dataset skin cancer classification",
    "melanoma nevus difference visual features",
]


def run_strategy_comparison(queries: list[str] = COMPARE_QUERIES) -> pd.DataFrame:
    """Compare BM25-only, Dense-only, and Hybrid retrieval strategies."""
    print("\n── Retrieval Strategy Comparison ────────────────────────")
    rows = []

    for query in tqdm(queries, desc="Strategy comparison"):
        bm25_only   = bm25_search(query, top_k=5)
        bm25_scores = [r.get("rerank_score", 0)
                       for r in do_rerank(query, bm25_only, top_k=5)]

        dense_only   = vector_search(query, top_k=5)
        dense_scores = [r.get("rerank_score", 0)
                        for r in do_rerank(query, dense_only, top_k=5)]

        hybrid       = hybrid_retrieve(query, top_k=5)
        hybrid_scores = [r.get("rerank_score", 0) for r in hybrid["results"]]

        rows.append({
            "query":             query[:45],
            "bm25_top1":         round(max(bm25_scores),   3) if bm25_scores   else 0,
            "dense_top1":        round(max(dense_scores),  3) if dense_scores  else 0,
            "hybrid_top1":       round(max(hybrid_scores), 3) if hybrid_scores else 0,
            "bm25_mean":         round(sum(bm25_scores)   / max(len(bm25_scores),   1), 3),
            "dense_mean":        round(sum(dense_scores)  / max(len(dense_scores),  1), 3),
            "hybrid_mean":       round(sum(hybrid_scores) / max(len(hybrid_scores), 1), 3),
            "hybrid_candidates": hybrid["stats"]["merged"],
            "hybrid_graph_used": bool(hybrid.get("graph_ctx", "")),
        })

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "retrieval_comparison.csv", index=False)
    _print_comparison_table(df)
    plot_comparison(df)
    return df


def _print_comparison_table(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print(f"{'Query':<45} {'BM25':>7} {'Dense':>7} {'Hybrid':>7}")
    print(f"{'─'*45} {'─'*7} {'─'*7} {'─'*7}")
    for _, row in df.iterrows():
        print(f"{row['query']:<45} "
              f"{row['bm25_top1']:>7.3f} "
              f"{row['dense_top1']:>7.3f} "
              f"{row['hybrid_top1']:>7.3f}")
    print(f"\n{'─'*70}")
    print(f"{'AVERAGE':<45} "
          f"{df['bm25_top1'].mean():>7.3f} "
          f"{df['dense_top1'].mean():>7.3f} "
          f"{df['hybrid_top1'].mean():>7.3f}")
    winner_scores = {
        "BM25":   df["bm25_top1"].mean(),
        "Dense":  df["dense_top1"].mean(),
        "Hybrid": df["hybrid_top1"].mean(),
    }
    best = max(winner_scores, key=winner_scores.get)
    print(f"\n  Best strategy: {best} "
          f"(avg top-1 score = {winner_scores[best]:.3f})")


def plot_comparison(df: pd.DataFrame,
                    save_path: Path | None = None) -> None:
    """Bar chart: BM25 vs Dense vs Hybrid per query and averaged."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x, w = range(len(df)), 0.25
    axes[0].bar([i - w for i in x], df["bm25_top1"],  w,
                label="BM25 only", color="#4A90D9", alpha=0.85)
    axes[0].bar([i     for i in x], df["dense_top1"], w,
                label="Dense only", color="#E67E22", alpha=0.85)
    axes[0].bar([i + w for i in x], df["hybrid_top1"], w,
                label="Hybrid", color="#27AE60", alpha=0.85)
    axes[0].set_title("Top-1 Rerank Score by Strategy", fontsize=13, fontweight="bold")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(
        [q[:20] + "..." for q in df["query"]],
        rotation=30, ha="right", fontsize=8,
    )
    axes[0].set_ylabel("Rerank Score")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    strategies = ["BM25 only", "Dense only", "Hybrid"]
    avgs       = [df["bm25_top1"].mean(), df["dense_top1"].mean(), df["hybrid_top1"].mean()]
    colors     = ["#4A90D9", "#E67E22", "#27AE60"]
    bars       = axes[1].bar(strategies, avgs, color=colors, alpha=0.85, width=0.4)
    axes[1].set_title("Average Top-1 Score", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Avg Top-1 Rerank Score")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, avgs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11,
        )
    best_idx = avgs.index(max(avgs))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2.5)
    axes[1].text(best_idx, avgs[best_idx] + 0.05, "★ BEST",
                 ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Retrieval Strategy Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = save_path or (EVAL_DIR / "retrieval_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Comparison chart saved → {out}")
    plt.show()
