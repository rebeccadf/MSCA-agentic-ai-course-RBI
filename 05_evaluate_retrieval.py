"""Measure retrieval quality offline (no LLM calls) and log it to MLflow.

FinanceBench labels the page(s) of the filing that answer each question. For each
question we retrieve chunks the same way 02_run_agents.py does and measure how
much of the evidence page the retrieved chunks cover. The evidence is only used
here, for scoring; the agents never see it.

Examples:
    python 05_evaluate_retrieval.py                                         # current default
    python 05_evaluate_retrieval.py --filter company                        # whole company, as originally
"""
import argparse
import time

import mlflow
from datasets import load_dataset

import rag
from rag import Retriever, tokenize

HIT_THRESHOLD = 0.25
NGRAM = 5


def shingles(text: str):
    tokens = tokenize(text)
    return {tuple(tokens[i:i + NGRAM]) for i in range(len(tokens) - NGRAM + 1)}


def evidence_coverage(evidence_text: str, chunks) -> float:
    """Share of the evidence page's 5-word sequences that appear in the retrieved chunks.

    Sequences rather than single words, so the score only rises when the
    evidence passage itself was retrieved, not just text with similar vocabulary.
    """
    evidence = shingles(evidence_text)
    if not evidence:
        return 0.0
    retrieved = set()
    for chunk in chunks:
        retrieved.update(shingles(chunk.get_content()))
    return len(evidence & retrieved) / len(evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", choices=["doc_name", "company"], default="doc_name")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    qa = load_dataset("Liadmagen/financebench_QA")["train"]
    if args.limit:
        qa = qa.shuffle(seed=0).select(range(args.limit))

    retriever = Retriever(filter_key=args.filter)

    coverages, by_type = [], {}
    start = time.perf_counter()
    for i, row in enumerate(qa):
        chunks = retriever.search(row["question"], row[args.filter], top_k=args.top_k)
        # A question can cite several pages; score the best-covered one
        cov = max(evidence_coverage(e["evidence_text"], chunks) for e in row["evidence"])
        coverages.append(cov)
        by_type.setdefault(row["question_type"], []).append(cov)
        print(f"{i + 1}/{len(qa)} coverage={cov:.2f}", end="\r", flush=True)
    elapsed = time.perf_counter() - start

    metrics = {
        "evidence_coverage": sum(coverages) / len(coverages),
        "evidence_hit_rate": sum(c >= HIT_THRESHOLD for c in coverages) / len(coverages),
        "seconds_per_query": elapsed / len(coverages),
    }
    for qtype, covs in by_type.items():
        metrics[f"hit_rate_{qtype}"] = sum(c >= HIT_THRESHOLD for c in covs) / len(covs)

    rag.setup_mlflow("financebench-retrieval")
    name = f"chroma-{args.filter}-k{args.top_k}"
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "vector_store": "chroma",
            "filter": args.filter,
            "top_k": args.top_k,
            "chunk_size": rag.CHUNK_SIZE,
            "questions": len(coverages),
        })
        mlflow.log_metrics(metrics)

    print()
    print(f"RETRIEVAL REPORT ({name}, {len(coverages)} questions)")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"{k:28s} {v:.3f}")


if __name__ == "__main__":
    main()
