import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ast
import asyncio
import json
import operator
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlflow
from datasets import load_dataset
from pydantic import BaseModel
from agents import (
    Agent,
    MaxTurnsExceeded,
    Runner,
    RunContextWrapper,
    function_tool,
)

import llm
import rag
from llm import fast_model, strong_model
from rag import HybridRetriever

PROJECT_DIR = Path(__file__).parent
MLFLOW_EXPERIMENT = "financebench-agents"

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Answer FinanceBench questions with agents.")
    parser.add_argument("--run", default="full",
                        help="Run name; results go to runs/<run>/results.json (default: full)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run a fixed random sample of N questions (same sample every time)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=80,
                        help="Chunks given to the agents up front (default: 80, about 40k tokens)")
    parser.add_argument("--hybrid", action="store_true", help="Add BM25 keyword retrieval")
    parser.add_argument("--rerank", action="store_true", help="Rerank with the cross-encoder")
    parser.add_argument("--search-top-k", type=int, default=6,
                        help="Chunks returned by each search_filing call (default: 6)")
    return parser.parse_args()

# ------------------------------------------------------------------
# RESULTS FILE HANDLING
# ------------------------------------------------------------------

def load_results(results_file: Path):
    if results_file.exists():
        with results_file.open() as f:
            return json.load(f)
    return []


def append_result(results_file: Path, result):
    data = load_results(results_file)
    data.append(result)
    with results_file.open("w") as f:
        json.dump(data, f, indent=2)

# ------------------------------------------------------------------
# TOOLS
# ------------------------------------------------------------------

MAX_LOOKUPS = 6  # search_filing + find_in_filing calls per question, across all agents

LIMIT_REACHED = (
    "Lookup limit reached. Do not search again: answer with the information you "
    "already have, and list anything still missing."
)


@dataclass
class QuestionContext:
    doc_name: str
    retriever: HybridRetriever
    search_top_k: int
    lookups: int = 0


def _take_lookup(c: QuestionContext) -> bool:
    """Count a lookup; False once the per-question budget is spent."""
    c.lookups += 1
    return c.lookups <= MAX_LOOKUPS


@function_tool
def search_filing(ctx: RunContextWrapper[QuestionContext], query: str) -> str:
    """Search the company filing by meaning. Best for topics and explanations,
    e.g. "drivers of operating margin change" or "restructuring liability"."""
    c = ctx.context
    if not _take_lookup(c):
        return LIMIT_REACHED
    print(f"   🔍 search_filing({query!r})", flush=True)
    return c.retriever.search_text(query, c.doc_name, top_k=c.search_top_k)


@function_tool
def find_in_filing(ctx: RunContextWrapper[QuestionContext], phrase: str) -> str:
    """Find the exact words in the company filing, like Ctrl+F. Best for financial
    statement line items: pass just the line item name, e.g. "Total current liabilities"
    or "Purchases of property and equipment" (no years, no quotes)."""
    c = ctx.context
    if not _take_lookup(c):
        return LIMIT_REACHED
    print(f"   🔎 find_in_filing({phrase!r})", flush=True)
    nodes = c.retriever.find_phrase(phrase, c.doc_name)
    if not nodes:
        return f'"{phrase}" does not appear in the filing. Try a shorter or alternative line item name.'
    return "\n\n--- CHUNK SEPARATOR ---\n\n".join(n.get_content() for n in nodes)


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")

@function_tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "365 * (1200 + 1350) / 2 / 9800".
    Supports numbers, + - * / ** and parentheses."""
    try:
        return str(_eval_node(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"error: {e}"

# ------------------------------------------------------------------
# AGENTS
# ------------------------------------------------------------------

class Route(BaseModel):
    kind: Literal["calculation", "text"]


router = Agent(
    name="Router",
    instructions="""
Classify a question about a company's SEC filing.

- "calculation": the answer is a number or ratio taken from, or computed from,
  financial statement line items (e.g. "What is FY2018 capex?", "What is the
  FY2019 quick ratio?", "3-year average capex as % of revenue").
- "text": anything else, including yes/no judgements, explanations, lists,
  descriptions of segments, risks, debt securities, or drivers of change.
""",
    output_type=Route,
    model=fast_model,
)


class Fact(BaseModel):
    metric: str
    value: str
    period: str
    unit: str
    source_line: str


class Facts(BaseModel):
    facts: list[Fact]
    missing: list[str]


extractor = Agent[QuestionContext](
    name="Extractor",
    instructions="""
You are a STRICT information extraction engine for SEC filings.

TASK:
1. Decide which line items are needed to answer the question.
2. Extract them from the context. Tables are written one row per line, with
   columns separated by " | ". The column order (years / periods) is given in
   the table's header rows above.
3. If a needed line item is not in the context, look it up:
   - `find_in_filing` with just the line item name (e.g. "Total current
     liabilities") finds the table row word for word. Prefer this.
   - `search_filing` searches by meaning, for items without a fixed name.
   You have at most 6 lookups in total. If an item is still missing after
   2 attempts, stop looking for it and list it in `missing`.

HARD CONSTRAINTS:
- Do NOT infer, calculate, convert, or rephrase values.
- `value` is the verbatim number from the filing.
- `period` is the fiscal year or date of the column the value comes from.
  Double-check it against the table header: picking the wrong column is the
  most common mistake.
- `unit` is the table's stated unit (e.g. "USD millions"), or "" if none.
- `source_line` is the verbatim row the value came from.
- List needed items you still could not find in `missing`.
""",
    tools=[find_in_filing, search_filing],
    output_type=Facts,
    model=strong_model,
)


calculator = Agent(
    name="Calculator",
    instructions="""
You are a deterministic financial calculator.

INPUTS:
- A JSON object called `facts`
- A user question

TASK:
- Compute ONLY the metrics explicitly REQUIRED to answer the question.
- Use ONLY values present in `facts`.
- Use the `calculate` tool for every arithmetic operation.
- Follow any definition and rounding given in the question.
- If required inputs are missing, return:
  {"error": "insufficient data"}

RULES:
- Do NOT explain your reasoning.
- Do NOT compute extra metrics.
- Do NOT guess missing values.
- All outputs must be valid JSON.

OUTPUT FORMAT:
{
  "computed_metrics": {
    "<metric_name>": <numeric_value>
  }
}
""",
    tools=[calculate],
    model=fast_model,
)

composer = Agent(
    name="Composer",
    instructions="""
You are a final answer formatter.

TASK:
- Answer the question using ONLY the provided facts and computed_metrics.
- Do NOT introduce new numbers, assumptions, or explanations.
- Use the units and rounding the question asks for.

STYLE:
- Short (1–3 sentences)
- Declarative
- No calculations shown

Do NOT output JSON.
""",
    tools=[],
    model=fast_model,
)

analyst = Agent[QuestionContext](
    name="Analyst",
    instructions="""
You are a financial analyst answering a question about one company's SEC filing.

TASK:
- Answer using the provided context. Tables are written one row per line, with
  columns separated by " | "; the column periods are in the header rows above.
- If the context is not enough, look things up before concluding anything is
  missing: `find_in_filing` with a line item name (e.g. "Total current
  liabilities") finds table rows word for word; `search_filing` searches by
  meaning for topics and explanations. You have at most 6 lookups in total.
- Use `calculate` for any arithmetic.
- For judgement questions (e.g. "is X capital-intensive?"), give a clear
  answer and back it with the relevant figures or ratios from the filing.
- Only say the question cannot be answered if you searched and the filing
  truly lacks the information.

STYLE:
- Concise: 1–4 sentences, or a short list if the question asks for one.
- Use the units and rounding the question asks for.
""",
    tools=[find_in_filing, search_filing, calculate],
    model=strong_model,
)

# ------------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------------

async def run_agent(agent, messages, context=None, attempts=3):
    """Runner.run with retries and exponential backoff for transient API errors."""
    for attempt in range(1, attempts + 1):
        try:
            return await Runner.run(agent, messages, context=context, max_turns=12)
        except Exception as e:
            # Running out of turns would just happen again, at the same token cost
            if attempt == attempts or isinstance(e, MaxTurnsExceeded):
                raise
            wait = 5 * 2 ** (attempt - 1)
            print(f"   ⚠️ {agent.name} failed ({e}); retrying in {wait}s", flush=True)
            await asyncio.sleep(wait)


def user_message(content: str):
    return [{"role": "user", "content": content}]


async def answer_text(question: str, context: str, qctx: QuestionContext):
    res = await run_agent(
        analyst,
        user_message(f"Question:\n{question}\n\nContext:\n{context}"),
        context=qctx,
    )
    return res.final_output


async def answer_question(question: str, context: str, qctx: QuestionContext):
    # STEP 0: Route
    route_res = await run_agent(router, user_message(question))
    route = route_res.final_output.kind

    if route == "text":
        return route, await answer_text(question, context, qctx)

    # STEP 1: Extract (may search the filing again for missing items)
    try:
        extract_res = await run_agent(
            extractor,
            user_message(f"Question:\n{question}\n\nContext:\n{context}\n\nExtract the facts needed."),
            context=qctx,
        )
    except MaxTurnsExceeded:
        print("   ↪️ Extractor ran out of turns; falling back to Analyst", flush=True)
        return "calculation->text", await answer_text(question, context, qctx)
    facts_json = extract_res.final_output.model_dump_json(indent=2)

    # STEP 2: Calculate
    calc_res = await run_agent(
        calculator,
        user_message(f"Question:\n{question}\n\nExtracted facts (JSON):\n{facts_json}"),
    )
    computed_json = calc_res.final_output

    # Fallback: let the analyst work from the filing directly
    if "insufficient data" in computed_json:
        print("   ↪️ Calculator lacked data; falling back to Analyst", flush=True)
        return "calculation->text", await answer_text(question, context, qctx)

    # STEP 3: Compose
    final_res = await run_agent(
        composer,
        user_message(f"Question:\n{question}\n\nFacts:\n{facts_json}\n\nComputed metrics:\n{computed_json}"),
    )
    return route, final_res.final_output

# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------

async def run_batch(rows, retriever, results_file, args):

    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_one(i, row):
        async with semaphore:
            question = row["question"]
            doc_name = row["doc_name"]

            print(
                f"🔵 Running Question {i} | "
                f"id={row['financebench_id']} | {doc_name}",
                flush=True
            )

            qctx = QuestionContext(doc_name=doc_name, retriever=retriever, search_top_k=args.search_top_k)
            ctx = retriever.search_text(question, doc_name, top_k=args.top_k)

            try:
                route, ans = await answer_question(question, ctx, qctx)
            except Exception as e:
                route, ans = None, f"ERROR: {e}"

            result = {
                "financebench_id": row["financebench_id"],
                "company": row["company"],
                "doc_name": doc_name,
                "question_type": row["question_type"],
                "route": route,
                "question": question,
                "answer": ans
            }

            append_result(results_file, result)  # ✅ SAVE IMMEDIATELY
            return result

    tasks = [
        run_one(i, row)
        for i, row in enumerate(rows)
    ]

    return await asyncio.gather(*tasks)


async def main():
    args = parse_args()

    run_dir = PROJECT_DIR / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file = run_dir / "results.json"
    print(f"📁 Writing results to: {results_file}", flush=True)

    rows = load_dataset("Liadmagen/financebench_QA")["train"]
    if args.limit:
        rows = rows.shuffle(seed=0).select(range(args.limit))

    # Errored answers are retried on the next run
    completed_ids = {
        r["financebench_id"] for r in load_results(results_file)
        if not r["answer"].startswith("ERROR")
    }
    print(f"✅ Found {len(completed_ids)} completed questions", flush=True)

    remaining_rows = [
        row for row in rows
        if row["financebench_id"] not in completed_ids
    ]

    print(
        f"🕒 Remaining questions to run: {len(remaining_rows)}",
        flush=True
    )

    if not remaining_rows:
        print("🎉 All questions already completed!", flush=True)
        return

    # Drop previous errored answers for the questions about to be rerun
    remaining_ids = {row["financebench_id"] for row in remaining_rows}
    kept = [r for r in load_results(results_file) if r["financebench_id"] not in remaining_ids]
    with results_file.open("w") as f:
        json.dump(kept, f, indent=2)

    retriever = HybridRetriever(hybrid=args.hybrid, rerank=args.rerank)

    # One MLflow run per runs/<run> folder; 03 logs the accuracy into the same run
    rag.setup_mlflow(MLFLOW_EXPERIMENT)
    mlflow_id_file = run_dir / "mlflow_run_id.txt"
    run_id = mlflow_id_file.read_text().strip() if mlflow_id_file.exists() else None

    with mlflow.start_run(run_id=run_id, run_name=args.run) as mlrun:
        mlflow_id_file.write_text(mlrun.info.run_id)
        if run_id is None:
            mlflow.log_params({
                "fast_model": llm.FAST_MODEL,
                "strong_model": llm.STRONG_MODEL,
                "vector_store": "chroma",
                "top_k": args.top_k,
                "search_top_k": args.search_top_k,
                "hybrid": args.hybrid,
                "rerank": args.rerank,
                "embed_model": rag.EMBED_MODEL,
                "rerank_model": rag.RERANK_MODEL if args.rerank else None,
                "chunk_size": rag.CHUNK_SIZE,
                "chunk_overlap": rag.CHUNK_OVERLAP,
                "limit": args.limit or len(rows),
                "concurrency": args.concurrency,
            })

        start = time.perf_counter()
        await run_batch(
            remaining_rows,
            retriever,
            results_file,
            args
        )
        elapsed = time.perf_counter() - start

        results = load_results(results_file)
        mlflow.log_metrics({
            "answered": len(results),
            "errors": sum(r["answer"].startswith("ERROR") for r in results),
            "fallbacks": sum(r.get("route") == "calculation->text" for r in results),
            "seconds_per_question": elapsed / len(remaining_rows),
        })
        mlflow.log_artifact(str(results_file))

    print("✅ Run finished", flush=True)


if __name__ == "__main__":

    asyncio.run(main())
