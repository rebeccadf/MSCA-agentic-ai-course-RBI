import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
from pathlib import Path

import mlflow
from datasets import load_dataset

from agents import Agent, Runner

from llm import strong_model
from rag import setup_mlflow

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).parent
CONCURRENCY = 2

# ============================================================
# FILE HELPERS
# ============================================================

def load_json(path: Path):
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return []


# ============================================================
# MODEL
# ============================================================

# The judge uses the strong model
model = strong_model

# ============================================================
# JUDGE AGENT
# ============================================================

judge = Agent(
    name="Judge",
    instructions="""
You are grading answers for the FinanceBench benchmark.

Decide whether the MODEL ANSWER should be considered CORRECT.

RULES:
- Wording does not need to match exactly.
- Numeric formatting differences are allowed.
- Units may be implicit if clear from context.
- If the gold answer says the question cannot be answered
  and the model also says this, the answer is CORRECT.
- Qualitative explanations are acceptable when appropriate.
- Mark incorrect ONLY if clearly wrong or missing required facts.

OUTPUT:
Return EXACTLY one character:
1 = correct
0 = incorrect
""",
    tools=[],
    model=model,
)

# ============================================================
# JUDGE CALL (same style as your other agents)
# ============================================================

async def judge_answer(question, gold_answer, model_answer, attempts=3):
    messages = [
        {
            "role": "user",
            "content": f"""
QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

MODEL ANSWER:
{model_answer}
"""
        }
    ]

    for attempt in range(1, attempts + 1):
        try:
            res = await Runner.run(judge, messages)
            break
        except Exception as e:
            if attempt == attempts:
                raise
            wait = 5 * 2 ** (attempt - 1)
            print(f"⚠️ Judge failed ({e}); retrying in {wait}s", flush=True)
            await asyncio.sleep(wait)

    out = res.final_output.strip()

    if out.startswith("1"):
        return 1
    if out.startswith("0"):
        return 0

    print("⚠️ Ambiguous judge output:", out, flush=True)
    return 0


# ============================================================
# MAIN PIPELINE
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Judge a run's answers against the gold answers.")
    parser.add_argument("--run", default="full", help="Run name under runs/ (default: full)")
    args = parser.parse_args()

    run_dir = PROJECT_DIR / "runs" / args.run
    results_file = run_dir / "results.json"
    eval_file = run_dir / "eval.json"

    print("🚀 Evaluation started", flush=True)
    print(f"📁 Writing evaluation to: {eval_file}", flush=True)

    # Load datasets
    fb = load_dataset("Liadmagen/financebench_QA")["train"]
    gold_map = {r["financebench_id"]: r for r in fb}

    model_answers = load_json(results_file)
    completed_ids = {r["financebench_id"] for r in load_json(eval_file)}

    print(f"📊 Model answers found: {len(model_answers)}", flush=True)
    print(f"✅ Already judged: {len(completed_ids)}", flush=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def run_one(item):
        async with semaphore:
            fid = item["financebench_id"]

            if fid in completed_ids or fid not in gold_map:
                return

            gold = gold_map[fid]

            print(f"🔎 Judging id={fid}", flush=True)

            try:
                correct = await judge_answer(
                    question=gold["question"],
                    gold_answer=gold["answer"],
                    model_answer=item["answer"],
                )
            except Exception as e:
                print(f"❌ Could not judge id={fid}: {e}", flush=True)
                return  # left unjudged, picked up on the next run

            result = {
                "financebench_id": fid,
                "question_type": gold["question_type"],
                "question": gold["question"],
                "gold_answer": gold["answer"],
                "llm_answer": item["answer"],
                "correct": correct,
            }

            # SAVE IMMEDIATELY
            data = load_json(eval_file)
            data.append(result)

            with eval_file.open("w") as f:
                json.dump(data, f, indent=2)

            return result

    tasks = [run_one(item) for item in model_answers]
    await asyncio.gather(*tasks)

    print("✅ Evaluation finished", flush=True)
    log_to_mlflow(run_dir, args.run, load_json(eval_file))


def log_to_mlflow(run_dir: Path, run_name: str, evals):
    """Log accuracy into the MLflow run that 02 created for this run folder."""
    if not evals:
        return

    setup_mlflow("financebench-agents")
    mlflow_id_file = run_dir / "mlflow_run_id.txt"
    run_id = mlflow_id_file.read_text().strip() if mlflow_id_file.exists() else None

    metrics = {
        "judged": len(evals),
        "accuracy": sum(r["correct"] for r in evals) / len(evals),
    }
    for qtype in sorted({r["question_type"] for r in evals if r.get("question_type")}):
        subset = [r for r in evals if r.get("question_type") == qtype]
        metrics[f"accuracy_{qtype}"] = sum(r["correct"] for r in subset) / len(subset)

    with mlflow.start_run(run_id=run_id, run_name=run_name) as mlrun:
        mlflow_id_file.write_text(mlrun.info.run_id)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(run_dir / "eval.json"))

    print(f"📈 Logged to MLflow: accuracy={metrics['accuracy']:.2%}", flush=True)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
