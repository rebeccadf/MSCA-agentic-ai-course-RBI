import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from openai import AsyncAzureOpenAI

from agents import (
    Agent,
    Runner,
    set_tracing_disabled,
    OpenAIChatCompletionsModel
)

# ============================================================
# CONFIG
# ============================================================

RESULTS_PREFIX = "financebench_results"
EVAL_PREFIX = "financebench_eval"
RESULTS_DIR = Path(".")
CONCURRENCY = 2

# ============================================================
# FILE HELPERS
# ============================================================

def get_next_eval_file():
    existing = list(RESULTS_DIR.glob(f"{EVAL_PREFIX}*.json"))
    if not existing:
        return RESULTS_DIR / f"{EVAL_PREFIX}.json"

    indices = []
    for f in existing:
        m = re.match(rf"{EVAL_PREFIX}_(\d+)\.json", f.name)
        if m:
            indices.append(int(m.group(1)))
        elif f.name == f"{EVAL_PREFIX}.json":
            indices.append(0)

    return RESULTS_DIR / f"{EVAL_PREFIX}_{max(indices)+1}.json"


def load_all_model_answers():
    answers = []
    for f in RESULTS_DIR.glob(f"{RESULTS_PREFIX}*.json"):
        try:
            with f.open() as fh:
                answers.extend(json.load(fh))
        except Exception:
            pass
    return answers


def load_completed_eval_ids():
    completed = set()
    for f in RESULTS_DIR.glob(f"{EVAL_PREFIX}*.json"):
        try:
            with f.open() as fh:
                for r in json.load(fh):
                    completed.add(r["financebench_id"])
        except Exception:
            pass
    return completed


# ============================================================
# ENV + CLIENT
# ============================================================

load_dotenv()
set_tracing_disabled(True)

openai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

model = OpenAIChatCompletionsModel(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_client=openai_client,
)

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

async def judge_answer(question, gold_answer, model_answer):
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

    res = await Runner.run(judge, messages)
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
    print("🚀 Evaluation started", flush=True)

    eval_file = get_next_eval_file()
    print(f"📁 Writing evaluation to: {eval_file}", flush=True)

    # Load datasets
    fb = load_dataset("Liadmagen/financebench_QA")["train"]
    gold_map = {r["financebench_id"]: r for r in fb}

    model_answers = load_all_model_answers()
    completed_ids = load_completed_eval_ids()

    print(f"📊 Model answers found: {len(model_answers)}", flush=True)
    print(f"✅ Already judged: {len(completed_ids)}", flush=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = []

    async def run_one(item):
        async with semaphore:
            fid = item["financebench_id"]

            if fid in completed_ids or fid not in gold_map:
                return

            gold = gold_map[fid]

            print(f"🔎 Judging id={fid}", flush=True)

            correct = await judge_answer(
                question=gold["question"],
                gold_answer=gold["answer"],
                model_answer=item["answer"],
            )

            result = {
                "financebench_id": fid,
                "question": gold["question"],
                "gold_answer": gold["answer"],
                "llm_answer": item["answer"],
                "correct": correct,
            }

            # SAVE IMMEDIATELY
            if eval_file.exists():
                with eval_file.open() as f:
                    data = json.load(f)
            else:
                data = []

            data.append(result)

            with eval_file.open("w") as f:
                json.dump(data, f, indent=2)

            return result

    tasks = [run_one(item) for item in model_answers]
    await asyncio.gather(*tasks)

    print("✅ Evaluation finished", flush=True)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
