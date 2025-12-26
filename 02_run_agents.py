import os
import sqlite3
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from datasets import load_dataset
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from pathlib import Path
import json
import re
from agents import (
    Agent, 
    Runner, 
    function_tool, 
    set_tracing_disabled, 
    OpenAIChatCompletionsModel
)
from llama_index.core import StorageContext, load_index_from_storage
import textwrap

financebench_content_ds = load_dataset("Liadmagen/financebench_content")
financebench_qa_ds = load_dataset("Liadmagen/financebench_QA")

# ------------------------------------------------------------------
# RESULTS FILE HANDLING
# ------------------------------------------------------------------

RESULTS_PREFIX = "financebench_results"
RESULTS_DIR = Path(".")


def get_next_results_file():
    existing = list(RESULTS_DIR.glob(f"{RESULTS_PREFIX}*.json"))

    if not existing:
        return RESULTS_DIR / f"{RESULTS_PREFIX}.json"

    indices = []
    for f in existing:
        m = re.match(rf"{RESULTS_PREFIX}_(\d+)\.json", f.name)
        if m:
            indices.append(int(m.group(1)))
        elif f.name == f"{RESULTS_PREFIX}.json":
            indices.append(0)

    next_idx = max(indices) + 1
    return RESULTS_DIR / f"{RESULTS_PREFIX}_{next_idx}.json"



def load_all_completed_ids():
    completed = set()

    for f in RESULTS_DIR.glob(f"{RESULTS_PREFIX}*.json"):
        try:
            with f.open() as fh:
                data = json.load(fh)
                for r in data:
                    if isinstance(r, dict) and "financebench_id" in r:
                        completed.add(r["financebench_id"])
        except Exception:
            pass  # ignore corrupted / partial files safely

    return completed

RESULTS_FILE = get_next_results_file()

def append_result(result):
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            data = json.load(f)
    else:
        data = []

    data.append(result)

    with RESULTS_FILE.open("w") as f:
        json.dump(data, f, indent=2)

# ------------------------------------------------------------------
# ENV + INDEX
# ------------------------------------------------------------------
# Load environment variables from .env file
load_dotenv()

# Disable tracing to prevent non-fatal error messages about OpenAI API key
set_tracing_disabled(True)
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_BASE_URL"] = azure_endpoint
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load LlamaIndex from disk -------------------------
INDEX_DIR = "notebooks/day_3/financebench_index"
# --- Fix: use HF embedding
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)

# ------------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------------

def retrieve_context(question: str, company: str, top_k: int = 8) -> str:
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="company", value=company)
        ]
    )

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters
    )

    nodes = retriever.retrieve(question)

    context = "\n\n--- CHUNK SEPARATOR ---\n\n".join(
        textwrap.dedent(n.text) for n in nodes
    )
    return context

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------

openai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  # Note: Using subscription key
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),  # Default to a common version if not set
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

model=OpenAIChatCompletionsModel(
            model="gpt-5-nano", # This will use the deployment specified in your Azure OpenAI/APIM client
            openai_client=openai_client
        )

# ------------------------------------------------------------------
# AGENTS
# ------------------------------------------------------------------

@function_tool
def sql_tool(query: str) -> str:
    conn = sqlite3.connect("financials.db")
    cur = conn.cursor()
    try:
        cur.execute(query)
        rows = cur.fetchall()
        conn.commit()
        return str(rows)
    finally:
        conn.close()

extractor = Agent(
    name="Extractor",
    instructions="""
You are a STRICT information extraction engine.

TASK:
Extract ONLY numeric facts that are EXPLICITLY stated in the provided context.

HARD CONSTRAINTS (violations are errors):
- Do NOT infer, calculate, normalize, convert, or rephrase values.
- Do NOT add metrics that are not literally named in the text.
- Do NOT fix units, years, or formatting.
- If a value is ambiguous, SKIP it.
- If no facts exist, return {"facts": []}.

OUTPUT FORMAT (JSON ONLY, no markdown, no commentary):

{
  "facts": [
    {
      "metric": "<verbatim metric name>",
      "value": "<verbatim numeric string>",
      "year": "<verbatim year or null>",
      "unit": "<verbatim unit or null>"
    }
  ]
}
""",
    tools=[],
    model=model,
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
- Use ONLY values present in `facts` or returned by SQL queries.
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
    tools=[sql_tool],
    model=model,
)

composer = Agent(
    name="Composer",
    instructions="""
You are a final answer formatter.

TASK:
- Answer the question using ONLY the provided facts and computed_metrics.
- Do NOT introduce new numbers, assumptions, or explanations.
- If an error object is present, state that the question cannot be answered.

STYLE:
- Short (1–3 sentences)
- Declarative
- No calculations shown

Do NOT output JSON.
""",
    tools=[],
    model=model,
)

# ------------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------------

async def answer_question(question: str, context: str):
    # STEP 1: Extract
    extract_messages = [
        {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}\n\nExtract all facts as JSON."}
    ]
    extract_res = await Runner.run(extractor, extract_messages)
    facts_json = extract_res.final_output

    # STEP 2: Calculate
   
    calc_messages = [
    {"role": "user", "content": f"""
Question:
{question}

Extracted facts (JSON):
{facts_json}
"""}
]

    calc_res = await Runner.run(calculator, calc_messages)
    computed_json = calc_res.final_output

    # STEP 3: Compose
    compose_messages = [
    {"role": "user", "content": f"""
Question:
{question}

Facts:
{facts_json}

Computed metrics:
{computed_json}
"""}
]

    final_res = await Runner.run(composer, compose_messages)

    return final_res.final_output

# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------

async def run_batch(rows, concurrency=2):

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(i, row):
        async with semaphore:
            question = row["question"]
            company = row["company"]

            print(
                f"🔵 Running Question {i} | "
                f"id={row['financebench_id']} | {company}",
                flush=True
            )

            ctx = retrieve_context(
                question,
                company=company,
                top_k=40
            )

            try:
                ans = await answer_question(question, ctx)
            except Exception as e:
                ans = f"ERROR: {e}"

            result = {
                "financebench_id": row["financebench_id"],
                "company": company,
                "question": question,
                "answer": ans
            }

            append_result(result)  # ✅ SAVE IMMEDIATELY
            return result

    tasks = [
        run_one(i, row)
        for i, row in enumerate(rows)
    ]

    return await asyncio.gather(*tasks)
 
def get_completed_ids(results):
    return {
        r["financebench_id"]
        for r in results
        if isinstance(r, dict) and "financebench_id" in r
    }
def load_existing_results():
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            return json.load(f)
    return []


async def main():

    completed_ids = load_all_completed_ids()
    print(f"✅ Found {len(completed_ids)} completed questions", flush=True)

    rows = financebench_qa_ds["train"]

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

    await run_batch(
        remaining_rows,
        concurrency=4
    )

    print("✅ Run finished", flush=True)


if __name__ == "__main__":

    asyncio.run(main())
