# Agentic AI on FinanceBench

A multi-agent retrieval-augmented pipeline that answers financial questions from the [FinanceBench](https://huggingface.co/datasets/Liadmagen/financebench_QA) benchmark, plus an LLM-as-judge evaluation of the answers. Experiments are tracked with MLflow, and the pipeline can run in Docker with Chroma as a separate service.

The original assignment is in `final_ir_project.ipynb`.

## Results

| Run | Questions | Accuracy | Domain-relevant | Metrics-generated | Novel-generated |
|-----|-----------|----------|-----------------|-------------------|-----------------|
| `full` | all 150 (`--top-k 80`) | **85.3%** (128/150) | 86.0% | 90.0% | 80.0% |
| `quick2` | 30 (fixed sample, `--top-k 40`) | 83.3% (25/30) | 70.0% | 87.5% | 91.7% |

Answers from `gpt-5-nano` / `gpt-5-mini`, graded by `gpt-5-mini`. The full run had 0 errors and 0 refusals; 8 calculation questions fell back to the Analyst.

The original pipeline scored 35.33% on all 150 questions. Most of its wrong answers were refusals ("cannot be answered") caused by retrieval, so this version focuses on retrieval. Its code and results are in git history (commit `b451235`).

### Retrieval quality (measured offline with `05_evaluate_retrieval.py`)

The hit rate is the share of the 150 questions where the retrieved chunks cover at least 25% of the labelled evidence page.

| Setup | Context | Hit rate |
|-------|---------|----------|
| Filter by filing, 80 chunks (**current default**) | ~40k tokens | 89% |
| Same + BM25 hybrid | ~40k tokens | 87% |
| Filter by filing, 40 chunks | ~20k tokens | 83% |

Filtering by filing, rather than by company as the original pipeline did, is the main improvement: on the original index it raised the hit rate from 69% to 93%. BM25 lowered recall, so it is off by default.

Ranked search, by meaning or by keywords, often misses table rows: for PayPal, the chunk with `Total current liabilities | 45,101 | 43,029` never reached the top results. That is why the agents also have an exact-phrase lookup, `find_in_filing`.

## Pipeline

```
FinanceBench filings ──► 01 build index ──► Chroma (chroma_db/ or Chroma server)
                                                  │
FinanceBench QA ──► 02 run agents ──► vector retrieval in the question's filing
                                      (optional: + BM25, cross-encoder rerank)
                                                  │
                                               Router
                                    ┌─────────────┴─────────────┐
                              calculation                      text
                     Extractor ─► Calculator ─► Composer      Analyst
                     (can look things up)  │ insufficient data ──►  ▲
                                                  │
                                       runs/<run>/results.json
                                                  │
                         03 evaluate (Judge agent vs. gold answers) ──► MLflow
                                                  │
                                       runs/<run>/eval.json
                                                  │
                          04 display accuracy (notebook, compares runs)
```

| Step | File | What it does |
|------|------|--------------|
| 1 | `01_build_financebench_index.py` | Embeds the 84 filings the questions use into Chroma. It uses the `text_formatted` field with whitespace collapsed, so each table row stays on one line (`Total current liabilities \| 7,118 \| 5,964`). Chunks are 512 tokens with 128 overlap. |
| 2 | `02_run_agents.py` | Answers every question (see [Agents](#agents)) and writes `runs/<run>/results.json`. Logs settings and timings to MLflow. |
| 3 | `03_evaluate_all_results.py` | A **Judge** agent marks each answer `1` (correct) or `0` (incorrect) against the gold answer, writes `runs/<run>/eval.json`, and logs overall and per-category accuracy to MLflow. |
| 4 | `04_display_accuracy.ipynb` | Accuracy, refusals and errors per run, accuracy per question type, and a comparison on the questions every run shares. |
| 5 | `05_evaluate_retrieval.py` | Optional. Scores retrieval alone against FinanceBench's labelled evidence pages, with **no LLM calls**, and logs to MLflow. Useful for tuning retrieval cheaply. |

Shared indexing and retrieval code is in `rag.py`; the LLM client is in `llm.py`.

### Retrieval

For each question, `rag.HybridRetriever`:

1. **Filters to the filing the question is about** (`doc_name`, e.g. `3M_2018_10K`), rather than to all of a company's filings. This matches FinanceBench's single-document setting.
2. Retrieves the 80 most similar chunks with the BGE embedding model.
3. Optional `--hybrid`: also runs BM25 keyword retrieval and merges both lists with reciprocal rank fusion.
4. Optional `--rerank`: reorders the pool with a cross-encoder (`BAAI/bge-reranker-base`, a one-off 1.1 GB download). It has not been measured yet; test it with `05_evaluate_retrieval.py --rerank --top-k 40`.

### Agents

Built with the OpenAI Agents SDK on the OpenAI API (client setup is in `llm.py`).

| Agent | Model | Role |
|-------|-------|------|
| **Router** | fast | Classifies the question as a `calculation` or a `text` question. |
| **Extractor** | strong | Extracts the needed line items with their period, unit and source row. Can look up items missing from the context. |
| **Calculator** | fast | Computes the answer with the `calculate` tool (safe arithmetic only), or returns `insufficient data`. |
| **Composer** | fast | Writes the final 1–3 sentence answer. |
| **Analyst** | strong | Answers text questions from the filing, with the lookup tools and `calculate`. Also the fallback when the Calculator lacks data or the Extractor runs out of turns. |
| **Judge** (step 3) | strong | Grades answers. |

"Fast" is `OPENAI_MODEL` (default `gpt-5-nano`). "Strong" is `OPENAI_STRONG_MODEL` (default `gpt-5-mini`).

Lookup tools, shared by the Extractor and Analyst:
- **`find_in_filing(phrase)`:** exact-phrase search, like Ctrl+F. Returns the chunks containing a line item name such as `Total current liabilities`.
- **`search_filing(query)`:** search by meaning, for topics and explanations.

Each question gets at most 6 lookups in total, enforced in code; after that the tools answer "limit reached". Agent calls that fail with an API error are retried up to 3 times with backoff, but running out of turns is not retried. Answers that still error are retried the next time you run step 2.

## Setup

Requires Python 3.12.

```bash
cd agentic_ai_rbi_project
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your OPENAI_API_KEY
```

The first run downloads the embedding model from Hugging Face (about 130 MB). `--rerank` needs another 1.1 GB. Both are cached in `~/.cache/huggingface/` after the first download.

Docker is optional. You only need it to run the project on another machine; locally, Chroma runs inside Python and stores its data in `chroma_db/`.

## Usage

### Everything in one command

```bash
./run_pipeline.sh                         # builds the index if missing, then steps 2 and 3, as run "full"
./run_pipeline.sh --run quick --limit 30  # same 30 questions every time, for quick comparisons
```

Options passed to `run_pipeline.sh` go to `02_run_agents.py`:

| Option | Default | Meaning |
|--------|---------|---------|
| `--run` | `full` | Run name; output goes to `runs/<run>/` |
| `--limit` | all 150 | Fixed random sample of N questions |
| `--top-k` | 80 | Chunks given to the agents up front |
| `--search-top-k` | 6 | Chunks returned by each `search_filing` call |
| `--hybrid` | off | Add BM25 keyword retrieval |
| `--rerank` | off | Rerank with the cross-encoder |
| `--concurrency` | 4 | Questions answered in parallel |

Then open `04_display_accuracy.ipynb` to compare runs.

### Rate limits (tokens per minute)

With `--top-k 80`, one question uses roughly 50–150k tokens, because the Extractor and Analyst resend the context with each lookup. Match `--concurrency` to your tokens-per-minute (TPM) limit. It depends on your OpenAI usage tier and is listed at https://platform.openai.com/settings/organization/limits.

| TPM limit | `--concurrency` |
|----------------|-----------------|
| under 200k | 1 |
| 200k–500k | 2 |
| over 500k | 4 (default) |

When the API returns 429 (rate limit), the client waits for the time given in `Retry-After` and retries up to 10 times (`LLM_MAX_RETRIES` in `.env`). `--top-k 40` roughly halves token use.

### Step by step

```bash
python 01_build_financebench_index.py     # once
python 02_run_agents.py --run full
python 03_evaluate_all_results.py --run full
```

Steps 2 and 3 can be resumed. Rerunning with the same `--run` skips the questions that are already done.

### Comparing runs in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000:
- **`financebench-agents`:** one MLflow run per `runs/<run>` folder, with the settings, accuracy (overall and per question type), refusal fallbacks, errors and seconds per question.
- **`financebench-retrieval`:** results from `05_evaluate_retrieval.py`.

To compare retrieval setups without spending API credits:

```bash
python 05_evaluate_retrieval.py                                         # current default
python 05_evaluate_retrieval.py --top-k 40                              # fewer chunks
python 05_evaluate_retrieval.py --top-k 40 --rerank                     # does the reranker help?
python 05_evaluate_retrieval.py --filter company                        # whole company, as originally
```

### Docker

```bash
docker compose up -d chroma                          # start the Chroma server
docker compose run --rm index                        # build the index into it (once)
docker compose run --rm pipeline --run docker --limit 10
```

The `pipeline` service reads `.env`, writes results to `./runs` and tracking to `./mlflow-docker`. View that tracking data with `mlflow ui --backend-store-uri sqlite:///mlflow-docker/mlflow.db`. Without Docker, Chroma runs embedded in `./chroma_db` instead, so no server is needed.

## Notes

- **Evidence labels:** these are only used by `05_evaluate_retrieval.py` for scoring. The agents never see them.
