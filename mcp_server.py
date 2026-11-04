"""MCP server for the FinanceBench filings.

Lets an MCP client (Claude Code, Claude Desktop, ...) list the indexed filings,
search them, and ask the multi-agent pipeline of 02_run_agents.py.

Needs the Chroma index built by 01_build_financebench_index.py. The lookup tools
run locally; only answer_question calls the OpenAI API (OPENAI_API_KEY in .env).

    claude mcp add financebench -- python /path/to/mcp_server.py
"""
import importlib.util
from functools import cache
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import rag

PROJECT_DIR = Path(__file__).parent
TOP_K = 80         # chunks the pipeline gets up front, as in 02_run_agents.py
SEARCH_TOP_K = 6   # chunks per search_filing call, as in 02_run_agents.py

server = MCPServer(
    "financebench",
    instructions=(
        "Search SEC filings (10-K, 10-Q, 8-K, earnings reports) from the FinanceBench benchmark. "
        "Call list_filings first to get the doc_name of the filing you need, e.g. 3M_2018_10K."
    ),
)


@cache
def retriever():
    return rag.Retriever()


@cache
def pipeline():
    """02_run_agents.py, imported on first use: it needs an OpenAI key, the lookup tools don't."""
    spec = importlib.util.spec_from_file_location("run_agents", PROJECT_DIR / "02_run_agents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def filings():
    """doc_name -> metadata of every indexed filing."""
    got = retriever().collection.get(include=["metadatas"])
    return {m["doc_name"]: m for m in got["metadatas"]}


def check_filing(doc_name: str):
    if doc_name not in filings():
        raise ToolError(f"Unknown filing {doc_name!r}. Call list_filings for the valid doc_name values.")


@server.tool()
def list_filings(company: str = "") -> str:
    """List the indexed filings as `doc_name (company, type, period)`, one per line.
    Optionally keep only companies whose name contains `company` (case-insensitive)."""
    rows = [
        f"{name} ({m['company']}, {m['doc_type']}, {m['doc_period']})"
        for name, m in sorted(filings().items())
        if company.lower() in m["company"].lower()
    ]
    return "\n".join(rows) or f"No filings for a company matching {company!r}."


@server.tool()
def search_filing(doc_name: str, query: str, top_k: int = SEARCH_TOP_K) -> str:
    """Search one filing by meaning. Best for topics and explanations,
    e.g. "drivers of operating margin change" or "restructuring liability"."""
    check_filing(doc_name)
    return retriever().search_text(query, doc_name, top_k=top_k)


@server.tool()
def find_in_filing(doc_name: str, phrase: str) -> str:
    """Find the exact words in one filing, like Ctrl+F. Best for financial statement
    line items: pass just the line item name, e.g. "Total current liabilities"."""
    check_filing(doc_name)
    nodes = retriever().find_phrase(phrase, doc_name)
    if not nodes:
        return f'"{phrase}" does not appear in the filing. Try a shorter or alternative line item name.'
    return "\n\n--- CHUNK SEPARATOR ---\n\n".join(n.get_content() for n in nodes)


@server.tool()
async def answer_question(doc_name: str, question: str) -> str:
    """Answer a question about one filing with the project's multi-agent pipeline
    (router, then extractor/calculator/composer or analyst). Calls the OpenAI API."""
    check_filing(doc_name)
    p = pipeline()
    context = retriever().search_text(question, doc_name, top_k=TOP_K)
    qctx = p.QuestionContext(doc_name=doc_name, retriever=retriever(), search_top_k=SEARCH_TOP_K)
    route, answer = await p.answer_question(question, context, qctx)
    return f"{answer}\n\n(route: {route})"


if __name__ == "__main__":
    server.run()
