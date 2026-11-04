"""The MCP tools, called through an in-process MCP client, over a fake two-filing index."""
import asyncio
from types import SimpleNamespace

import pytest
from mcp import Client

import mcp_server

FILINGS = {
    "3M_2018_10K": {"doc_name": "3M_2018_10K", "company": "3M", "doc_type": "10k", "doc_period": 2018},
    "PAYPAL_2022_10K": {"doc_name": "PAYPAL_2022_10K", "company": "PayPal", "doc_type": "10k", "doc_period": 2022},
}


@pytest.fixture
def fake_index(monkeypatch, make_retriever):
    retriever = make_retriever(["Total current liabilities | 45,101 | 43,029"], doc_name="PAYPAL_2022_10K")
    retriever.search_text = lambda query, doc_name, top_k: f"{top_k} chunks from {doc_name} about {query}"
    monkeypatch.setattr(mcp_server, "retriever", lambda: retriever)
    monkeypatch.setattr(mcp_server, "filings", lambda: FILINGS)
    return retriever


def call(tool, **arguments):
    async def go():
        async with Client(mcp_server.server) as client:
            return await client.call_tool(tool, arguments)
    result = asyncio.run(go())
    return result.is_error, result.content[0].text


def test_tools_are_registered():
    async def go():
        async with Client(mcp_server.server) as client:
            return await client.list_tools()
    names = {t.name for t in asyncio.run(go()).tools}
    assert names == {"list_filings", "search_filing", "find_in_filing", "answer_question"}


def test_list_filings(fake_index):
    assert call("list_filings") == (False, "3M_2018_10K (3M, 10k, 2018)\nPAYPAL_2022_10K (PayPal, 10k, 2022)")
    assert call("list_filings", company="paypal") == (False, "PAYPAL_2022_10K (PayPal, 10k, 2022)")
    assert call("list_filings", company="Apple")[1] == "No filings for a company matching 'Apple'."


def test_find_in_filing(fake_index):
    is_error, text = call("find_in_filing", doc_name="PAYPAL_2022_10K", phrase="total current liabilities")
    assert not is_error and "45,101" in text
    is_error, text = call("find_in_filing", doc_name="PAYPAL_2022_10K", phrase="goodwill")
    assert not is_error and "does not appear" in text


def test_search_filing(fake_index):
    assert call("search_filing", doc_name="3M_2018_10K", query="capex") == (False, "6 chunks from 3M_2018_10K about capex")


def test_unknown_filing_is_a_tool_error(fake_index):
    is_error, text = call("search_filing", doc_name="APPLE_2020_10K", query="capex")
    assert is_error and "list_filings" in text


def test_answer_question_runs_the_pipeline(fake_index, monkeypatch):
    async def answer_question(question, context, qctx):
        assert context == "80 chunks from 3M_2018_10K about What is FY2018 capex?"
        assert qctx.doc_name == "3M_2018_10K" and qctx.search_top_k == 6
        return "calculation", "Capex was $1,577 million."

    pipeline = SimpleNamespace(QuestionContext=SimpleNamespace, answer_question=answer_question)
    monkeypatch.setattr(mcp_server, "pipeline", lambda: pipeline)
    is_error, text = call("answer_question", doc_name="3M_2018_10K", question="What is FY2018 capex?")
    assert (is_error, text) == (False, "Capex was $1,577 million.\n\n(route: calculation)")
