"""Shared fixtures. The tests make no network calls: no OpenAI, Chroma or Hugging Face."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
from llama_index.core.schema import TextNode

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# llm.py builds an OpenAI client at import time, which needs a key (never used)
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import rag  # noqa: E402


def load_script(filename: str):
    """Import a numbered script like 02_run_agents.py, whose name isn't a valid module name."""
    name = "script_" + Path(filename).stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(scope="session")
def agents_script():
    return load_script("02_run_agents.py")


@pytest.fixture(scope="session")
def eval_script():
    return load_script("03_evaluate_all_results.py")


@pytest.fixture(scope="session")
def retrieval_script():
    return load_script("05_evaluate_retrieval.py")


@pytest.fixture
def make_retriever():
    """A Retriever whose chunks come from a list instead of Chroma."""
    def make(texts, doc_name="ACME_2023_10K"):
        retriever = object.__new__(rag.Retriever)
        retriever.filter_key = "doc_name"
        retriever._nodes = {
            doc_name: [TextNode(id_=str(i), text=t, metadata={"doc_name": doc_name}) for i, t in enumerate(texts)]
        }
        return retriever
    return make
