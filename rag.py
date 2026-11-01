"""Shared indexing and retrieval code for the FinanceBench pipeline.

Chunks live in a Chroma collection, stored in ./chroma_db, or on a Chroma server
when CHROMA_HOST is set (e.g. in docker-compose).
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import textwrap
from collections import defaultdict
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
CHROMA_COLLECTION = "financebench"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128


def use_embed_model():
    Settings.embed_model = HuggingFaceEmbedding(EMBED_MODEL)


def clean_filing(text: str) -> str:
    """Turn the layout-preserving `text_formatted` field into compact text.

    Runs of 3+ spaces separate table columns, so they become " | ". This keeps
    every table row (line item + values for each period) on a single line.
    """
    lines = (re.sub(r" {3,}", " | ", line.strip()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def setup_mlflow(experiment: str):
    """Track in ./mlflow.db (or MLFLOW_TRACKING_URI), whatever folder the script runs from."""
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_DIR / 'mlflow.db'}"))
    if mlflow.get_experiment_by_name(experiment) is None:
        artifact_root = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", PROJECT_DIR / "mlartifacts"))
        mlflow.create_experiment(experiment, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(experiment)

# ------------------------------------------------------------------
# VECTOR STORE
# ------------------------------------------------------------------

def chroma_collection(reset: bool = False):
    import chromadb

    host = os.getenv("CHROMA_HOST")
    if host:
        client = chromadb.HttpClient(host=host, port=int(os.getenv("CHROMA_PORT", "8000")))
    else:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset and CHROMA_COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(CHROMA_COLLECTION)
    return client.get_or_create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})


def build_index(documents):
    from llama_index.vector_stores.chroma import ChromaVectorStore

    use_embed_model()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection(reset=True))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

# ------------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------------

class HybridRetriever:
    """Dense (+ optional BM25) retrieval restricted to one filing (or company), fused and reranked.

    filter_key: metadata field to restrict on, "doc_name" or "company"
    hybrid:     add BM25 keyword retrieval (off by default: it lowered evidence
                recall in 05_evaluate_retrieval.py)
    rerank:     rerank with a cross-encoder (off by default: not yet measured)
    """

    def __init__(self, filter_key: str = "doc_name",
                 hybrid: bool = False, rerank: bool = False, candidates: int = 30):
        from llama_index.vector_stores.chroma import ChromaVectorStore

        use_embed_model()
        self.filter_key = filter_key
        self.hybrid = hybrid
        self.candidates = candidates
        self.reranker = None
        if rerank:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL)

        self.collection = chroma_collection()
        self.index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=self.collection))

        self._nodes = {}
        self._bm25 = {}

    def nodes_for(self, value: str):
        """All chunks of one filing/company, for BM25 and phrase search."""
        if value not in self._nodes:
            got = self.collection.get(where={self.filter_key: value}, include=["documents", "metadatas"])
            self._nodes[value] = [
                TextNode(id_=i, text=t, metadata=m)
                for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])
            ]
        return self._nodes[value]

    def _bm25_for(self, value: str):
        from rank_bm25 import BM25Okapi

        if value not in self._bm25:
            self._bm25[value] = BM25Okapi([tokenize(n.get_content()) for n in self.nodes_for(value)])
        return self._bm25[value]

    def search(self, query: str, value: str, top_k: int = 8):
        # Fusion and reranking pick from a wider pool; never fewer than top_k
        candidates = max(self.candidates, top_k) if (self.hybrid or self.reranker) else top_k

        # 1. Dense retrieval, restricted to the filing (or company)
        retriever = self.index.as_retriever(
            similarity_top_k=candidates,
            filters=MetadataFilters(filters=[ExactMatchFilter(key=self.filter_key, value=value)]),
        )
        dense = [n.node for n in retriever.retrieve(query)]
        rankings = [dense]

        # 2. Keyword retrieval, which handles exact line-item names better
        if self.hybrid:
            nodes = self.nodes_for(value)
            if nodes:
                scores = self._bm25_for(value).get_scores(tokenize(query))
                ranked = sorted(range(len(nodes)), key=lambda i: scores[i], reverse=True)
                rankings.append([nodes[i] for i in ranked[:candidates]])

        # 3. Reciprocal rank fusion
        fused = defaultdict(float)
        by_id = {}
        for ranking in rankings:
            for rank, node in enumerate(ranking):
                fused[node.node_id] += 1 / (60 + rank)
                by_id[node.node_id] = node
        pool = sorted(by_id.values(), key=lambda n: fused[n.node_id], reverse=True)[:candidates]

        # 4. Cross-encoder rerank
        if self.reranker is None or not pool:
            return pool[:top_k]
        scores = self.reranker.predict([(query, n.get_content()) for n in pool])
        return [n for n, _ in sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[:top_k]]

    def find_phrase(self, phrase: str, value: str, max_chunks: int = 3):
        """Chunks containing `phrase` word for word (like Ctrl+F), ignoring case and
        punctuation, most occurrences first. Finds table rows that ranked search misses."""
        needle = " " + " ".join(tokenize(phrase)) + " "
        if needle.strip() == "":
            return []
        matches = []
        for node in self.nodes_for(value):
            count = (" " + " ".join(tokenize(node.get_content())) + " ").count(needle)
            if count:
                matches.append((count, node))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in matches[:max_chunks]]

    def search_text(self, query: str, value: str, top_k: int = 8) -> str:
        return "\n\n--- CHUNK SEPARATOR ---\n\n".join(
            textwrap.dedent(n.get_content()) for n in self.search(query, value, top_k)
        )
