from datasets import load_dataset
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter

from rag import CHUNK_SIZE, CHUNK_OVERLAP, build_index, clean_filing


Settings.node_parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    paragraph_separator="\n",
)

content_ds = load_dataset("Liadmagen/financebench_content")["train"]
qa_ds = load_dataset("Liadmagen/financebench_QA")["train"]

# Only the filings that questions refer to are ever retrieved from
needed_docs = set(qa_ds["doc_name"])

documents = [
    Document(
        text=clean_filing(row["text_formatted"]),
        metadata={
            "doc_name": row["doc_name"],
            "company": row["company"],
            "doc_type": row["doc_type"],
            "doc_period": row["doc_period"],
        }
    )
    for row in content_ds
    if row["doc_name"] in needed_docs
]
print(f"Indexing {len(documents)} filings into Chroma...", flush=True)

build_index(documents)
print("Index saved!")
