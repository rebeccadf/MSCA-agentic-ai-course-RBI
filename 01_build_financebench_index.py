from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv
import os
from datasets import load_dataset


load_dotenv()

llm_model_name = "gpt-5-mini"
llm_deploy_name = "gpt-5-mini"

aoai_api_key = os.getenv("OPENAI_API_KEY")
aoai_endpoint = "https://aa-dsa-training-msca.openai.azure.com/"
aoai_api_version = "2024-12-01-preview"

llm = AzureOpenAI(
    engine=llm_deploy_name,
    api_key=aoai_api_key,
    azure_endpoint=aoai_endpoint,
    api_version=aoai_api_version,
    temperature=1,
)

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
Settings.node_parser = SimpleNodeParser.from_defaults(
    chunk_size=1024,
    chunk_overlap=128
)
financebench_content_ds = load_dataset("Liadmagen/financebench_content")
ds_small = financebench_content_ds["train"]

documents = [
    Document(
        text=row["text"],  
        metadata={
            "doc_name": row["doc_name"],
            "company": row["company"]
        }
    )
    for row in ds_small
]

index = VectorStoreIndex.from_documents(documents)


index.storage_context.persist(persist_dir="./financebench_index")
print("Index saved!")