# ===========================================================================
# DocumentSummaryIndex for Retrieval
# Date: 31, Jan 2025
# Writer: Ted, Jung
# Description: 
#    1. Query based retrieval
#    2. LLM-based Retrieval
#    3. Embedding-based Retrieval
# ===========================================================================

import logging
import sys
import os
import openai
import nest_asyncio
import requests

from llama_index.core import (
    SimpleDirectoryReader, 
    get_response_synthesizer,
    DocumentSummaryIndex,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# use retriever as part of a query engine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
)

from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexEmbeddingRetriever,
)

from pathlib import Path
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext

current_path = os.getcwd()

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# # Uncomment if you want to temporarily disable logger
# logger = logging.getLogger()
# logger.disabled = True


# enable nested event loop
# It schedules and runs these coroutines concurrently, 
# allowing your program to handle multiple tasks efficiently without blocking.
nest_asyncio.apply()


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


# List to look up
wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]


for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)



# Load all wiki documents
city_docs = []
for wiki_title in wiki_titles:
    docs = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()
    docs[0].doc_id = wiki_title
    city_docs.extend(docs)



# Build DocumentSummaryIndex
# LLM (gpt-4o-mini)
# chatgpt = OpenAI(temperature=0, model="gpt-4o-mini")
chatgpt = OpenAI(temperature=0, model="llama3.2", request_timeout=720.0)
splitter = SentenceSplitter(chunk_size=1024)

# default mode of building the index
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)
doc_summary_index = DocumentSummaryIndex.from_documents(
    city_docs,
    llm=chatgpt,
    transformations=[splitter],
    response_synthesizer=response_synthesizer,
    show_progress=True,
)


doc_summary_index.get_document_summary("Boston")
doc_summary_index.storage_context.persist(f"{current_path}/src_retrieval/index")



# Rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir=f"{current_path}/src_retrieval/index")
doc_summary_index = load_index_from_storage(storage_context)



# Perform Retrieval from Document Summary Index
# 1. High-level Querying
query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)

response = query_engine.query("What are the sports teams in Toronto?")
print(response)



# 2. LLM-based Retrieval


retriever = DocumentSummaryIndexLLMRetriever(
    doc_summary_index,
    # choice_select_prompt=None,
    # choice_batch_size=10,
    # choice_top_k=1,
    # format_node_batch_fn=None,
    # parse_choice_select_answer_fn=None,
)

retrieved_nodes = retriever.retrieve("What are the sports teams in Toronto?")
print(100*"=")
print(len(retrieved_nodes))

print(retrieved_nodes[0].score)
print(retrieved_nodes[0].node.get_text())



# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query("What are the sports teams in Toronto?")
print(100*"=")
print(response)



# 3. Embedding-based Retrival

retriever = DocumentSummaryIndexEmbeddingRetriever(
    doc_summary_index,
    # similarity_top_k=1,
)
retrieved_nodes = retriever.retrieve("What are the sports teams in Toronto?")

# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query("What are the sports teams in Toronto?")
print(response)