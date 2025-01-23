# ===========================================================================
# HyDE with HyDETransform
# Date: 23, Jan 2025
# Writer: Ted, Jung
# Description: How to trasform a engine to HyDEEngine
# ===========================================================================

import os
import logging
import sys

from llama_index.core import (Settings)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=720.0)

# Load a file, Make documents and index it
current_dir = os.getcwd()

documents = SimpleDirectoryReader(f"{current_dir}/data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents)


# Turn index into queryengine, do a query with question

query_str = "what did paul graham do after going to RISD"
query_engine = index.as_query_engine()
response = query_engine.query(query_str)

print(str(response))



# Turn the queryengine into hydequeryengine
# : leverage HyDEQueryTransform to generate
# : hypothetical document and use it for embedding lookup
# Parameters
#   - include_original: This argument likely indicates that the original query
#                       should also be included in the transformed query. 
#                       This allows the query engine to consider both the original and the enhanced versions.

hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)
response = hyde_query_engine.query(query_str)

print(str(response))

query_bundle = hyde(query_str)
hyde_doc = query_bundle.embedding_strs[0]

print(hyde_doc)

