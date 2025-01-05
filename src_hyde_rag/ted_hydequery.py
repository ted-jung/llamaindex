import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from IPython.display import Markdown, display


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=720.0)

# initialize settings (set chunk size)
Settings.chunk_size = 1024
# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(documents)

query_str = "what did paul graham do after going to RISD"

query_engine = index.as_query_engine()
response = query_engine.query(query_str)
print("normal vectorstoreindex engine****************")
print(response)


hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)
response = hyde_query_engine.query(query_str)
print("hypotetical document for embedding****************")
print(response)

query_bundle = hyde(query_str)
hyde_doc = query_bundle.embedding_strs[0]

print("this is the hyde docs****************")
print(hyde_doc)
