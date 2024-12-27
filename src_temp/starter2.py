from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from pathlib import Path
from llama_index.core.node_parser import TokenTextSplitter

# document directory
current_dir = Path.cwd()
documents = SimpleDirectoryReader(f"{current_dir}/llama/data").load_data()

# bge-base embedding model, llm model (ollama)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", temperature=0.2, request_timeout=720.0)

splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

print(nodes[0])

index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine(similarity_top_k=5)

response = query_engine.query("What did Paul Graham do growing up?")

print(response)

for node in response.source_nodes:
    print("=======================")
    print(node)

