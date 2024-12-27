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

# document directory
current_dir = Path.cwd()
documents = SimpleDirectoryReader(f"{current_dir}/llama/data").load_data()

# bge-base embedding model, llm model (ollama)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=720.0)

# transforming text into a structured format (searching, querying)
# embedding model is engaged in this step
index = VectorStoreIndex.from_documents(documents,)

# LLM is engaged in this step
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

print(response)

index.storage_context.persist(persist_dir=f"{current_dir}/llama/storage")