from llama_index.core import (
    SimpleDirectoryReader, 
    Settings, 
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from pathlib import Path
from llama_index.core import SummaryIndex

# document directory
current_dir = Path.cwd()
documents = SimpleDirectoryReader(f"{current_dir}/llama/data").load_data()

# bge-base embedding model, llm model (ollama)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", temperature=0.2, request_timeout=720.0)

summary_index = SummaryIndex(documents)
query_engine = summary_index.as_query_engine()

summary = query_engine.query("Provide the summary of the document.")
print(summary)