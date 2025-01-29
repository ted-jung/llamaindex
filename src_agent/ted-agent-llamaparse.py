from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0, temperature=0)

current_dir = Path.cwd()
documents = SimpleDirectoryReader(f"{current_dir}/data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query(
    "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?"
)
print(response)

documents2 = LlamaParse(result_type="markdown").load_data(
    f"{current_dir}/data/2023_canadian_budget.pdf"
)
index2 = VectorStoreIndex.from_documents(documents2)
query_engine2 = index2.as_query_engine()

response2 = query_engine2.query(
    "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?"
)
print(response2)