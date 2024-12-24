from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.anthropic import Anthropic
from pathlib import Path

# document directory
current_dir = Path.cwd()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

data = SimpleDirectoryReader(input_dir=f"{current_dir}/data").load_data()
index = VectorStoreIndex.from_documents(data)

# configure chat_engine
#chat_engine = index.as_chat_engine(chat_mode="best", llm=llm, verbose=False)
chat_engine = index.as_chat_engine(chat_mode="best", verbose=False)

response = chat_engine.chat(
    "What are the first programs Paul Graham tried writing?"
)

print(response)

response = chat_engine.chat("And did he visit India?")
print(response)

chat_engine.chat_repl()


