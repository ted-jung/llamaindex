from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.anthropic import Anthropic
from pathlib import Path
from llama_index.core.node_parser import TokenTextSplitter


# document directory
current_dir = Path.cwd()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", temperature=0.2, request_timeout=720.0)

data = SimpleDirectoryReader(input_dir=f"{current_dir}/data").load_data()
index = VectorStoreIndex.from_documents(data)

# configure chat_engine
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=False)
# chat_engine = index.as_chat_engine(chat_mode="condense_question", llm = llm, verbose=False)

response = chat_engine.chat("What did Paul Graham do after YC?")

print(response)

response = chat_engine.chat("What about after that")
print(response)

response = chat_engine.chat("Can you tell me more")
print(response)
