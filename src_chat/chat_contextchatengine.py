from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

current_dir = Path.cwd()
data = SimpleDirectoryReader(input_dir=f"{current_dir}/data").load_data()
index = VectorStoreIndex.from_documents(data)

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an essay discussing Paul Grahams life."
    ),
)

response = chat_engine.chat("Hello")
print(response)

response = chat_engine.chat("What did Paul Graham do after YC?")
print(response)

response = chat_engine.chat("What about after that?")
print(response)

response = chat_engine.chat("Can you tell me more?")
print(response)