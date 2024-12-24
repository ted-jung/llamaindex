from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.prompts.system import SHAKESPEARE_WRITING_ASSISTANT

# document directory
current_dir = Path.cwd()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", temperature=0, request_timeout=720.0)

chat_engine = SimpleChatEngine.from_defaults(
    system_prompt=SHAKESPEARE_WRITING_ASSISTANT
)

response = chat_engine.chat(
    "Say something profound and romantic about fourth of July"
)
print(response)