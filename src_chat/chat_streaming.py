from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.memory import ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

from pathlib import Path

# document directory
current_dir = Path.cwd()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", temperature=0, request_timeout=720.0)

data = SimpleDirectoryReader(input_dir=f"{current_dir}/llama/data").load_data()
index = VectorStoreIndex.from_documents(data)

chat_engine = index.as_chat_engine(
                    chat_mode="condense_plus_context",
                    context_prompt=(
                        "You are a chatbot, able to have normal interactions, as well as talk"
                        " about an essay discussing Paul Grahams life."
                        "Here are the relevant documents for the context:\n"
                        "{context_str}"
                        "\nInstruction: Based on the above documents, provide a detailed answer for the user question below."
                    ),
                )

response = chat_engine.stream_chat("What did Paul Graham do after YC?")

for token in response.response_gen:
    print(token, end="")