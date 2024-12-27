from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.memory import ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

from pathlib import Path

# document directory
current_dir = Path.cwd()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

llm = Ollama(model="llama3.2", request_timeout=720.0)
data = SimpleDirectoryReader(input_dir=f"{current_dir}/llama/data").load_data()
index = VectorStoreIndex.from_documents(data)

# configure chat_engine
chat_engine = index.as_chat_engine(
                chat_mode="condense_plus_context",
                memory=memory,
                llm=llm,
                context_prompt=(
                    "You are a chatbot, able to have normal interactions, as well as talk"
                    " about an essay discussing Paul Grahams life."
                    "Here are the relevant documents for the context:\n"
                    "{context_str}"
                    "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
                ),
                verbose=False,
            )

response = chat_engine.chat(
    "What are the first programs Paul Graham tried writing?"
)

print(response)

response_2 = chat_engine.chat("Can you tell me more?")
print(response_2)

chat_engine.reset()

response_3 = chat_engine.chat("Hello! What do you know?")
print(response_3)