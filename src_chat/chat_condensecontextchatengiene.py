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

# documents = SimpleDirectoryReader(
#     input_files=["data/paul_graham_essay.txt"]
# ).load_data()

context_str = index[0].text

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an essay discussing Paul Grahams life."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=True,
)

response = chat_engine.chat("Hello")
print(response)
print ("================================")

response = chat_engine.chat("What did Paul Graham do after YC?")
print(response)
print ("================================")

response = chat_engine.chat("What about after that?")
print(response)
print ("================================")

response = chat_engine.chat("Can you tell me more?")
print(response)
print ("================================")