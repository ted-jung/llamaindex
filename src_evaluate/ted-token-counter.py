# ===========================================================================
# Count the number of tokens
# Date: 31, Jan 2025
# Writer: Ted, Jung
# Description: token(input, output) = cost
#              How to count it?
# ===========================================================================

import tiktoken

from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import (
    VectorStoreIndex, Settings, SimpleDirectoryReader
)
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4o-mini").encode
)
callback_manager = CallbackManager([token_counter])

Settings.llm = Ollama(model="llama3.2", temperature=0.0, request_timeout=720.0)
Settings.callback_manager = callback_manager
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


# build index
documents = SimpleDirectoryReader(input_files=["./data/paul_graham/paul_graham_essay_short.txt"]).load_data()
vector_index = VectorStoreIndex.from_documents(documents=documents)

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)

# Reset counts
token_counter.reset_counts()


# Count token when Query
query_engine = vector_index.as_query_engine()

response = query_engine.query("What did the Paul Graham do in college?")

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)