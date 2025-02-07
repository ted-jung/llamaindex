# ===========================================================================
# agent and queryenginetools as engine itself
# Created: 7, Feb 2025
# Updated: 7, Feb 2025
# Writer: Ted, Jung
# Description: 
#   1. OpenAIAgent with queryenginetool
#      : load -> index -> engines -> enginetool -> agent with enginetools -> Query
# ===========================================================================


import os 

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent

curr_dir = os.getcwd()

Settings.llm = OpenAI(model="gpt-4o-mini")

try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/lyft"
    )
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/uber"
    )
    uber_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False


# Build index with loading data
# Persist data 
if not index_loaded:
    # load data
    lyft_docs = SimpleDirectoryReader(
        input_files=[f"{curr_dir}/data/10k/lyft_2021.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=[f"{curr_dir}/data/10k/uber_2021.pdf"]
    ).load_data()

    # build index
    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    uber_index = VectorStoreIndex.from_documents(uber_docs)

    # persist index
    lyft_index.storage_context.persist(persist_dir="./storage/lyft")
    uber_index.storage_context.persist(persist_dir="./storage/uber")



lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)


# Add Engines with metadata of each engine in a QueryEngineTool's list
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
                "If there is no fit answer? then, return answer based on your knowledge."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
                "If there is no fit answer? then, return answer based on your knowledge."
            ),
        ),
    ),
]


agent = OpenAIAgent.from_tools(query_engine_tools, verbose=True)

agent.chat_repl()