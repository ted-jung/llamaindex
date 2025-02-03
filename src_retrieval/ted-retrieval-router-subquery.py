# ===========================================================================
# RouterQueryEngine, subquery to simplify complex question
# Date: 3, Feb 2025
# Writer: Ted, Jung
# Description: 
#       1. answering on the complex question against relevant data sources
#       2. gather all intermediate response and synthesize final response
# ===========================================================================

import os
import nest_asyncio
import logging
import sys
import openai
import asyncio

from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    Settings,
)

from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
)

from llama_index.core.query_engine.sub_question_query_engine import (
    SubQuestionQueryEngine,
)

from llama_index.core.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.tools.query_engine import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from IPython.display import display, HTML
from llama_index.core.text_splitter import SentenceSplitter


llm = OpenAI(model="gpt-4o-mini", request_timeout=720.0)
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# llm = Ollama(model="llama3.2", request_timeout=720.0)
# Settings.llm = llm
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


curr_dir = os.getcwd()
nest_asyncio.apply()

# Set up the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set logger level to INFO

# Clear out any existing handlers
logger.handlers = []

# Set up the StreamHandler to output to sys.stdout (Colab's output)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Set handler level to INFO

# Add the handler to the logger
logger.addHandler(handler)



# load documents
# create parser and parse document into nodes

documents = SimpleDirectoryReader(f"{curr_dir}/data/paul_graham").load_data()
parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
nodes = parser(documents)

# Create Index(Summary, Vector)
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)


# Define queryengine(summary, vector)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()



# Create Tool(Summary Index tool, Vector Index tool)
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to Paul Graham eassy on What I Worked On.",
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On.",
)



# Define Router query engine with Selecors
# 1. PydanticSingleSelector (only for OpenAI)
query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
)

print(query_engine.query("What is the summary of the document?"))


# 2. LLMSingleSelector
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
)

print(query_engine.query("What is the summary of the document?"))



keyword_index = SimpleKeywordTableIndex(nodes)

keyword_query_engine = keyword_index.as_query_engine()

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_query_engine,
    description="Useful for retrieving specific context using keywords from Paul Graham essay on What I Worked On.",
)


# 3. PydanticMultiSelector
query_engine = RouterQueryEngine(
    selector=PydanticMultiSelector.from_defaults(),
    query_engine_tools=[vector_tool, keyword_tool, summary_tool],
)

response = query_engine.query(
    "What were noteable events and people from the authors time at Interleaf and YC?"
)


# SubQuestion Engine

lyft_docs = SimpleDirectoryReader(input_files=[f"{curr_dir}/data/10k/lyft_2021.pdf"]).load_data()
uber_docs = SimpleDirectoryReader(input_files=[f"{curr_dir}/data/10k/uber_2021.pdf"]).load_data()

lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)


query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description="Provides information about Lyft financials for year 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description="Provides information about Uber financials for year 2021",
        ),
    ),
]

sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)


async def lyft_func():
    response = await lyft_engine.aquery(
        "What is the revenue of Lyft in 2021? Answer in millions with page reference"
    )
    return response

async def uber_func():
    response = await uber_engine.aquery(
        "What is the revenue of Uber in 2021? Answer in millions, with page reference"
    )

    return response

async def subq_func():
    response = await sub_question_query_engine.aquery(
        "Compare revenue growth of Uber and Lyft from 2020 to 2021"
    )
    print(response)

    return response


if __name__ == "__main__":
    # print(lyft_func())

    # print(uber_func())

    print(asyncio.run(subq_func()))
