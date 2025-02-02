# ===========================================================================
# Dynamically Retrieve Chunks depending on your task
# Date: 2, Feb 2025
# Writer: Ted, Jung
# Description: Summary or Comparison by router and data agent
#       index to engine(list,vector) to tool to queryenginetools to roterqueryengine
# ===========================================================================


from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    SimpleKeywordTableIndex,
)


from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine

from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
    LLMSingleSelector,
    LLMMultiSelector,
)


Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# initialize settings (set chunk size)
Settings.chunk_size = 1024
nodes = Settings.node_parser.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)


# Define Index(Summary, vector) over same data
# Define Query Engines  and Set Metadata

summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

list_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()


list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description=(
        "Useful for summarization questions related to Paul Graham eassy on"
        " What I Worked On."
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)


# Define Router Query Engine

# PydanticSingleSelector
query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)


response = query_engine.query("What is the summary of the document?")
print(str(response))


response = query_engine.query("What did Paul Graham do after RICS?")
print(str(response))

print(20*"=====")

# LLMSingleSelector
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")
print(str(response))


# PydanticMultiSelector

keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context using keywords from Paul"
        " Graham essay on What I Worked On."
    ),
)

query_engine = RouterQueryEngine(
    selector=PydanticMultiSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
        keyword_tool,
    ],
)

# This query could use either a keyword or vector query engine, so it will combine responses from both
response = query_engine.query(
    "What were noteable events and people from the authors time at Interleaf"
    " and YC?"
)
print(str(response))


# [optional] look at selected results
print(str(response.metadata["selector_result"]))