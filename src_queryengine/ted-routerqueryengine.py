from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex


# load documents
documents = SimpleDirectoryReader("./data/").load_data()
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)


# initialize settings (set chunk size)
Settings.chunk_size = 1024
nodes = Settings.node_parser.get_nodes_from_documents(documents)


# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)


summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)


list_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()


from llama_index.core.tools import QueryEngineTool


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

from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import (
    LLMMultiSelector,
    LLMSingleSelector,
)



query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)
response = query_engine.query("What is the summary of the document?")
print(str(response))


query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")
print(str(response))

from llama_index.core import SimpleKeywordTableIndex

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