# =============================================================================
# Advanced Retrieval
# Created: 10, Feb 2025
# Updated: 10, Feb 2025
# Writer: Ted, Jung
# Description: 
#       1. Assistant Agent from [Tools from [Engine from [Indexes]]] (Summary, Vector)
#       2. AutoRetrieval from Vector Database (function interface to OpenAI)
#       3. Joint Text-to-SQL and Semantic Search
#       4. 
# =============================================================================


import nest_asyncio
import os
import clickhouse_connect


from llama_index.llms.openai import OpenAI
from llama_index.core import (
    Settings,
    StorageContext, 
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.agent.openai import OpenAIAssistantAgent
from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode,Document

# define function tool
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import (
    VectorStoreInfo,
    MetadataInfo,
    ExactMatchFilter,
    MetadataFilters,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from typing import List, Tuple, Any
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
    insert,
)
from llama_index.core import SQLDatabase
from llama_index.core.indices import SQLStructStoreIndex
from llama_index.core.query_engine import NLSQLTableQueryEngine



embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# load documents
curr_dir = os.getcwd()
documents = SimpleDirectoryReader(f"{curr_dir}/data/paul_graham/").load_data()

nest_asyncio.apply()

def textnode_to_document(textnode):
  """
  Converts a TextNode object to a Document object.

  Args:
    textnode: The TextNode object to convert.

  Returns:
    A Document object with the text and metadata from the TextNode.
  """
  return Document(text=textnode.text, metadata=textnode.metadata)


# 1 Setup Vector + Summary Indexes/Query Engines/Tools

# initialize settings (set chunk size)
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.chunk_size = 1024
nodes = Settings.node_parser.get_nodes_from_documents(documents)

# Initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# Define Index(Summary , Vector) over Same Data
summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

# Define query engines
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()


# Define Queryenginetool with mapped tool.
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    name="summary_tool",
    description=(
        "Useful for summarization questions related to the author's life"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    name="vector_tool",
    description=(
        "Useful for retrieving specific context to answer specific questions about the author's life"
    ),
)


# Define assistant agent
agent = OpenAIAssistantAgent.from_new(
    name="QA bot",
    instructions="You are a bot designed to answer questions about the author",
    openai_tools=[],
    tools=[summary_tool, vector_tool],
    verbose=True,
    run_retrieve_sleep_time=1.0,
)


response = agent.chat("Can you give me a summary about the author's life?")
print(str(response))




# 2. AutoRetrieval from Vector Database
# initialize client
clickhouse_client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    username="default",
    password="magic",
)

vector_store = ClickHouseVectorStore(
    clickhouse_client, 
    table="ted_index",
    embed_model=embed_model
)
vector_store = ClickHouseVectorStore(clickhouse_client=clickhouse_client)


nodes = [
    TextNode(
        text=(
            "Michael Jordan is a retired professional basketball player,"
            " widely regarded as one of the greatest basketball players of all"
            " time."
        ),
        metadata={
            "category": "Sports",
            "country": "United States",
        },
    ),
    TextNode(
        text=(
            "Angelina Jolie is an American actress, filmmaker, and"
            " humanitarian. She has received numerous awards for her acting"
            " and is known for her philanthropic work."
        ),
        metadata={
            "category": "Entertainment",
            "country": "United States",
        },
    ),
    TextNode(
        text=(
            "Elon Musk is a business magnate, industrial designer, and"
            " engineer. He is the founder, CEO, and lead designer of SpaceX,"
            " Tesla, Inc., Neuralink, and The Boring Company."
        ),
        metadata={
            "category": "Business",
            "country": "United States",
        },
    ),
    TextNode(
        text=(
            "Rihanna is a Barbadian singer, actress, and businesswoman. She"
            " has achieved significant success in the music industry and is"
            " known for her versatile musical style."
        ),
        metadata={
            "category": "Music",
            "country": "Barbados",
        },
    ),
    TextNode(
        text=(
            "Cristiano Ronaldo is a Portuguese professional footballer who is"
            " considered one of the greatest football players of all time. He"
            " has won numerous awards and set multiple records during his"
            " career."
        ),
        metadata={
            "category": "Sports",
            "country": "Portugal",
        },
    ),
]

storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = [textnode_to_document(node) for node in nodes]
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
storage_context.persist()




# hardcode top k for now
top_k = 3

# define vector store info describing schema of vector store
vector_store_info = VectorStoreInfo(
    content_info="brief biography of celebrities",
    metadata_info=[
        MetadataInfo(
            name="category",
            type="str",
            description=(
                "Category of the celebrity, one of [Sports, Entertainment,"
                " Business, Music]"
            ),
        ),
        MetadataInfo(
            name="country",
            type="str",
            description=(
                "Country of the celebrity, one of [United States, Barbados,"
                " Portugal]"
            ),
        ),
    ],
)


# define pydantic model for auto-retrieval function
class AutoRetrieveModel(BaseModel):
    query: str = Field(..., description="natural language query string")
    filter_key_list: List[str] = Field(
        ..., description="List of metadata filter field names"
    )
    filter_value_list: List[str] = Field(
        ...,
        description=(
            "List of metadata filter field values (corresponding to names"
            " specified in filter_key_list)"
        ),
    )


def auto_retrieve_fn(
    query: str, filter_key_list: List[str], filter_value_list: List[str]
):
    """Auto retrieval function.

    Performs auto-retrieval from a vector database, and then applies a set of filters.

    """
    query = query or "Query"

    exact_match_filters = [
        ExactMatchFilter(key=k, value=v)
        for k, v in zip(filter_key_list, filter_value_list)
    ]
    retriever = VectorIndexRetriever(
        index,
        filters=MetadataFilters(filters=exact_match_filters),
        top_k=top_k,
    )
    results = retriever.retrieve(query)
    return [r.get_content() for r in results]


description = f"""\
Use this tool to look up biographical information about celebrities.
The vector database schema is given below:
{vector_store_info.model_dump_json()}
"""

auto_retrieve_tool = FunctionTool.from_defaults(
    fn=auto_retrieve_fn,
    name="celebrity_bios",
    description=description,
    fn_schema=AutoRetrieveModel,
)



agent = OpenAIAssistantAgent.from_new(
    name="Celebrity bot",
    instructions="You are a bot designed to answer questions about celebrities.",
    tools=[auto_retrieve_tool],
    verbose=True,
)

response = agent.chat("Tell me about two celebrities from the United States. ")
print(str(response))



# 3. Joint Text-to-SQL and Semantic Search


engine = create_engine("sqlite:///:memory:", future=True)
metadata_obj = MetaData()
# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)

metadata_obj.create_all(engine)
# print tables
metadata_obj.tables.keys()



rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)



with engine.connect() as connection:
    cursor = connection.exec_driver_sql("SELECT * FROM city_stats")
    print(cursor.fetchall())


sql_database = SQLDatabase(engine, include_tables=["city_stats"])


query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"],
)