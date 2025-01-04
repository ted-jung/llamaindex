from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings,
    SQLDatabase,
)
from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool

import os
import clickhouse_connect


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=1500.0)

client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    username="default",
    password="magic",
)


# define clickhouse vector index
vector_store = ClickHouseVectorStore(clickhouse_client=client, table="ted_llama")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex([], storage_context=storage_context)


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
    
cities = ["Totonto", "Berlin", "Tokyo"]
wikipedia_reader = WikipediaReader() 
wiki_docs = wikipedia_reader.load_data(pages=cities)


sql_database = SQLDatabase(engine, include_tables=["city_stats"])
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"],
)


# Insert documents into vector index
# Each document has metadata of the city attached
for city, wiki_doc in zip(cities, wiki_docs):
    nodes = Settings.node_parser.get_nodes_from_documents([wiki_doc])
    # add metadata to each node
    for node in nodes:
        node.metadata = {"title": city,"name": "ted"}
    vector_index.insert_nodes(nodes)
    

vector_store_info = VectorStoreInfo(
    content_info="articles about different cities",
    metadata_info=[
        MetadataInfo(
            name="title", type="str", description="The name of the city"
        ),
    ],
)
vector_auto_retriever = VectorIndexAutoRetriever(vector_index, vector_store_info=vector_store_info)
retriever_query_engine = RetrieverQueryEngine.from_args(vector_auto_retriever)



# queryengine tools
sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        "Useful for translating a natural language query into a SQL query over"
        " a table containing: city_stats, containing the population/country of"
        " each city"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=retriever_query_engine,
    description=(
        f"Useful for answering semantic questions about different cities"
    ),
)


from llama_index.core.query_engine import SQLAutoVectorQueryEngine

# two tools (one is for sql the other is for vector)
query_engine = SQLAutoVectorQueryEngine(
    sql_tool, vector_tool,
)

response = query_engine.query(
    "Tell me about the arts and culture of the city with the highest"
    " population"
)

print(response)
print(100*"*")

response = query_engine.query("Tell me about the history of Berlin")
print(response)
print(100*"*")