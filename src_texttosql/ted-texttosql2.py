# ===========================================================================
# ObjectIndex
# Date: 6, Feb 2025
# Writer: Ted, Jung
# Description: ObjectIndex with (QueryEngine, Retriever)
#              SQLTableRetrieverQueryEngine, 
#              NLSQLRetriever, 
#              RetrieverQueryEngine
# ===========================================================================


import matplotlib.pyplot as plt

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)

from llama_index.core.retrievers import NLSQLRetriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.query_engine import RetrieverQueryEngine

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    insert,
)

from llama_index.core import VectorStoreIndex, SQLDatabase, Settings



Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=720.0)
# Settings.llm = OpenAI(model="gpt-4o-mini", request_timeout=720.0)


engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)

metadata_obj.create_all(engine)
sql_database = SQLDatabase(engine, include_tables=["city_stats"])



rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {"city_name": "Chicago", "population": 2679000, "country": "United States"},
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]


for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)


# set Logging to DEBUG for more detailed outputs
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    (SQLTableSchema(table_name="city_stats"))
]  # add a SQLTableSchema for each table


# save table schema in an index if we do not know ahead of time which table we would like to use
obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=1)
)
response = query_engine.query("Which city has the highest population?")

#response.source_nodes
for i in response.source_nodes:
    print(i.text)

response = query_engine.query("the sum of total population of all cities?")
print(response)


# Using Text-to-SQL Retriever (NLSQLRetriever)


# default retrieval (return_raw=True)
nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=["city_stats"], return_raw=True
)
results = nl_sql_retriever.retrieve(
    "Return the top 5 cities (along with their populations) with the highest population."
)

for result in results:
    print(result)


for n in results:
    display_source_node(n)


query_engine = RetrieverQueryEngine.from_args(nl_sql_retriever)
response = query_engine.query(
    "Return the top 5 cities (along with their populations) with the highest population."
)
print(response)