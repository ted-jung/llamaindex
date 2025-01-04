from llama_index.llms.ollama import Ollama
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

import ollama
import os
import clickhouse_connect
import textwrap


# documents = SimpleDirectoryReader("./paul_graham/source_files").load_data()
# print("Document ID:", documents[0].doc_id)
# print("Number of Documents: ", len(documents))
client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    username="default",
    password="magic",
)
loader = SimpleDirectoryReader("./paul_graham/source_files")
documents = loader.load_data()
for file in loader.input_files:
    print(file)


for document in documents:
    document.metadata = {"user_id": "123", "favorite_color": "blue"}

vector_store = ClickHouseVectorStore(clickhouse_client=client,table="ted_llama")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(key="user_id", value="123"),
        ]
    ),
    similarity_top_k=2,
    vector_store_query_mode="hybrid",
)
response = query_engine.query("What did the author learn?")
print(textwrap.fill(str(response), 100))

index.storage_context.persist()

# Query the index
response = query_engine.query("What is the capital of France?")
print(response)
