# ===========================================================================
# AutoRetrieval from a Vector Database
# Created: 6, Feb 2025
# Updated: 6, Feb 2025
# Writer: Ted, Jung
# Description: 
#   1. VectorIndexAutoRetriever capabilities("auto-retrieval")
#      : allow LLM to infer the right query parameter
# ===========================================================================

import os
import getpass
from IPython import embed
import openai
import clickhouse_connect

from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import (
    VectorStoreInfo,
    MetadataInfo,
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

from typing import List, Tuple, Any
from pydantic import BaseModel, Field

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
from llama_index.core.schema import TextNode, Document

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.embed_model = embed_model

def textnode_to_document(textnode):
  """
  Converts a TextNode object to a Document object.

  Args:
    textnode: The TextNode object to convert.

  Returns:
    A Document object with the text and metadata from the TextNode.
  """
  return Document(text=textnode.text, metadata=textnode.metadata)

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
            "gender": "male",
            "born": 1963,
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
            "gender": "female",
            "born": 1975,
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
            "gender": "male",
            "born": 1971,
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
            "gender": "female",
            "born": 1988,
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
            "gender": "male",
            "born": 1985,
        },
    ),
]


ch_client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    username="default",
    password="magic",
    database="default",
)

vector_store = ClickHouseVectorStore(
    ch_client, 
    table="quickstart_index",
    embed_model=embed_model
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = [textnode_to_document(node) for node in nodes]

# temp_index = VectorStoreIndex(nodes, storage_context=storage_context) 
# : init with nodes does not work. so, reverse nodes to documents.
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
storage_context.persist()



# Define function tool
# OpenAI doesn't work with pydantic objects..so VectorStoreInfo is requried
# It could be used for auto retrieval

# hardcode top k for now
top_k = 3

# Define vector store info describing schema of VectorStoreIndex
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
        MetadataInfo(
            name="gender",
            type="str",
            description=("Gender of the celebrity, one of [male, female]"),
        ),
        MetadataInfo(
            name="born",
            type="int",
            description=("Born year of the celebrity, could be any integer"),
        ),
    ],
)


# define pydantic model for auto-retrieval function
class AutoRetrieveModel(BaseModel):
    query: str = Field(..., description="natural language query string")
    filter_key_list: List[str] = Field(
        ..., description="List of metadata filter field names"
    )
    filter_value_list: List[Any] = Field(
        ...,
        description=(
            "List of metadata filter field values (corresponding to names"
            " specified in filter_key_list)"
        ),
    )
    filter_operator_list: List[str] = Field(
        ...,
        description=(
            "Metadata filters conditions (could be one of <, <=, >, >=, ==, !=)"
        ),
    )
    filter_condition: str = Field(
        ...,
        description=("Metadata filters condition values (could be AND or OR)"),
    )


description = f"""\
Use this tool to look up biographical information about celebrities.
The vector database schema is given below:
{vector_store_info.model_dump_json()}
"""


# Define Autoretrieve Functions
def auto_retrieve_fn(
    query: str,
    filter_key_list: List[str],
    filter_value_list: List[any],
    filter_operator_list: List[str],
    filter_condition: str,
):
    """Auto retrieval function.

    Performs auto-retrieval from a vector database, and then applies a set of filters.

    """
    query = query or "Query"

    metadata_filters = [
        MetadataFilter(key=k, value=v, operator=op)
        for k, v, op in zip(
            filter_key_list, filter_value_list, filter_operator_list
        )
    ]
    retriever = VectorIndexRetriever(
        index,
        filters=MetadataFilters(
            filters=metadata_filters, condition=filter_condition
        ),
        top_k=top_k,
    )
    query_engine = RetrieverQueryEngine.from_args(retriever)

    response = query_engine.query(query)
    return str(response)



# FunctionTool having (RetriverQueryEngine, description,function schema)
auto_retrieve_tool = FunctionTool.from_defaults(
    fn=auto_retrieve_fn,
    name="celebrity_bios",
    description=description,
    fn_schema=AutoRetrieveModel,
)



# Initialize Agent

agent = OpenAIAgent.from_tools(
    [auto_retrieve_tool],
    llm=OpenAI(temperature=0, model="gpt-4o-mini"),
    verbose=True,
)

# response = agent.chat("Tell me about two celebrities from the United States. ")
# print(str(response))

# response2 = agent.chat("Tell me about two celebrities born after 1980. ")
# print(str(response2))

response3 = agent.chat(
    "Tell me about few celebrities under category business and born after 1950. "
)
print(str(response3))