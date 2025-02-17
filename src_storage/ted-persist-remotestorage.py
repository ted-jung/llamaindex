# ===========================================================================
# Persist index to Remote Store
# Created: 6, Feb 2025
# Updated: 17, Feb 2025
# Writer: Ted, Jung
# Description: 
#   1. prepare data
#   2. Create a client to access remote store(i.g, ClickHouse)
#   3. Prepare VectorStore and let storagecontext know the store
#   4. Load document and persist it
# ===========================================================================


import textwrap
import clickhouse_connect

from urllib import response
from warnings import filters


from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, Document

from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)


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



# Create a client to connect ClickHouse
ch_client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    username="default",
    password="magic",
    database="default",
)



# Create VectorStore on Store(ClickHouse)
# A utility container for storing nodes,indices, and vectors.
# Turn nodes to documents
# Create an index based on documents and persist it

documents = [textnode_to_document(node) for node in nodes]

vector_store = ClickHouseVectorStore(
    ch_client, 
    table="quickstart_index",
    embed_model=embed_model
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

storage_context.persist()


query_engine = index.as_query_engine(
    filters=MetadataFilters(filters=[ExactMatchFilter(key="category", value="Entertainment")]),
    similarity_top_k=2,
    vector_store_query_mode="hybrid",
    llm=llm
)


response = query_engine.query("Who is Angelina Jolie? add regarding shes family")
print(textwrap.fill(str(response), 100))



# Clear All indexes -> vectorstore -> clickhouse

for document in documents:
  index.delete_ref_doc(document.doc_id)