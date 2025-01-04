import camelot
import os

# https://en.wikipedia.org/wiki/The_World%27s_Billionaires
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    get_response_synthesizer,
)
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.schema import IndexNode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.readers.file import PyMuPDFReader
from typing import List

from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=1500.0)

file_path = "./data/billionaire/billionaires_page.pdf"
# initialize PDF reader
reader = PyMuPDFReader()
docs = reader.load(file_path)

# use camelot to parse tables
def get_tables(path: str, pages: List[int]):
    table_dfs = []
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        table_df = table_list[0].df
        table_df = (
            table_df.rename(columns=table_df.iloc[0])
            .drop(table_df.index[0])
            .reset_index(drop=True)
        )
        table_dfs.append(table_df)
    return table_dfs

table_dfs = get_tables(file_path, pages=[3,27])
# shows list of top billionaires in 2023
print(table_dfs[0])

# show list of top billionaires
print(table_dfs[1])

df_query_engines = [
    PandasQueryEngine(table_df) for table_df in table_dfs
]

response = df_query_engines[1].query(
    "How many billionaires were there in 2009?"
)
print(str(response)) 

response = df_query_engines[0].query(
    "What's the Net worth\n(USD) of the second richest billionaire in 2023?"
)
print(str(response))


doc_nodes = Settings.node_parser.get_nodes_from_documents(docs,show_progress=False)

# define index nodes
summaries = [
    (
        "This node provides information about the world's richest billionaires"
        " in 2023"
    ),
    (
        "This node provides information on the number of billionaires and"
        " their combined Net worth\n(USD) from 2000 to 2023."
    ),
]

df_nodes = [
    IndexNode(text=summary, index_id=f"pandas{idx}") for idx, summary in enumerate(summaries)
]

df_id_query_engine_mapping = {
    f"pandas{idx}": df_query_engine for idx, df_query_engine in enumerate(df_query_engines)
}

# construct top-level vector index + query engine
vector_index = VectorStoreIndex(doc_nodes + df_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)


# baseline vector index (that doesn't include the extra df nodes).
# used to benchmark
vector_index0 = VectorStoreIndex(doc_nodes)
vector_query_engine0 = vector_index0.as_query_engine()


recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_id_query_engine_mapping,
    verbose=True,
)

response_synthesizer = get_response_synthesizer(response_mode="compact")

query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever, response_synthesizer=response_synthesizer
)

response = query_engine.query(
    "What's the Net worth\n(USD) of the second richest billionaire in 2023?"
)

print(response)

response = vector_query_engine0.query(
    "How many billionaires were there in 2009?"
)

print(str(response))

response.source_nodes[0].node.get_content()

response = query_engine.query(
    "Which billionaires are excluded from this list?"
)

print(str(response))