from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.workflow.decorators import StepConfig

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

current_dir = Path.cwd()

# create nodes
documents = SimpleDirectoryReader(input_files=[f"{current_dir}/data/paul_graham_essay.txt"]).load_data()
splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

# Construct an index by loading documents into a VectorStoreIndex.
index = VectorStoreIndex(nodes)

# configure retriever
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

# configure response synthesizer
synthesizer = get_response_synthesizer(response_mode="refine")

# construct query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
)

response = query_engine.query("What did Paul Graham do growing up?")
print(response)

retriever = index.as_retriever(similarity_top_k=3)
retrieved_nodes = retriever.retrieve("What did Paul Graham do growing up?")

for text_node in retrieved_nodes:
    display_source_node(text_node, source_length=500)
    print(text_node)
    print("========")