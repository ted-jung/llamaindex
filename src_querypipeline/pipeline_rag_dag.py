from dotenv import load_dotenv
load_dotenv() 
from llama_index.core.query_pipeline import QueryPipeline
import os
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.response_synthesizers import TreeSummarize

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
#Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
llm = Ollama(model="llama3.2", request_timeout=360.0)

current_dir = os.getcwd()
reader = SimpleDirectoryReader(f"{current_dir}/paul_graham")
docs = reader.load_data()

if not os.path.exists("storage"):
    index = VectorStoreIndex.from_documents(docs)
    # save index to disk
    index.set_index_id("vector_index")
    index.storage_context.persist(f"{current_dir}/storage")
else:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=f"{current_dir}/storage")
    # load index
    index = load_index_from_storage(storage_context, index_id="vector_index")
    
    
# define modules
prompt_str = "Please generate a question about Paul Graham's life regarding the following topic {topic}"
prompt_tmpl = PromptTemplate(prompt_str)
retriever = index.as_retriever(similarity_top_k=3)
reranker = CohereRerank()
summarizer = TreeSummarize(llm=llm)

# define query pipeline
p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "llm": llm,
        "prompt_tmpl": prompt_tmpl,
        "retriever": retriever,
        "summarizer": summarizer,
        "reranker": reranker,
    }
)

p.add_link("prompt_tmpl", "llm")
p.add_link("llm", "retriever")
p.add_link("retriever", "reranker", dest_key="nodes")
p.add_link("llm", "reranker", dest_key="query_str")
p.add_link("reranker", "summarizer", dest_key="nodes")
p.add_link("llm", "summarizer", dest_key="query_str")

# look at summarizer input keys
print(summarizer.as_query_component().input_keys)

response = p.run(topic="YC")
print(str(response))

## create graph
# from pyvis.network import Network

# net = Network(notebook=True, cdn_resources="in_line", directed=True)
# net.from_nx(p.dag)
# net.show("rag_dag.html")