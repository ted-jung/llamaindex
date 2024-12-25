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
    
# generate question regarding topic
prompt_str1 = "Please generate a concise question about Paul Graham's life regarding the following topic {topic}"
prompt_tmpl1 = PromptTemplate(prompt_str1)
# use HyDE to hallucinate answer.
prompt_str2 = (
    "Please write a passage to answer the question\n"
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{query_str}\n"
    "\n"
    "\n"
    'Passage:"""\n'
)
prompt_tmpl2 = PromptTemplate(prompt_str2)

retriever = index.as_retriever(similarity_top_k=5)
p = QueryPipeline(
    chain=[prompt_tmpl1, llm, prompt_tmpl2, llm, retriever], verbose=True
)

nodes = p.run(topic="college")
print(len(nodes))
for node in nodes:
    print(node.text)

