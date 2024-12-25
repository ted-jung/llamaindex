from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from pathlib import Path

current_dir = Path.cwd()

import os,pickle
from flask import Flask, request
from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.node_parser import SentenceSplitter



index = None
stored_docs = {}
lock = Lock()

#index_name = "./saved_index"
index_dir = f"{current_dir}/web-app/.index"
pkl_name = "stored_documents.pkl"


def initialize_index():
    global index, stored_docs
    
    transformations = SentenceSplitter(chunk_size=512)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = Ollama(model="llama3.2", request_timeout=360.0, temperature=0)
            
    with lock:
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=f"{index_dir}"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=f"{index_dir}"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=f"{index_dir}"),
        )

        if os.path.exists(index_dir):
            index = load_index_from_storage(storage_context)
        else:
            index = VectorStoreIndex(nodes=[])
            index.storage_context.persist(persist_dir=index_dir)
            #documents = SimpleDirectoryReader(f"{current_dir}/web-app/documents").load_data()
            # index = VectorStoreIndex.from_documents(
            #     documents, storage_context=storage_context
            # )
            
            # storage_context.persist(index_dir)
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                stored_docs = pickle.load(f)


def query_index(query_text):
    global index
    query_engine = index.as_query_engine(similarity_top_k=2)
    response = query_engine.query(query_text)
    return str(response)


def insert_into_index(doc_file_path, doc_id=None):
    global index, stored_docs
    documents = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()[0]
    
    with lock:
        for document in documents:
            if doc_id is not None:
                document.id_ = doc_id
            index.insert(document)
                
            stored_docs[document.id_] = document.text[0:200]
                
        index.storage_context.persist(persist_dir=index_dir)

        first_document = documents[0]
        # Keep track of stored docs -- llama_index doesn't make this easy
        stored_docs[first_document.doc_id] = first_document.text[0:200] # only take the first 200 chars

        with open(pkl_name, "wb") as f:
            pickle.dump(stored_docs, f)

    return
      

def get_documents_list():
    """Get the list of currently stored documents."""
    global stored_doc
    documents_list = []
    for doc_id, doc_text in stored_docs.items():
        documents_list.append({"id": doc_id, "text": doc_text})

    return documents_list

  
if __name__ == "__main__":
    # init the global index
    print("initializing index...")
    initialize_index()

    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(("", 5602), b"password")
    manager.register("query_index", query_index)
    manager.register("insert_into_index", insert_into_index)
    manager.register('get_documents_list', get_documents_list)
    server = manager.get_server()

    print("starting server...")
    server.serve_forever()