from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from pathlib import Path

current_dir = Path.cwd()

def initialize_index():
    global index
    storage_context = StorageContext.from_defaults()
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = Ollama(model="llama3.2", request_timeout=360.0, temperature=0)
    
    index_dir = f"{current_dir}/web-app/.index"
    if os.path.exists(index_dir):
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(f"{current_dir}/web-app/documents").load_data()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        storage_context.persist(index_dir)
        
from flask import Flask
from flask import request
import os


app = Flask(__name__)


@app.route("/")
def home():
    return "Hello World!"

@app.route("/query", methods=["GET"])
def query_index():
    global index
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return str(response), 200



if __name__ == "__main__":
    initialize_index()
    app.run(host="0.0.0.0", port=5601)
    
