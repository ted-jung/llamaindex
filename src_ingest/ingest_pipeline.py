from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="./llama", required_exts=[".md"])
docs = reader.load_data()
print(docs)
documents = []
for idx, doc in enumerate(docs):
    print(f"{idx} - {doc.metadata} - {doc.text}")
    documents.append(doc.text)
    
from llama_index.core import Document, VectorStoreIndex

# text_list = ["this is the sentence", "i am ted jung"]
# documents = [Document(text=t) for t in text_list.text]

print(documents[0])
from llama_index.core import Document
from langchain_ollama import OllamaEmbeddings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
        OllamaEmbeddings(model="llama3.2"),
    ]
)

# run the pipeline
nodes = pipeline.run(docs)

print(nodes)