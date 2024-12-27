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

docs = dict(name = "John", age = 36, country = "Norway")
# run the pipeline
nodes = pipeline.run(documents=docs)

print(nodes)