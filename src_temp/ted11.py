from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.readers.wikipedia import WikipediaReader
from typing import List
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex


from pydantic import BaseModel


reader = WikipediaReader()

tool = OnDemandLoaderTool.from_defaults(
    reader,
    index_cls=VectorStoreIndex,
    name="Wikipedia Tool",
    description="A tool for loading and querying articles from Wikipedia"
)

response = tool(["Berlin"], query_str="What's the arts and culture scene in Berlin?")
print(response)