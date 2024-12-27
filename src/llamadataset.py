from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core import VectorStoreIndex

import nest_asyncio


# download and install dependencies
rag_dataset, documents = download_llama_dataset(
    "PaulGrahamEssayDataset", "./paul_graham"
)

print(rag_dataset.to_pandas()[:5])

# a basic RAG pipeline, uses defaults
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()

nest_asyncio.apply()

prediction_dataset = await rag_dataset.amake_predictions_with(
    query_engine=query_engine, show_progress=True
)

