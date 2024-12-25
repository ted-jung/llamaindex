from langchain_ollama import OllamaEmbeddings
from llama_index.core import Settings

# Settings.embeddings_model = OllamaEmbeddings(model="llama3.2")
Settings.embed_model = OllamaEmbeddings(model="llama3.2")

embeddings = Settings.embed_model.get_text_embedding("It is raining cats and dogs here!")

print(embeddings)