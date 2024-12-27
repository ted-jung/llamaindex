from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from pathlib import Path

current_dir = Path.cwd()

documents = SimpleDirectoryReader(
    input_files=["data/paul_graham_essay.txt"]
).load_data()

print(documents[0].text)