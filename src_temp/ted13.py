from llama_index.core import Settings

# tiktoken
import tiktoken

# Settings.tokenizer = tiktoken.encoding_for_model("llama3.2").encode

from transformers import AutoTokenizer

token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)
# Tokenize a text input
inputs = tokenizer("Hello, world!", return_tensors="pt")

print(inputs)