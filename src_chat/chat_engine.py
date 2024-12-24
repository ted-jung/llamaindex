from llama_index.llms.ollama import Ollama

from llama_index.core.chat_engine import SimpleChatEngine

llm = Ollama(model="llama3.2", request_timeout=720.0)
chat_engine = SimpleChatEngine.from_defaults(llm=llm)

response = chat_engine.chat("Hello")

print(response)

response = chat_engine.chat("What did steve jobs do growing up?")
print(response)

chat_engine.chat_repl()