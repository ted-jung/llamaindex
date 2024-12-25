from langchain_ollama import OllamaEmbeddings

embeddings_model = OllamaEmbeddings(model="llama3.2")
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(embeddings)

len(embeddings), len(embeddings[0])

for _ in embeddings:
    print("=======================================================================")
    print(embeddings[0],len(embeddings[0]))
    
    
    
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print(embedded_query[:5])