from llama_index.core import GPTVectorStoreIndex, Document 

# Load your data 

documents = [ 
    Document("Product 1 Description", "This is a high-quality product..."), 
    Document("User Reviews", "The product is amazing! ..."), 
    # More product data and reviews 
] 

# Create an index 

index = GPTVectorStoreIndex(documents)  