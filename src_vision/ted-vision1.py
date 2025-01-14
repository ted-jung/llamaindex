import ollama

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': ['/Users/tedj/Ted-pjt/python_pjt/LlamaIndex/src_vision/image.png']
    }]
)

print(response)