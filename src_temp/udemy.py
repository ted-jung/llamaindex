from openai import OpenAI
MODEL = "llama3.2"
openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

response = openai.chat.completions.create(
 model=MODEL,
 messages=[{"role": "user", "content": "What is 2 + 2?"}]
)

print(response.choices[0].message.content)

response2 = openai.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What are the first programs Paul Graham tried writing?"}]
)

print(response2.choices[0].message.content)