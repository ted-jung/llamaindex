# ===========================================================================
# Transform, 
# Date: 23, Jan 2025
# Writer: Ted, Jung
# Description: How to 
# ===========================================================================


from IPython.display import Markdown, display
from llama_index.core import (Settings)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.query.query_transform import HyDEQueryTransform


from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.question_gen.openai import OpenAIQuestionGenerator


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.3", request_timeout=720.0)



# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))


query_gen_str = """\
You are a helpful assistant that generates multiple search queries based on a \
single input query. Generate {num_queries} search queries, one on each line, \
related to the following input query:
Query: {query}
Queries:
"""
query_gen_prompt = PromptTemplate(query_gen_str)

# llm = OpenAI(model="gpt-3.5-turbo")
llm = Ollama(model="llama3.3", request_timeout=720.0)


def generate_queries(query: str, llm, num_queries: int = 4):
    response = llm.predict(
        query_gen_prompt, num_queries=num_queries, query=query
    )
    # assume LLM proper put each query on a newline
    queries = response.split("\n")
    queries_str = "\n".join(queries)
    print(f"Generated queries:\n{queries_str}")
    return queries



queries = generate_queries("What happened at Interleaf and Viaweb?", llm)




hyde = HyDEQueryTransform(include_original=True)

query_bundle = hyde.run("What is Bel?")

new_query.custom_embedding_strs



# Create sub-questions
question_gen = OpenAIQuestionGenerator.from_defaults(llm=llm)
display_prompt_dict(question_gen.get_prompts())

