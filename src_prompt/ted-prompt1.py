# Basic prompt
# Date: 20, Jan 2025
# Writer: Ted, Jung
# Description: it is important to clarify on user's question and make exact answer
#              There are three ways to handle prompt with value that i wish to pass

 
from llama_index.core import PromptTemplate, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=720.0)
Settings.llm = llm


# 1. It is a partial prompt
qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Please write the answer in the style of {tone_name}
Query: {query_str}
Answer: \
"""

prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
partial_prompt_tmpl = prompt_tmpl.partial_format(tone_name="Shakespeare")

fmt_prompt = partial_prompt_tmpl.format(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
print(fmt_prompt)


# 2. Prompt template and variable mapping
# NOTE: here notice we use `my_context` and `my_query` as template variables

qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{my_context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {my_query}
Answer: \
"""

template_var_mappings = {"context_str": "my_context", "query_str": "my_query"}

prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str, template_var_mappings=template_var_mappings
)

fmt_prompt = prompt_tmpl.format(
    context_str="In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters",
    query_str="How many params does llama 2 have",
)
print(fmt_prompt)



# 3. Prompt Function Mapping
qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \
"""


def format_context_fn(**kwargs):
    # format context with bullet points
    context_list = kwargs["context_str"].split("\n\n")
    fmtted_context = "\n\n".join([f"- {c}" for c in context_list])
    return fmtted_context


prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str, function_mappings={"context_str": format_context_fn}
)

context_str = """\
In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters.

Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases.

Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models.
"""

fmt_prompt = prompt_tmpl.format(
    context_str=context_str, query_str="How many params does llama 2 have"
)
print(fmt_prompt)