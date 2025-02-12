# ===========================================================================
# Custom prompt
# Created: 12, Feb 2025
# Updated: 
# Writer: Ted, Jung
# Description: response mode on different engine
#              Each engine has a different template
#              can also apply a modified template to get the intended resposne 
# ===========================================================================


import os

from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    Settings,
    PromptTemplate,
)

from llama_index.core.query_engine import (
    RouterQueryEngine,
    FLAREInstructQueryEngine,
)
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.evaluation import FaithfulnessEvaluator, DatasetGenerator
from llama_index.core.postprocessor import LLMRerank

from IPython.display import Markdown, display

# setup sample router query engine
from llama_index.core.tools import QueryEngineTool


def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        print(Markdown(text_md))
        print(p.get_template())
        print(Markdown("<br><br>"))


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = OpenAI(model="gpt-4o-mini", request_timeout=720.0)
Settings.llm = llm


curr_dir = os.getcwd()
documents = SimpleDirectoryReader(f"{curr_dir}/data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents)



# 1. response mode (tree_summarize)
query_engine = index.as_query_engine(response_mode="tree_summarize")
prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)
response = query_engine.query("What did the author do growing up?")
print(str(response))


# Custom prompts using PromptTemplate, Update prompt and do a query again

new_summary_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in the style of a Shakespeare play.\n"
    "Query: {query_str}\n"
    "Answer: "
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

query_engine.update_prompts(
    {"response_synthesizer:summary_template": new_summary_tmpl}
)
prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)
response = query_engine.query("What did the author do growing up?")
print(str(response))


# Can access prompts from other modules
query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine, description="test description"
)

router_query_engine = RouterQueryEngine.from_defaults([query_tool])
prompts_dict = router_query_engine.get_prompts()
display_prompt_dict(prompts_dict)
response = router_query_engine.query("What did the author do growing up?")
print(response)





# 2. response mode (compact)
query_engine = index.as_query_engine(response_mode="compact")
prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)

response = query_engine.query("What did the author do growing up?")
print(str(response))



dataset_generator = DatasetGenerator.from_documents(documents)
prompts_dict = dataset_generator.get_prompts()
display_prompt_dict(prompts_dict)