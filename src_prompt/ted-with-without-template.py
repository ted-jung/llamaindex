# ===========================================================================
# Basic prompt
# Created: 31, Jan 2025
# Updated: 
# Writer: Ted, Jung
# Description: Answering without tempate and then with template
#              What differences are there?
# ===========================================================================


from llama_index.core import (
    PromptTemplate, 
    Settings,
    VectorStoreIndex,
)
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.readers import SimpleDirectoryReader


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# llm = Ollama(model="llama3.2", request_timeout=720.0)
llm = OpenAI(model="gpt-4o-mini")
Settings.llm = llm


# Define tempate (one )
text_qa_template_str = (
    "Context information is"
    " below.\n---------------------\n{context_str}\n---------------------\nUsing"
    " both the context information and also using your own knowledge, answer"
    " the question: {query_str}\nIf the context isn't helpful, you can also"
    " answer the question on your own.\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

refine_template_str = (
    "The original question is as follows: {query_str}\nWe have provided an"
    " existing answer: {existing_answer}\nWe have the opportunity to refine"
    " the existing answer (only if needed) with some more context"
    " below.\n------------\n{context_msg}\n------------\nUsing both the new"
    " context and your own knowledge, update or repeat the existing answer.\n"
)
refine_template = PromptTemplate(refine_template_str)



# Load data
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(documents)


# Before adding tempate
print(index.as_query_engine(llm=llm).query("Who is Joe Biden?"))


# After adding template
print(index.as_query_engine(
    llm=llm,
    text_qa_template = text_qa_template,
    refine_template = refine_template
    ).query("Who is Joe Biden?")
)
