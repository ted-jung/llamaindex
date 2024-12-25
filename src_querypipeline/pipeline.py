# pipeline (Prompt + LLM)

from llama_index.core.query_pipeline import QueryPipeline

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)
from llama_index.core import set_global_handler as sgh
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
#Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
llm = Ollama(model="llama3.2", request_timeout=360.0)

# setup Arize Phoenix for logging/observability
import phoenix as px
import os

px.launch_app()
#import llama_index.core

sgh("arize_phoenix")
current_dir = os.getcwd()
reader = SimpleDirectoryReader(f"{current_dir}/paul_graham")
docs = reader.load_data()


if not os.path.exists("storage"):
    index = VectorStoreIndex.from_documents(docs)
    # save index to disk
    index.set_index_id("vector_index")
    index.storage_context.persist(f"{current_dir}/storage")
else:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=f"{current_dir}/storage")
    # load index
    index = load_index_from_storage(storage_context, index_id="vector_index")
    
    
# try chaining basic prompts
prompt_str = "Please generate related movies to {movie_name}"
prompt_tmpl = PromptTemplate(prompt_str)

p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)

output = p.run(movie_name="The Departed")
print(str(output))


# output, intermediates = p.run_with_intermediates(movie_name="The Departed")

from typing import List
from pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser


class Movie(BaseModel):
    """Object representing a single movie."""

    name: str = Field(..., description="Name of the movie.")
    year: int = Field(..., description="Year of the movie.")


class Movies(BaseModel):
    """Object representing a list of movies."""

    movies: List[Movie] = Field(..., description="List of movies.")


output_parser = PydanticOutputParser(Movies)
json_prompt_str = """\
Please generate related movies to {movie_name}. Output with the following JSON format: 
"""
json_prompt_str = output_parser.format(json_prompt_str)

# add JSON spec to prompt template
json_prompt_tmpl = PromptTemplate(json_prompt_str)

p = QueryPipeline(chain=[json_prompt_tmpl, llm, output_parser], verbose=True)
#output = p.run(movie_name="Toy Story")

#print(str(output))



prompt_str = "Please generate related movies to {movie_name}"
prompt_tmpl = PromptTemplate(prompt_str)
# let's add some subsequent prompts for fun
prompt_str2 = """\
Here's some text:

{text}

Can you rewrite this with a summary of each movie?
"""
prompt_tmpl2 = PromptTemplate(prompt_str2)
llm_c = llm.as_query_component(streaming=True)

p = QueryPipeline(
    chain=[prompt_tmpl, llm_c, prompt_tmpl2, llm_c], verbose=True
)
# p = QueryPipeline(chain=[prompt_tmpl, llm_c], verbose=True)

output = p.run(movie_name="The Dark Knight")
for o in output:
    print(o.delta, end="")
    

p = QueryPipeline(
    chain=[
        json_prompt_tmpl,
        llm.as_query_component(streaming=True),
        output_parser,
    ],
    verbose=True,
)
output = p.run(movie_name="Toy Story")
print(output)