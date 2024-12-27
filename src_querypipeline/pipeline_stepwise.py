from dotenv import load_dotenv
load_dotenv() 
from llama_index.core.query_pipeline import QueryPipeline
import os
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.response_synthesizers import TreeSummarize

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
#Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
llm = Ollama(model="llama3.2", request_timeout=360.0)

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
    
    
prompt_str = "Please generate related movies to {movie_name}"
prompt_tmpl = PromptTemplate(prompt_str)
p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)

run_state = p.get_run_state(movie_name="The Departed")

next_module_keys = p.get_next_module_keys(run_state)

while True:
    for module_key in next_module_keys:
        # get the module and input
        module = run_state.module_dict[module_key]
        module_input = run_state.all_module_inputs[module_key]

        # run the module
        output_dict = module.run_component(**module_input)

        # process the output
        p.process_component_output(
            output_dict,
            module_key,
            run_state,
        )

    # get the next module keys
    next_module_keys = p.get_next_module_keys(
        run_state,
    )

    # if no more modules to run, break
    if not next_module_keys:
        run_state.result_outputs[module_key] = output_dict
        break

# the final result is at `module_key`
# it is a dict of 'output' -> ChatResponse object in this case
print(run_state.result_outputs[module_key]["output"].message.content)