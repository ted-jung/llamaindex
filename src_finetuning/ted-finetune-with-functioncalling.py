# ===========================================================================
# Fine tuning with function calling
# Date: 27, Jan 2025
# Writer: Ted, Jung
# Description:  base model + fine-tune with function call
#               distilling teacher model's output to improve student model
# ===========================================================================


from llama_index.program.openai import OpenAIPydanticProgram
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.finetuning.callbacks import OpenAIFineTuningHandler
from llama_index.core.callbacks import CallbackManager
from typing import List
from tqdm import tqdm
from llama_index.finetuning import OpenAIFinetuneEngine

from pydantic import Field
from typing import List

from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path

from llama_index.core import Settings

import os
import openai
import nest_asyncio

nest_asyncio.apply()
openai.api_key = os.environ["OPENAI_API_KEY"]


# Data Validation/Modeling/Reduced boilerplate code/Integration with other libraries

class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]


# Define CallbackManager to know(Callback, Event Handling, Custom Actions)
# Improved understanding, Enhanced debugging, Increased control, Improved monitoring

finetuning_handler = OpenAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])
llm = OpenAI(model="gpt-4o-mini", callback_manager=callback_manager)


prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
program = OpenAIPydanticProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    llm=llm,
    verbose=False,
)


# Log Input/Outputs
# NOTE: we need >= 10 movies to use OpenAI fine-tuning

movie_names = [
    "The Shining",
    "The Departed",
    "Titanic",
    # "Goodfellas",
    # "Pretty Woman",
    # "Home Alone",
    # "Caged Fury",
    # "Edward Scissorhands",
    # "Total Recall",
    # "Ghost",
    # "Tremors",
    # "RoboCop",
    # "Rocky V",
]

for movie_name in tqdm(movie_names):
    output = program(movie_name=movie_name)
    print(output.json())

finetuning_handler.save_finetuning_events("mock_finetune_songs.jsonl")

finetune_engine = OpenAIFinetuneEngine(
    "gpt-3.5-turbo",
    "mock_finetune_songs.jsonl",
    # start_job_id="<start-job-id>"  # if you have an existing job, can specify id here
    validate_json=False,  # openai validate json code doesn't support function calling yet
)

finetune_engine.finetune()
finetune_engine.get_current_job()

ft_llm = finetune_engine.get_finetuned_model(temperature=0.3)

ft_program = OpenAIPydanticProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    llm=ft_llm,
    verbose=False,
)

ft_program(movie_name="Goodfellas")


# class Citation(BaseModel):
#     """Citation class."""

#     author: str = Field(
#         ..., description="Inferred first author (usually last name"
#     )
#     year: int = Field(..., description="Inferred year")
#     desc: str = Field(
#         ...,
#         description=(
#             "Inferred description from the text of the work that the author is"
#             " cited for"
#         ),
#     )


# class Response(BaseModel):
#     """List of author citations.

#     Extracted over unstructured text.

#     """

#     citations: List[Citation] = Field(
#         ...,
#         description=(
#             "List of author citations (organized by author, year, and"
#             " description)."
#         ),
#     )


# loader = PyMuPDFReader()
# docs0 = loader.load(file_path=Path("./data/llama2.pdf"))

# doc_text = "\n\n".join([d.get_content() for d in docs0])
# metadata = {
#     "paper_title": "Llama 2: Open Foundation and Fine-Tuned Chat Models"
# }
# docs = [Document(text=doc_text, metadata=metadata)]

# chunk_size = 1024
# node_parser = SentenceSplitter(chunk_size=chunk_size)
# nodes = node_parser.get_nodes_from_documents(docs)



# finetuning_handler = OpenAIFineTuningHandler()
# callback_manager = CallbackManager([finetuning_handler])

# Settings.chunk_size = chunk_size

# gpt_4_llm = OpenAI(
#     model="gpt-4-0613", temperature=0.3, callback_manager=callback_manager
# )

# gpt_35_llm = OpenAI(
#     model="gpt-3.5-turbo-0613",temperature=0.3,callback_manager=callback_manager,
# )

# eval_llm = OpenAI(model="gpt-4-0613", temperature=0)