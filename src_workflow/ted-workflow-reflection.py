# =============================================================================
# Reflection Workflow (plain text -> class (structured))
# Created: 13, Feb 2025
# Updated: 13, Feb 2025
# Writer: Ted, Jung
# Description: Workflow to provide reliable structured outputs
#              through retries and reflection on mistakes.
# =============================================================================


import asyncio
import json


from pydantic import BaseModel
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.deepseek import DeepSeek

from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    Context,
    step,
    Event,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm1 = OpenAI(model="gpt-4o-mini")
llm2 = Ollama(model="llama3.2")
llm3 = DeepSeek(model="deepseek-r1")

# Two classes
# to validate the structured output of an LLM
# Loop (generate -> validate -> error feedback -> generate ->) until it is valid

class ExtractionDone(Event):
    output: str
    passage: str


class ValidationErrorEvent(Event):
    error: str
    wrong_output: str
    passage: str


# Define pydantic model "Car" inherit from BaseModel
# for validating (i.g, model_validate_json(), model_json_schema() )

class Car(BaseModel):
    brand: str
    model: str
    power: int


class CarCollection(BaseModel):
    cars: list[Car]


# template for extraction with placeholders(passage, schema)
EXTRACTION_PROMPT = """
Context information is below:
---------------------
{passage}
---------------------

Given the context information and not prior knowledge, create a JSON object from the information in the context.
The JSON object must follow the JSON schema:
{schema}

"""

REFLECTION_PROMPT = """
You already created this output previously:
---------------------
{wrong_answer}
---------------------

This caused the JSON decode error: {error}

Try again, the response must contain only valid JSON code. Do not add any sentence before or after the JSON object.
Do not repeat the schema.
"""


class ReflectionWorkflow(Workflow):
    max_retries: int = 3

    @step
    async def extract(self, ctx: Context, ev: StartEvent | ValidationErrorEvent) -> StopEvent | ExtractionDone:
        current_retries = await ctx.get("retries", default=0)
        if current_retries >= self.max_retries:
            return StopEvent(result="Max retries reached")
        else:
            await ctx.set("retries", current_retries + 1)

        if isinstance(ev, StartEvent):
            passage = ev.get("passage")
            if not passage:
                return StopEvent(result="Please provide some text in input")
            reflection_prompt = ""
        elif isinstance(ev, ValidationErrorEvent):
            passage = ev.passage
            reflection_prompt = REFLECTION_PROMPT.format(
                wrong_answer=ev.wrong_output, error=ev.error
            )

        # llm = DeepSeek(model="deepseek-r1", request_timeout=720)
        # llm = Ollama(model="llama3.3", request_timeout=720)
        llm = OpenAI(model="gpt-4o-mini", request_timeout=30)
        prompt = EXTRACTION_PROMPT.format(
            passage=passage, schema=CarCollection.model_json_schema()
        )
        if reflection_prompt:
            prompt += reflection_prompt

        # asynchronously wait for the response from the language model, 
        # allowing other tasks to run concurrently if used within an async environment.
        output = await llm.acomplete(prompt)

        return ExtractionDone(output=str(output), passage=passage)

    @step
    async def validate(self, ev: ExtractionDone) -> StopEvent | ValidationErrorEvent:
        try:
            CarCollection.model_validate_json(ev.output)
        except Exception as e:
            print("Validation failed, retrying...")
            return ValidationErrorEvent(
                error=str(e), wrong_output=ev.output, passage=ev.passage
            )

        return StopEvent(result=ev.output)


async def ted():
    w = ReflectionWorkflow(timeout=120, verbose=True)

    # Run the workflow
    ret = await w.run(
        passage="I own two cars: a Fiat Panda with 45Hp and a Honda Civic with 330Hp."
    )

    print(ret)



if __name__ == "__main__":
    asyncio.run(ted())