# ===========================================================================
# Workflow, event Stream
# Create Date: 31, Jan 2025
# Update Date: 1, Feb 2025
# Writer: Ted, Jung
# Description: 
# ===========================================================================


from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
    draw_all_possible_flows,
)
import asyncio

from llama_index.core import Settings 
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=720.0)

Settings.llm = Ollama(model="llama3.2", request_timeout=720.0)


class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str
    response: str


class ProgressEvent(Event):
    msg: str
    
    

class MyWorkflow(Workflow):
    
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening"))
        return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        generator = await llm.astream_complete(
            "Please give me the first 3 paragraphs of Moby Dick, a book in the public domain."
        )
        async for response in generator:
            # Allow the workflow to stream this piece of response
            ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
        return SecondEvent(
            second_output="Second step complete, full response attached",
            response=str(response),
        )

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step three is happening"))
        return StopEvent(result="Workflow complete.")
    
    

async def main():
    w = MyWorkflow(timeout=320, verbose=True)
    handler = w.run(first_input="Start the workflow.")

    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.msg, end= "")

    final_result = await handler
    print("Final result", final_result)

    draw_all_possible_flows(MyWorkflow, filename="streaming_workflow.html")


if __name__ == "__main__":
    asyncio.run(main())