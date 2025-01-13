from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
    draw_all_possible_flows,
    draw_most_recent_execution,
)
import asyncio

from llama_index.core import Settings 
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=720.0)

Settings.llm = Ollama(model="llama3.2", request_timeout=720.0)


class JokeEvent(Event):
    joke: str
    
class CriticEvent(Event):
    critic_output: str

class ProgressEvent(Event):
    msg: str

class SecondEvent(Event):
    second_output: str
    response: str


class JokeFlow(Workflow):
    @step
    async def generate_joke(self, ctx: Context, ev: StartEvent) -> JokeEvent:
        topic = ev.topic
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening---------------------------------\n"))
        prompt = f"Write your best joke about {topic}."

        generator = await llm.astream_complete(prompt)
        async for response in generator:
            # Allow the workflow to stream this piece of response
            ctx.write_event_to_stream(ProgressEvent(msg=response.delta)) 
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ctx: Context, ev: JokeEvent) -> CriticEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step critic joke is happening-------------------------\n"))
        joke = ev.joke
        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        
        generator = await llm.astream_complete(prompt)
        async for response in generator:
            # Allow the workflow to stream this piece of response
            ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
        return CriticEvent(critic_output=str(response))

    @step
    async def step_two(self, ctx: Context, ev: CriticEvent) -> SecondEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step step two is happening-----------------------------\n"))
        
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
    async def empty_joke(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step final is happening---------------------------------\n"))
        return StopEvent(result="Workflow complete.")


async def main():
    w = JokeFlow(timeout=720, verbose=True)
    handler = w.run(topic="Ted who loves analog likes AI")

    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.msg, end="")

    final_result = await handler
    print("Final result", final_result)
    
    draw_most_recent_execution(w, filename="workflow_with_stream.html")


if __name__ == "__main__":
    asyncio.run(main())