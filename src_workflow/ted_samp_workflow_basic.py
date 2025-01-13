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

# `pip install llama-index-llms-openai` if you don't already have it
from llama_index.llms.ollama import Ollama


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
    llm = Ollama(model="llama3.2", request_timeout=360.0)

    @step
    async def generate_joke(self, ctx: Context, ev: StartEvent) -> JokeEvent:
        topic = ev.topic
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening"))
        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ctx: Context, ev: JokeEvent) -> CriticEvent:
        joke = ev.joke
        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        
        response = await self.llm.acomplete(prompt)
        print("here===================")
        print(response)
        print("here===================")
        return CriticEvent(critic_output=str(response))

    @step
    async def step_two(self, ctx: Context, ev: CriticEvent) -> SecondEvent:
        generator = await self.llm.astream_complete(
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
        ctx.write_event_to_stream(ProgressEvent(msg="Step fourth is happening"))
        return StopEvent(result="Workflow complete.")


async def main():
    # Draw all
    draw_all_possible_flows(JokeFlow, filename="joke_flow_all.html")

    # Draw an execution
    w = JokeFlow(timeout=360, verbose=False)
    result = await w.run(topic="Ted who loves analog likes AI")
    print(str(result))
        
    draw_most_recent_execution(w, filename="joke_flow_recent.html")


asyncio.run(main())