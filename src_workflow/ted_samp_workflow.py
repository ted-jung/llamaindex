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


class JokeFlow(Workflow):
    llm = Ollama(model="llama3.2", request_timeout=360.0)

    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        print("generated joke =========================")
        topic = ev.topic

        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        print("1------------------------------------")
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        print("criticise joke==========================")
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        print("2------------------------------------")
        return StopEvent(result=str(response))


async def main():
    # Draw all
    draw_all_possible_flows(JokeFlow, filename="joke_flow_all.html")

    # Draw an execution
    w = JokeFlow(timeout=120, verbose=False)
    result = await w.run(topic="Ted who loves analog likes AI")
    print(str(result))
    draw_most_recent_execution(w, filename="joke_flow_recent.html")
    
    
asyncio.run(main())