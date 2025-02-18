# ===========================================================================
# Agent as as Service (Agent + Workflow)
# Created: 18, Feb 2025
# Updated: 18, Feb 2025
# Writer: Ted, Jung
# Description: 
#   External Service (create agent having lot of tools)
# ===========================================================================


import asyncio
import os


from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from toolhouse import Toolhouse, Provider



# Set key to env
TOOLHOUSE_API_KEY = api_key=os.environ.get("TOOLHOUSE_API_KEY")


# Define llm (here OpenAI) and Configure toolhouse
llm = OpenAI(model="gpt-4o-mini", timeout=720.0)
th = Toolhouse(api_key=TOOLHOUSE_API_KEY, provider=Provider.OPENAI)
th.set_metadata("id", "tedjung")

# th = Toolhouse(api_key=os.environ.get("TOOLHOUSE_API_KEY"),provider=Provider.OPENAI)
# th.set_metadata("id", "openai_agent")
# th.set_metadata("timezone", 0)




# Define four events
class WebsiteContentEvent(Event):
    contents: str

class WebSearchEvent(Event):
    results: str

class RankingEvent(Event):
    results: str

class LogEvent(Event):
    msg: str


# ReAct will use bundle(from ToolHouse) as a tool with llm (create a bundle before using it)
# CMB will be used to store history of actions to keep the context

class SalesRepWorkflow(Workflow):
    agent = ReActAgent(
        tools=th.get_tools(bundle="Ted-Bundle-10ed9a04"),
        llm=llm,
        memory=ChatMemoryBuffer.from_defaults(),
        verbose=True
    )

    @step
    async def get_company_info(self, ctx: Context, ev: StartEvent) -> WebsiteContentEvent:
        ctx.write_event_to_stream(
            LogEvent(msg=f"Getting the contents of {ev.url}…")
        )
        prompt = f"Get the page contents of {ev.url}, then summarize its key value propositions in a few bullet points."
        contents = await self.agent.achat(prompt)
        
        return WebsiteContentEvent(contents=str(contents.response))

    @step
    async def find_prospects(self, ctx: Context, ev: WebsiteContentEvent) -> WebSearchEvent:
        ctx.write_event_to_stream(
            LogEvent(
                msg="Performing web searches to identify companies who can benefit from the business's offerings."
            )
        )
        prompt = """With that you know about the business, perform a web search to find 5 tech companies who may benefit
                 from the business's product. Only answer with the names of the companies you chose."""
        results = await self.agent.achat(prompt)
        return WebSearchEvent(results=str(results.response))

    @step
    async def select_best_company(self, ctx: Context, ev: WebSearchEvent) -> RankingEvent:
        ctx.write_event_to_stream(
            LogEvent(
                msg="Selecting the best company who can benefit from the business's offering…"
            )
        )
        prompt = """Select one company that can benefit from the business's product. Only use your knowledge to select the company.
                    Respond with just the name of the company. Do not use tools."""
        results = await self.agent.achat(prompt)
        ctx.write_event_to_stream(
            LogEvent(
                msg=f"The agent selected this company: {results.response}"
            )
        )
        return RankingEvent(results=str(results.response))

    @step
    async def prepare_email(self, ctx: Context, ev: RankingEvent) -> StopEvent:
        ctx.write_event_to_stream(
            LogEvent(msg="Drafting a short email for sales outreach…")
        )
        prompt = "Draft a short cold sales outreach email for the company you picked. Do not use tools."
        email = await self.agent.achat(prompt)
        ctx.write_event_to_stream(
            LogEvent(msg=f"Here is the email: {email.response}")
        )
        return StopEvent(result=str(email.response))
    



async def main():
    workflow = SalesRepWorkflow(timeout=None)
    handler = await workflow.run(url="https://toolhouse.ai")
    async for event in handler.stream_events():
        if isinstance(event, LogEvent):
            print(event.msg)



if __name__ == "__main__":
    asyncio.run(main())