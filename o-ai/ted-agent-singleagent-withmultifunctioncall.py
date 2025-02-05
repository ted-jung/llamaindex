# ===========================================================================
# SingleAgent-multiple functioncall
# Date: 5, Feb 2025
# Writer: Ted, Jung
# Description: function calling in parallel
# ===========================================================================

import asyncio
import nest_asyncio

from llama_index.core import (
    Settings,
)
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


llm = OpenAI(temperature=0, model="gpt-4o-mini")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

llm = OpenAI(model="gpt-3.5-turbo-1106")
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool], llm=llm, verbose=True
)

response = agent.chat("What is (121 * 3) + 42?")
print(str(response))


print(100*"+")

response = agent.stream_chat("What is (121 * 3) + 42?")
print(str(response))

print(100*"+")

nest_asyncio.apply()

async def ted_run1():
    response = await agent.achat("What is (121 * 3) + 42?")
    print(str(response))
    print(100*"+")


async def ted_run2():
    response = await agent.astream_chat("What is (121 * 3) + 42?")
    # response_gen = response.response_gen

    async for token in response.async_response_gen():
        print(token, end="")


if __name__ == "__main__":
    asyncio.run(ted_run1())

    asyncio.run(ted_run2())