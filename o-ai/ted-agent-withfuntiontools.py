# ===========================================================================
# An agent with functiontools
# Date: 7, Feb 2025
# Writer: Ted, Jung
# Description: 
#        1. Define funtions
#        2. Add functions in a functiontool
#        3. Create objectindex with tool in step 2
#        4. Create an agent with turning objectindex to retriever and do a query
# ===========================================================================


import json
from typing import Sequence

from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.agent.openai import OpenAIAgent


# Define Functions (3 functions over here)
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def useless(a: int, b: int) -> int:
    """Toy useless function."""
    pass



# Add pre-defined functions in a functiontool via FunctionTool

multiply_tool = FunctionTool.from_defaults(fn=multiply, name="multiply")
useless_tools = [
    FunctionTool.from_defaults(fn=useless, name=f"useless_{str(idx)}")
    for idx in range(28)
]
add_tool = FunctionTool.from_defaults(fn=add, name="add")

all_tools = [multiply_tool] + [add_tool] + useless_tools
all_tools_map = {t.metadata.name: t for t in all_tools}



# Define an "object"(tool objects) index over these tools
# handle serialiation to/from the object, and use an underying index

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

agent = OpenAIAgent.from_tools(
    tool_retriever=obj_index.as_retriever(similarity_top_k=2), verbose=True
)

agent.chat("What's 212 multiplied by 122? Make sure to use Tools")

agent.chat("What's 212 added to 122 ? Make sure to use Tools")