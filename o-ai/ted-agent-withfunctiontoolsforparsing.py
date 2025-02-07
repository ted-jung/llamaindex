# ===========================================================================
# An agent with functiontools and tool call parser
# Date: 7, Feb 2025
# Writer: Ted, Jung
# Description: 
#        1. Define a few funtions
#        2. Add functions in a functiontool (callable functions used by LLM)
#        3. Create objectindex with [tool, class object, etc] in step 2
#        4. Create an agent with parser via OpenAIToolCall (json passed as a result)
# ===========================================================================


import re
import nest_asyncio
import json

from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from typing import Dict
from llama_index.llms.openai.utils import OpenAIToolCall
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-4o-mini")

nest_asyncio.apply()


# Define two functions(caculation)
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)


# The same parser is available as
# from llama_index.agent.openai import advanced_tool_call_parser


def custom_tool_call_parser(tool_call: OpenAIToolCall) -> Dict:
    r"""Parse tool calls that are not standard json.
    Also parses tool calls of the following forms:
    variable = \"\"\"Some long text\"\"\"
    variable = "Some long text"'
    variable = '''Some long text'''
    variable = 'Some long text'
    """
    arguments_str = tool_call.function.arguments
    if len(arguments_str.strip()) == 0:
        # OpenAI returns an empty string for functions containing no args
        return {}
    try:
        tool_call = json.loads(arguments_str)
        if not isinstance(tool_call, dict):
            raise ValueError("Tool call must be a dictionary")
        return tool_call
    except json.JSONDecodeError as e:
        # pattern to match variable names and content within quotes
        pattern = r'([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*["\']+(.*?)["\']+'
        match = re.search(pattern, arguments_str)

        if match:
            variable_name = match.group(1)  # This is the variable name
            content = match.group(2)  # This is the content within the quotes
            return {variable_name: content}
        raise ValueError(f"Invalid tool call: {e!s}")




llm = OpenAI(model="gpt-4o-mini")
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True,
    tool_call_parser=custom_tool_call_parser,
)


response = agent.chat("What is (121 * 3) + (42 - 10)?")
print(str(response))