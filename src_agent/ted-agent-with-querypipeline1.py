# Agent using Query pipelines 
# Date: 17, Jan 2025
# Writer: Ted, Jung
# Description: Agent and QueryPipeline as tool
#              NLSQLTableQueryEngine (text to sql)
#       ~~

import llama_index.core
import phoenix as px

# Define Agent Input Component
from typing import Any, Dict, List, Optional, Tuple, cast, Set

from pyvis.network import Network
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.tools import BaseTool
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser

from llama_index.core.agent import AgentChatResponse, Task
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.llms import MessageRole
from llama_index.core.query_pipeline import (
    InputComponent,
    Link,
    QueryComponent,
    StatefulFnComponent,
    ToolRunnerComponent,
)

from llama_index.core import (
    Settings,
    SQLDatabase
)
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.query_pipeline import QueryPipeline as QP
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool

from tabnanny import verbose

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    column,
    create_engine,
    select,
)


from llama_index.core.agent.types import Task
from llama_index.core.agent import FnAgentWorker


callback_manager = CallbackManager()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=720.0)
Settings.llm = llm
Settings.callback_manager = callback_manager

engine = create_engine("sqlite:///./data/chinook/chinook.db")
sql_database = SQLDatabase(engine)



px.launch_app()
llama_index.core.set_global_handler("arize_phoenix")


# ========================================
# Setup Text-to-SQL NL Query Engine / Tool
# ========================================

sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
    verbose=True,
)

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    name = "sql_tool",
    description=(
        "Useful for translating a natural language query into a SQL query"
    ),
)

# ================================================================================
# Setup ReAct Agent with Stateful Pipeline  (stateful agent)
#
# Two components(FnAgentWorker, StatefulFnComponent)
#
#       1. Input
#       2. ReAct prompt using LLM -> generate next action/tool or return response
#       3. Selected(tool/action), Call tool pipeline to execute + collect response
#       4. get response if response is generated
# ================================================================================

qp = QP(verbose=True)



# Input Component
## This is the component that produces agent inputs to the rest of the components
## Can also put initialization logic here.

def agent_input_fn(state: Dict[str, Any]) -> str:
    """Agent input function.

    Returns:
        A Dictionary of output keys and values. If you are specifying
        src_key when defining links between this component and other
        components, make sure the src_key matches the specified output_key.

    """
    task = state["task"]
    if len(state["current_reasoning"]) == 0:
        reasoning_step = ObservationReasoningStep(observation=task.input)
        state["current_reasoning"].append(reasoning_step)
    return task.input


agent_input_component = StatefulFnComponent(fn=agent_input_fn)



## Define Agent Prompt
## generate a ReAct prompt (output by LLM, parses into a structured object)
## define prompt function

def react_prompt_fn(
    state: Dict[str, Any], input: str, tools: List[BaseTool]
) -> List[ChatMessage]:
    task = state["task"]
    # Add input to reasoning
    chat_formatter = ReActChatFormatter()
    cur_prompt = chat_formatter.format(
        tools,
        chat_history=task.memory.get(),
        current_reasoning=state["current_reasoning"],
    )
    return cur_prompt


react_prompt_component = StatefulFnComponent(
    fn=react_prompt_fn, partial_dict={"tools": [sql_tool]}
)



## Define Agent Output Parser + Tool Pipeline
## output by LLM with decision tree
## given: 
##        action -> need to execute the specified tool with args -> process the output
##                  tool(name+action) calling via ToolRunnerComponent
##        answer -> process the output


def parse_react_output_fn(state: Dict[str, Any], chat_response: ChatResponse):
    """Parse ReAct output into a reasoning step."""
    output_parser = ReActOutputParser()
    reasoning_step = output_parser.parse(chat_response.message.content)
    return {"done": reasoning_step.is_done, "reasoning_step": reasoning_step}


parse_react_output = StatefulFnComponent(fn=parse_react_output_fn)


def run_tool_fn(state: Dict[str, Any], reasoning_step: ActionReasoningStep):
    """Run tool and process tool output."""
    task = state["task"]
    tool_runner_component = ToolRunnerComponent(
        [sql_tool], callback_manager=task.callback_manager
    )
    tool_output = tool_runner_component.run_component(
        tool_name=reasoning_step.action,
        tool_input=reasoning_step.action_input,
    )
    observation_step = ObservationReasoningStep(observation=str(tool_output))
    state["current_reasoning"].append(observation_step)
    # TODO: get output

    # return tuple of current output and False for is_done
    return observation_step.get_content(), False

run_tool = StatefulFnComponent(fn=run_tool_fn)


def process_response_fn(
    state: Dict[str, Any], response_step: ResponseReasoningStep
):
    """Process response."""
    state["current_reasoning"].append(response_step)
    return response_step.response, True

process_response = StatefulFnComponent(fn=process_response_fn)



## Stitch together Agent Query Pipeline
## agent_input -> react_prompt -> llm -> react_output

qp.add_modules(
    {
        "agent_input": agent_input_component,
        "react_prompt": react_prompt_component,
        "llm": llm,
        "react_output_parser": parse_react_output,
        "run_tool": run_tool,
        "process_response": process_response
    }
)

# link input to react prompt to parsed out response (either tool action/input or observation)
qp.add_chain(["agent_input", "react_prompt", "llm", "react_output_parser"])

# add conditional link from react output to tool call (if not done)
qp.add_link(
    "react_output_parser",
    "run_tool",
    condition_fn=lambda x: not x["done"],
    input_fn=lambda x: x["reasoning_step"],
)
# add conditional link from react output to final response processing (if done)
qp.add_link(
    "react_output_parser",
    "process_response",
    condition_fn=lambda x: x["done"],
    input_fn=lambda x: x["reasoning_step"],
)

net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(qp.clean_dag)
net.show("agent_dag2.html")



## Setup Agent Worker around Text-to-SQL Query Pipeline
##
## Custom agent implementation(simple python) => FnAgentWorker
## ReAct loop (<-query pipeline)
##           : right step at a given state

def run_agent_fn(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Run agent function."""
    task, qp = state["__task__"], state["query_pipeline"]
    # if first run, then set query pipeline state to initial variables
    if state["is_first"]:
        qp.set_state(
            {
                "task": task,
                "current_reasoning": [],
            }
        )
        state["is_first"] = False

    # no explicit input here, just run root node
    response_str, is_done = qp.run()
    # if done, store output and log to memory
    # a core memory module is available in the `task` variable. Of course you can log
    # and store your own memory as well
    state["__output__"] = response_str
    if is_done:
        task.memory.put_messages(
            [
                ChatMessage(content=task.input, role=MessageRole.USER),
                ChatMessage(content=response_str, role=MessageRole.ASSISTANT),
            ]
        )
    return state, is_done


agent = FnAgentWorker(
    fn=run_agent_fn,
    initial_state={"query_pipeline": qp, "is_first": True},
).as_agent()

## Run the Agent
## start task
task = agent.create_task(
    "What are some tracks from the artist AC/DC? Limit it to 3"
)

# step_output = agent.run_step(task.task_id)
# step_output = agent.run_step(task.task_id)

# print(step_output)
# print(step_output.is_last)
# response = agent.finalize_response(task.task_id)
# print(str(response))


# run this e2e
agent.reset()
response = agent.chat(
    "What are some tracks from the artist AC/DC? Limit it to 3"
)
print(str(response))

response = agent.finalize_response(task.task_id)

