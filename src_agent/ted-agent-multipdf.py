# Multi-PDF agent using Query pipelines and Hyde
# Date: 15, Jan 2025
# Writer: Ted, Jung
# Description: Agents, reasoning with multiple tools.
#       RAG pipeline with HyDE over a document


import os
import logging
import sys
import phoenix as px
import llama_index.core

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.core.tools import BaseTool, QueryEngineTool, ToolMetadata
from llama_index.core.callbacks import CallbackManager

from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from IPython.display import Markdown, display

from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.agent import (
    Task, 
    AgentChatResponse, 
    ReActChatFormatter, 
    QueryPipelineAgentWorker, 
    FnAgentWorker
)
from llama_index.core.query_pipeline import (
    AgentInputComponent,
    AgentFnComponent,
    CustomAgentComponent,
    QueryComponent,
    ToolRunnerComponent,
    InputComponent,
    Link,
)

from llama_index.core.llms import MessageRole, ChatMessage, ChatResponse
from typing import Dict, Any, Optional, Tuple, List, cast, Set


from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.types import Task


callback_manager = CallbackManager()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=720.0)
Settings.llm = llm
Settings.callback_manager = callback_manager

import phoenix as px
px.launch_app()
llama_index.core.set_global_handler("arize_phoenix")

try:
    storage_context = StorageContext.from_defaults(persist_dir="./src_agent/storage/lyft")
    lyft_index = load_index_from_storage(storage_context)
    
    storage_context = StorageContext.from_defaults(persist_dir="./src_agent/storage/uber")
    uber_index = load_index_from_storage(storage_context)
    
    index_loaded = True
except Exception as e:
    print ({e})
    index_loaded = False
    

if not index_loaded:
    # load data
    lyft_docs = SimpleDirectoryReader(input_files=["./data/pdf/lyft/lyft_2021.pdf"]).load_data()
    uber_docs = SimpleDirectoryReader(input_files=["./data/pdf/uber/uber_2021.pdf"]).load_data()

    # build index
    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    uber_index = VectorStoreIndex.from_documents(uber_docs)

    # persist index (no directory? it creats it)
    lyft_index.storage_context.persist(persist_dir="./src_agent/storage/lyft")
    uber_index.storage_context.persist(persist_dir="./src_agent/storage/uber")
    
    
lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)


hyde = HyDEQueryTransform(include_original=True)
lyft_hyde_query_engine = TransformQueryEngine(lyft_engine, hyde)
uber_hyde_query_engine = TransformQueryEngine(uber_engine, hyde)


query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_hyde_query_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_hyde_query_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

## Agent Input Component
## This is the component that produces agent inputs to the rest of the components
## Can also put initialization logic here.
def agent_input_fn(task: Task, state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent input function.

    Returns:
        A Dictionary of output keys and values. If you are specifying
        src_key when defining links between this component and other
        components, make sure the src_key matches the specified output_key.

    """
    # initialize current_reasoning
    if "current_reasoning" not in state:
        state["current_reasoning"] = []
    reasoning_step = ObservationReasoningStep(observation=task.input)
    state["current_reasoning"].append(reasoning_step)
    return {"input": task.input}


agent_input_component = AgentInputComponent(fn=agent_input_fn)


# Define prompt function
def react_prompt_fn(
    task: Task, state: Dict[str, Any], input: str, tools: List[BaseTool]
) -> List[ChatMessage]:
    # Add input to reasoning
    chat_formatter = ReActChatFormatter()
    return chat_formatter.format(
        tools,
        chat_history=task.memory.get() + state["memory"].get_all(),
        current_reasoning=state["current_reasoning"],
    )


react_prompt_component = AgentFnComponent(
    fn=react_prompt_fn, partial_dict={"tools": query_engine_tools}
)



# Define agent output parser + Tool Pipeline
def parse_react_output_fn(
    task: Task, state: Dict[str, Any], chat_response: ChatResponse
):
    """Parse ReAct output into a reasoning step."""
    output_parser = ReActOutputParser()
    reasoning_step = output_parser.parse(chat_response.message.content)
    return {"done": reasoning_step.is_done, "reasoning_step": reasoning_step}


parse_react_output = AgentFnComponent(fn=parse_react_output_fn)


def run_tool_fn(
    task: Task, state: Dict[str, Any], reasoning_step: ActionReasoningStep
):
    """Run tool and process tool output."""
    tool_runner_component = ToolRunnerComponent(
        query_engine_tools, callback_manager=task.callback_manager
    )
    tool_output = tool_runner_component.run_component(
        tool_name=reasoning_step.action,
        tool_input=reasoning_step.action_input,
    )
    observation_step = ObservationReasoningStep(
        observation=str(tool_output["output"])
    )
    state["current_reasoning"].append(observation_step)
    # TODO: get output

    return {"response_str": observation_step.get_content(), "is_done": False}


run_tool = AgentFnComponent(fn=run_tool_fn)


def process_response_fn(
    task: Task, state: Dict[str, Any], response_step: ResponseReasoningStep
):
    """Process response."""
    state["current_reasoning"].append(response_step)
    response_str = response_step.response
    # Now that we're done with this step, put into memory
    state["memory"].put(ChatMessage(content=task.input, role=MessageRole.USER))
    state["memory"].put(
        ChatMessage(content=response_str, role=MessageRole.ASSISTANT)
    )

    return {"response_str": response_str, "is_done": True}


process_response = AgentFnComponent(fn=process_response_fn)


def process_agent_response_fn(
    task: Task, state: Dict[str, Any], response_dict: dict
):
    """Process agent response."""
    return (
        AgentChatResponse(response_dict["response_str"]),
        response_dict["is_done"],
    )


process_agent_response = AgentFnComponent(fn=process_agent_response_fn)



# build a query pipeline and let the agent have it
from llama_index.core.query_pipeline import QueryPipeline as QP

qp = QP(verbose=True)
qp.add_modules(
    {
        "agent_input": agent_input_component,
        "react_prompt": react_prompt_component,
        "llm": Ollama(model="llama3.2"),
        "react_output_parser": parse_react_output,
        "run_tool": run_tool,
        "process_response": process_response,
        "process_agent_response": process_agent_response,
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

# whether response processing or tool output processing, add link to final agent response
qp.add_link("process_response", "process_agent_response")
qp.add_link("run_tool", "process_agent_response")


# visualize the query pipeline
from pyvis.network import Network
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(qp.clean_dag)
print(net)
net.write_html("./agent_dag.html")


# Create an agent and start task to execute query pipeline
agent_worker = QueryPipelineAgentWorker(qp)
agent = agent_worker.as_agent(
    callback_manager=CallbackManager([]), verbose=True
)
task = agent.create_task(
    "What was Uber's Management's Report on Internal Control over Financial Reporting?"
)
step_output = agent.run_step(task.task_id)

print(step_output)