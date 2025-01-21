# Agent using Query pipelines
# Date: 20, Jan 2025
# Writer: Ted, Jung
# Description: Agent and QueryPipeline as tool
#              NLSQLTableQueryEngine (text to sql) + Retry max
#              Result: llama3.2 < llama3.3  (understand well for the complex question)

from typing import Any, Dict, Tuple

from llama_index.core import (
    PromptTemplate,
    Response,
    Settings,
    SQLDatabase,
)
from llama_index.core.agent import AgentChatResponse, FnAgentWorker, Task
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.query_pipeline import (
    InputComponent,
    Link,
    StatefulFnComponent,
)
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
)
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from pyvis.network import Network
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

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.3", request_timeout=720.0)
Settings.llm = llm


engine = create_engine("sqlite:///./data/chinook/chinook.db")
sql_database = SQLDatabase(engine)

sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
    verbose=True,
)


def agent_input_fn(state: Dict[str, Any]) -> Dict:
    """Agent input function."""
    task = state["task"]
    state["convo_history"].append(f"User: {task.input}")
    convo_history_str = "\n".join(state["convo_history"]) or "None"
    return {"input": task.input, "convo_history": convo_history_str}


agent_input_component = StatefulFnComponent(fn=agent_input_fn)



retry_prompt_str = """\
You are trying to generate a proper natural language query given a user input.

This query will then be interpreted by a downstream text-to-SQL agent which
will convert the query to a SQL statement. If the agent triggers an error,
then that will be reflected in the current conversation history (see below).

If the conversation history is None, use the user input. If its not None,
generate a new SQL query that avoids the problems of the previous SQL query.

Input: {input}
Convo history (failed attempts): 
{convo_history}

New input: """
retry_prompt = PromptTemplate(retry_prompt_str)


validate_prompt_str = """\
Given the user query, validate whether the inferred SQL query and response from executing the query is correct and answers the query.

Answer with YES or NO.

Query: {input}
Inferred SQL query: {sql_query}
SQL Response: {sql_response}

Result: """
validate_prompt = PromptTemplate(validate_prompt_str)

MAX_ITER = 3


def agent_output_fn(
    state: Dict[str, Any], output: Response
) -> Tuple[AgentChatResponse, bool]:
    """Agent output component."""
    task = state["task"]
    print(f"> Inferred SQL Query: {output.metadata['sql_query']}")
    print(f"> SQL Response: {str(output)}")
    state["convo_history"].append(
        f"Assistant (inferred SQL query): {output.metadata['sql_query']}"
    )
    state["convo_history"].append(f"Assistant (response): {str(output)}")

    # run a mini chain to get response
    validate_prompt_partial = validate_prompt.as_query_component(
        partial={
            "sql_query": output.metadata["sql_query"],
            "sql_response": str(output),
        }
    )
    qp = QP(chain=[validate_prompt_partial, llm])
    validate_output = qp.run(input=task.input)

    state["count"] += 1
    is_done = False
    if state["count"] >= MAX_ITER:
        is_done = True
    if "YES" in validate_output.message.content:
        is_done = True

    return str(output), is_done


agent_output_component = StatefulFnComponent(fn=agent_output_fn)


qp = QP(
    modules={
        "input": agent_input_component,
        "retry_prompt": retry_prompt,
        "llm": llm,
        "sql_query_engine": sql_query_engine,
        "output_component": agent_output_component,
    },
    verbose=True,
)
qp.add_link(
    "input", "retry_prompt", dest_key="input", input_fn=lambda x: x["input"]
)
qp.add_link(
    "input",
    "retry_prompt",
    dest_key="convo_history",
    input_fn=lambda x: x["convo_history"],
)
qp.add_chain(["retry_prompt", "llm", "sql_query_engine", "output_component"])


# net = Network(notebook=True, cdn_resources="in_line", directed=True)
# net.from_nx(qp.dag)
# net.show("agent_dag3.html")



def run_agent_fn(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Run agent function."""
    task, qp = state["__task__"], state["query_pipeline"]
    # if first run, then set query pipeline state to initial variables
    if state["is_first"]:
        qp.set_state({"task": task, "convo_history": [], "count": 0})
        state["is_first"] = False

    # run the pipeline, get response
    response_str, is_done = qp.run()
    if is_done:
        state["__output__"] = response_str
    return state, is_done


agent = FnAgentWorker(
    fn=run_agent_fn,
    initial_state={"query_pipeline": qp, "is_first": True},
).as_agent()


response = agent.chat(
    "How many albums did the artist who wrote 'Restless and Wild' release? (answer should be non-zero)?"
)
print(str(response))