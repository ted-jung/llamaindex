# ===========================================================================
# An agent with the defined plan tool
# Date: 22, Jan 2025
# Writer: Ted, Jung
# Description:  designed to enhance the capabilities of AI agents
#               by enabling them to plan and execute a series of actions
#               to answer complex user queries.
# ===========================================================================

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import QueryPlanTool
from llama_index.core import get_response_synthesizer
from llama_index.agent.openai import OpenAIAgent


# Define a llm
llm = OpenAI(temperature=0, model="gpt-4o-mini")

# Load data, Turn data into Documents & Nodes
march_2022 = SimpleDirectoryReader(input_files=["./data/10q/uber_10q_march_2022.pdf"]).load_data()
june_2022 = SimpleDirectoryReader(input_files=["./data/10q/uber_10q_june_2022.pdf"]).load_data()
sept_2022 = SimpleDirectoryReader(input_files=["./data/10q/uber_10q_sept_2022.pdf"]).load_data()

# Build an Index
march_index = VectorStoreIndex.from_documents(march_2022)
june_index = VectorStoreIndex.from_documents(june_2022)
sept_index = VectorStoreIndex.from_documents(sept_2022)

# Turn the index to an Engine
march_engine = march_index.as_query_engine(similarity_top_k=3, llm=llm)
june_engine = june_index.as_query_engine(similarity_top_k=3, llm=llm)
sept_engine = sept_index.as_query_engine(similarity_top_k=3, llm=llm)


# Build a QueryEngineTool with engines
query_tool_sept = QueryEngineTool.from_defaults(
    query_engine=sept_engine,
    name="sept_2022",
    description=(
        f"Provides information about Uber quarterly financials ending"
        f" September 2022"
    ),
)
query_tool_june = QueryEngineTool.from_defaults(
    query_engine=june_engine,
    name="june_2022",
    description=(
        f"Provides information about Uber quarterly financials ending June"
        f" 2022"
    ),
)
query_tool_march = QueryEngineTool.from_defaults(
    query_engine=march_engine,
    name="march_2022",
    description=(
        f"Provides information about Uber quarterly financials ending March"
        f" 2022"
    ),
)

response_synthesizer = get_response_synthesizer()

# Put the queryEngineTool into QueryPlanTool with synthesize for generating response
query_plan_tool = QueryPlanTool.from_defaults(
    query_engine_tools=[query_tool_sept, query_tool_june, query_tool_march],
    response_synthesizer=response_synthesizer,
)

query_plan_tool.metadata.to_openai_tool()


# Finally, create an agent with the defined plan tool
agent = OpenAIAgent.from_tools(
    [query_plan_tool],
    max_function_calls=10,
    llm=OpenAI(temperature=0, model="gpt-4-0613"),
    verbose=True,
)

# Do a query
response = agent.query("What were the risk factors in sept 2022?")
print(response)