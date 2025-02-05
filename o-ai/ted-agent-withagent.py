# ===========================================================================
# OpenAIAgent with QueryPlanTool and Tools
# Date: 5, Feb 2025
# Writer: Ted, Jung
# Description: other way how to handle limited length of description
# ===========================================================================

from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex,
    Settings,
    get_response_synthesizer,
)
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import QueryPlanTool

from llama_index.core.tools.types import ToolMetadata


llm = OpenAI(temperature=0, model="gpt-4o-mini")

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

march_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_march_2022.pdf"]
).load_data()
june_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_june_2022.pdf"]
).load_data()
sept_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_sept_2022.pdf"]
).load_data()


mar_index = VectorStoreIndex.from_documents(march_2022)
jun_index = VectorStoreIndex.from_documents(june_2022)
sep_index = VectorStoreIndex.from_documents(sept_2022)


mar_engine = mar_index.as_query_engine(similarity_top_k=3, llm=llm)
jun_engine = jun_index.as_query_engine(similarity_top_k=3, llm=llm)
sep_engine = sep_index.as_query_engine(similarity_top_k=3, llm=llm)


description_10q_general = """\
A Form 10-Q is a quarterly report required by the SEC for publicly traded companies,
providing an overview of the company's financial performance for the quarter.
It includes unaudited financial statements (income statement, balance sheet,
and cash flow statement) and the Management's Discussion and Analysis (MD&A),
where management explains significant changes and future expectations.
The 10-Q also discloses significant legal proceedings, updates on risk factors,
and information on the company's internal controls. Its primary purpose is to keep
investors informed about the company's financial status and operations,
enabling informed investment decisions."""

description_10q_specific = (
    "This 10-Q provides Uber quarterly financials ending"
)


query_tool_sept = QueryEngineTool.from_defaults(
    query_engine=sep_engine,
    name="sept_2022",
    description=f"{description_10q_general} {description_10q_specific} September 2022",
)
query_tool_june = QueryEngineTool.from_defaults(
    query_engine=jun_engine,
    name="june_2022",
    description=f"{description_10q_general} {description_10q_specific} June 2022",
)
query_tool_march = QueryEngineTool.from_defaults(
    query_engine=mar_engine,
    name="march_2022",
    description=f"{description_10q_general} {description_10q_specific} March 2022",
)

# print(len(query_tool_sept.metadata.description))
# print(len(query_tool_june.metadata.description))
# print(len(query_tool_march.metadata.description))



query_engine_tools = [query_tool_sept, query_tool_june, query_tool_march]

response_synthesizer = get_response_synthesizer()
query_plan_tool = QueryPlanTool.from_defaults(
    query_engine_tools=query_engine_tools,
    response_synthesizer=response_synthesizer,
)


# It could make an error regarding the tool description doesn't allow
# to exceed over 1024 characters (in that case, use prompt or shorten it)

# openai_tool = query_plan_tool.metadata.to_openai_tool()

introductory_tool_description_prefix = """\
This is a query plan tool that takes in a list of tools and executes a \
query plan over these tools to answer a query. The query plan is a DAG of query nodes.

Given a list of tool names and the query plan schema, you \
can choose to generate a query plan to answer a question.

The tool names and descriptions will be given alongside the query.
"""

# Modify metadata to only include the general query plan instructions
new_metadata = ToolMetadata(
    introductory_tool_description_prefix,
    query_plan_tool.metadata.name,
    query_plan_tool.metadata.fn_schema,
)
query_plan_tool.metadata = new_metadata
# print(query_plan_tool.metadata)


agent = OpenAIAgent.from_tools(
    [query_plan_tool],
    max_function_calls=10,
    llm=llm,
    verbose=True,
)

query = "What were the risk factors in sept 2022?"


# Reconstruct concatenated query engine tool descriptions
tools_description = "\n\n".join(
    [
        f"Tool Name: {tool.metadata.name}\n"
        + f"Tool Description: {tool.metadata.description} "
        for tool in query_engine_tools
    ]
)

# Concatenate tool descriptions and query
query_planned_query = f"{tools_description}\n\nQuery: {query}"
query_planned_query


response = agent.query(query_planned_query)

print(response)

