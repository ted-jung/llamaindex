from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# NOTE: lyft_index and uber_index are both SimpleVectorIndex instances
lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description="Provides information about Lyft financials for year 2021. "
            "Use a detailed plain text question as input to the tool.",
        ),
        return_direct=False,
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description="Provides information about Uber financials for year 2021. "
            "Use a detailed plain text question as input to the tool.",
        ),
        return_direct=False,
    ),
]

# initialize ReAct agent
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)