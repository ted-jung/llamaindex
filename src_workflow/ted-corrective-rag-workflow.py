# Corrective RAG with workflow (CRAG)
# Date: 14, Jan 2025
# Writer: Ted, Jung
# When: Enhance the robustness of language model generation by evaluating and 
#       Augmenting the relevance of retrieved documents through an evaluator and 
#       Large-scale web searches, 
#       Ensuring more accurate and reliable information is used in generation.

import os
import asyncio

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import draw_all_possible_flows

from llama_index.core.query_pipeline import QueryPipeline
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.base.base_retriever import BaseRetriever

from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

from llama_index.core import (
    VectorStoreIndex,
    Document,
    PromptTemplate,
    SummaryIndex,
    Settings,
    SimpleDirectoryReader,
)

from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)

from IPython.display import Markdown, display


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=720.0)
Settings.llm = llm

tavily_ai_api_key ="tvly-xxxx"



class PrepEvent(Event):
    """Prep event (prepares for retrieval)."""

    pass


class RetrieveEvent(Event):
    """Retrieve event (gets retrieved nodes)."""

    retrieved_nodes: list[NodeWithScore]


class RelevanceEvalEvent(Event):
    """Relevance evaluation event (gets results of relevance evaluation)."""

    relevant_results: list[str]


class TextExtractEvent(Event):
    """Text extract event. Extracts relevant text and concatenates."""

    relevant_text: str


class QueryEvent(Event):
    """Query event. Queries given relevant text and search text."""

    relevant_text: str
    search_text: str


DEFAULT_RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
    template="""As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

    Retrieved Document:
    -------------------
    {context_str}

    User Question:
    --------------
    {query_str}

    Evaluation Criteria:
    - Consider whether the document contains keywords or topics related to the user's question.
    - The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

    Decision:
    - Assign a binary score to indicate the document's relevance.
    - Use 'yes' if the document is relevant to the question, or 'no' if it is not.

    Please provide your binary score ('yes' or 'no') below to indicate the document's relevance to the user question."""
)

DEFAULT_TRANSFORM_QUERY_TEMPLATE = PromptTemplate(
    template="""Your task is to refine a query to ensure it is highly effective for retrieving relevant search results. \n
    Analyze the given input to grasp the core semantic intent or meaning. \n
    Original Query:
    \n ------- \n
    {query_str}
    \n ------- \n
    Your goal is to rephrase or enhance this query to improve its search performance. \n
    Ensure the revised query is concise and directly aligned with the intended search objective. \n
    Respond with the optimized query only:"""
)



class CorrectiveRAGWorkflow(Workflow):
    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Ingest step (for ingesting docs and initializing index)."""
        documents: list[Document] | None = ev.get("documents")

        if documents is None:
            return None

        index = VectorStoreIndex.from_documents(documents)

        return StopEvent(result=index)

    @step
    async def prepare_for_retrieval(self, ctx: Context, ev: StartEvent) -> PrepEvent | None:
        """Prepare for retrieval."""

        query_str: str | None = ev.get("query_str")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})

        if query_str is None:
            return None

        tavily_ai_apikey: str | None = ev.get("tavily_ai_apikey")
        index = ev.get("index")

        await ctx.set(
            "relevancy_pipeline",
            QueryPipeline(chain=[DEFAULT_RELEVANCY_PROMPT_TEMPLATE, llm]),
        )
        await ctx.set(
            "transform_query_pipeline",
            QueryPipeline(chain=[DEFAULT_TRANSFORM_QUERY_TEMPLATE, llm]),
        )

        await ctx.set("llm", llm)
        await ctx.set("index", index)
        await ctx.set("tavily_tool", TavilyToolSpec(api_key=tavily_ai_apikey))

        await ctx.set("query_str", query_str)
        await ctx.set("retriever_kwargs", retriever_kwargs)

        return PrepEvent()

    @step
    async def retrieve(self, ctx: Context, ev: PrepEvent) -> RetrieveEvent | None:
        """Retrieve the relevant nodes for the query."""
        query_str = await ctx.get("query_str")
        retriever_kwargs = await ctx.get("retriever_kwargs")

        if query_str is None:
            return None

        index = await ctx.get("index", default=None)
        tavily_tool = await ctx.get("tavily_tool", default=None)
        if not (index or tavily_tool):
            raise ValueError(
                "Index and tavily tool must be constructed. Run with 'documents' and 'tavily_ai_apikey' params first."
            )

        retriever: BaseRetriever = index.as_retriever(**retriever_kwargs)
        result = retriever.retrieve(query_str)
        await ctx.set("retrieved_nodes", result)
        await ctx.set("query_str", query_str)
        return RetrieveEvent(retrieved_nodes=result)

    @step
    async def eval_relevance(self, ctx: Context, ev: RetrieveEvent) -> RelevanceEvalEvent:
        """Evaluate relevancy of retrieved documents with the query."""
        retrieved_nodes = ev.retrieved_nodes
        query_str = await ctx.get("query_str")

        relevancy_results = []
        for node in retrieved_nodes:
            relevancy_pipeline = await ctx.get("relevancy_pipeline")
            relevancy = relevancy_pipeline.run(
                context_str=node.text, query_str=query_str
            )
            relevancy_results.append(relevancy.message.content.lower().strip())

        await ctx.set("relevancy_results", relevancy_results)
        return RelevanceEvalEvent(relevant_results=relevancy_results)

    @step
    async def extract_relevant_texts(self, ctx: Context, ev: RelevanceEvalEvent) -> TextExtractEvent:
        """Extract relevant texts from retrieved documents."""
        retrieved_nodes = await ctx.get("retrieved_nodes")
        relevancy_results = ev.relevant_results

        relevant_texts = [
            retrieved_nodes[i].text
            for i, result in enumerate(relevancy_results)
            if result == "yes"
        ]

        result = "\n".join(relevant_texts)
        return TextExtractEvent(relevant_text=result)

    @step
    async def transform_query_pipeline(self, ctx: Context, ev: TextExtractEvent) -> QueryEvent:
        """Search the transformed query with Tavily API."""
        relevant_text = ev.relevant_text
        relevancy_results = await ctx.get("relevancy_results")
        query_str = await ctx.get("query_str")

        # If any document is found irrelevant, transform the query string for better search results.
        if "no" in relevancy_results:
            tf_querypipeline = await ctx.get("transform_query_pipeline")
            transformed_query_str = tf_querypipeline.run(query_str=query_str).message.content
            # Conduct a search with the transformed query string and collect the results.
            tavily_tool = await ctx.get("tavily_tool")
            search_results = tavily_tool.search(
                transformed_query_str, max_results=5
            )
            search_text = "\n".join([result.text for result in search_results])
        else:
            search_text = ""

        return QueryEvent(relevant_text=relevant_text, search_text=search_text)

    @step
    async def query_result(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Get result with relevant text."""
        relevant_text = ev.relevant_text
        search_text = ev.search_text
        query_str = await ctx.get("query_str")

        documents = [Document(text=relevant_text + "\n" + search_text)]
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        result = query_engine.query(query_str)
        return StopEvent(result=result)
    
    

async def main():
    documents = SimpleDirectoryReader("src_workflow/data").load_data()
    ted_workflow = CorrectiveRAGWorkflow(timeout=360, verbose=True)
    
    index = await ted_workflow.run(documents=documents)
    
    response = await ted_workflow.run(
        query_str="How was Llama2 pretrained?",
        index=index,
        tavily_ai_apikey=tavily_ai_api_key,
    )

    print(str(response))
        
    response = await ted_workflow.run(
        query_str="What is the functionality of latest ChatGPT memory.",
        index=index,
        tavily_ai_apikey=tavily_ai_api_key
    )
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())
    # draw_all_possible_flows(CorrectiveRAGWorkflow, filename="crag_workflow.html")