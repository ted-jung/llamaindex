# ===========================================================================
# Evaluation
# Date: 30, Jan 2025
# Writer: Ted, Jung
# Description: the relevancy between retrieved context and answer
#              1. Question generation
# ===========================================================================


import asyncio

from llama_index.core import VectorStoreIndex

from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RelevancyEvaluator

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core import Settings

# Define llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", temperature=0.0, request_timeout=720.0)
gpt4o_mini = OpenAI(temperature=0, model="gpt-4o-mini")


# Define evaluator for (faithfulness, relevancy)
faithfulness_evaluator = FaithfulnessEvaluator(llm=gpt4o_mini)
relevancy_evaluator = RelevancyEvaluator(llm=gpt4o_mini)


# Create an index from documents
documents = SimpleDirectoryReader(input_files=["./data/paul_graham/paul_graham_essay_short.txt"]).load_data()
vector_index = VectorStoreIndex.from_documents(documents=documents)


# Define generator & generate questions using documents
dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    llm=llm,
    num_questions_per_chunk=2,  # set the number of questions per nodes
)


# Generated Questions
rag_dataset = dataset_generator.generate_questions_from_nodes()
questions = [e.query for e in rag_dataset.examples]


runner = BatchEvalRunner(
    {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
    workers=8,
)

async def ted():
    eval_results = await runner.aevaluate_queries(
        vector_index.as_query_engine(), queries=questions
    )

    print(eval_results)


if __name__ == "__main__":
    asyncio.run(ted())