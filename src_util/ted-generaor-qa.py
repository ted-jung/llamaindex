# ===========================================================================
# QA Generator
# Date: 14, Feb 2025
# Writer: Ted, Jung
# Description: Generate Question and Answer for testing
# ===========================================================================

import os
import nest_asyncio
import asyncio
import random
import numpy as np


from pathlib import Path
from llama_index.readers.file import (
    PDFReader,
    UnstructuredReader,
    PyMuPDFReader,
)
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
)

from llama_index.core.node_parser import SentenceSplitter, SimpleFileNodeParser
from llama_index.core.schema import IndexNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from llama_index.core.evaluation import (
    DatasetGenerator, 
    QueryResponseDataset,
    CorrectnessEvaluator,
    BatchEvalRunner,
)

from llama_index.core.evaluation.eval_utils import get_responses


llm = OpenAI(model="gpt-4o-mini")
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

nest_asyncio.apply()

nb_questions = 5

# Load document
curr_dir = os.getcwd()
loader = PDFReader()
docs0 = loader.load_data(file=Path(f"{curr_dir}/data/llama/llama2.pdf"))


# Read it and make documents to turn it into nodes
doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]
node_parser = SentenceSplitter(chunk_size=1024)
base_nodes = node_parser.get_nodes_from_documents(docs)


# Create an index using nodes and make an engine
index = VectorStoreIndex(base_nodes)
query_engine = index.as_query_engine(similarity_top_k=2)



evaluator_c = CorrectnessEvaluator(llm=llm)
evaluator_dict = {
    "correctness": evaluator_c,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)


async def gen_data_set():
    dataset_generator = DatasetGenerator(
        base_nodes[:nb_questions],
        llm=llm,
        show_progress=True,
        num_questions_per_chunk=3,
    )
    eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=60)
    eval_dataset.save_json(f"{curr_dir}/data/llama/llama2_eval_qr_dataset2.json")
    # optional
    eval_dataset = QueryResponseDataset.from_json(
        f"{curr_dir}/data/llama/llama2_eval_qr_dataset.json"
    )

    full_qr_pairs = eval_dataset.qr_pairs
    num_exemplars = 2
    num_eval = 40

    exemplar_qr_pairs = random.sample(full_qr_pairs, num_exemplars)
    eval_qr_pairs = random.sample(full_qr_pairs, num_eval)

    len(exemplar_qr_pairs)
    len(eval_qr_pairs)

    return eval_qr_pairs


async def get_correctness(query_engine, eval_qa_pairs, batch_runner):
    # then evaluate
    # TODO: evaluate a sample of generated results
    eval_qs = [q for q, _ in eval_qa_pairs]
    eval_answers = [a for _, a in eval_qa_pairs]
    pred_responses = get_responses(eval_qs, query_engine, show_progress=True)

    eval_results = await batch_runner.aevaluate_responses(
        eval_qs, responses=pred_responses, reference=eval_answers
    )
    avg_correctness = np.array(
        [r.score for r in eval_results["correctness"]]
    ).mean()
    return avg_correctness



if __name__ == "__main__":
    a = asyncio.run(gen_data_set())

    asyncio.run(get_correctness(query_engine,a, batch_runner))
