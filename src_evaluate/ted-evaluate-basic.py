from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, RetrieverEvaluator
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", temperature=0.0, request_timeout=720.0)
# create llm
#llm = OpenAI(model="gpt-4", temperature=0.0)

# build index
documents = SimpleDirectoryReader(input_files=["./data/paul_graham/paul_graham_essay_short.txt"]).load_data()
vector_index = VectorStoreIndex.from_documents(documents=documents)

# define evaluator
evaluator = FaithfulnessEvaluator(llm=llm)

# query index
# response(response + source)
query_engine = vector_index.as_query_engine()
response = query_engine.query(
    "What two things did Paul Graham before college?"
)
eval_result = evaluator.evaluate_response(response=response)
print(eval_result)
print(str(eval_result.passing))


retriever = vector_index.as_retriever(similarity_top_k=2)

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

eval_result2 = retriever_evaluator.evaluate(
    query="query", expected_ids=["node_id1", "node_id2"]
)

print(eval_result2)
