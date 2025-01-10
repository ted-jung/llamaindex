import os
import asyncio
import json
import re
import pandas as pd
from pathlib import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core import (
    Settings,
    SQLDatabase, 
    VectorStoreIndex,
    PromptTemplate,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Context,
    Event,
)

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=720.0)
Settings.llm = llm

def delete_file(directory_path):
  if not os.path.exists(directory_path):
    raise FileNotFoundError(f"Directory '{directory_path}' not found.")

  for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    try:
      if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except OSError as e:
      print(f"Error deleting file '{file_path}': {e}")

# Directory for wiki's data
data_dir = Path("./data/wiki/WikiTableQuestions/csv/100-csv")
csv_files = sorted([f for f in data_dir.glob("*.csv")])
dfs = []
for csv_file in csv_files:
    print(f"processing file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    except Exception as e:
        print(f"Error parsing {csv_file}: {str(e)}")


# Make json files where it's metadata stored in.
tableinfo_dir = "WikiTableQuestions_TableInfo"
delete_file("./data/wiki/"+tableinfo_dir)
if not os.path.exists("./data/wiki/"+tableinfo_dir):
    os.mkdir("./data/wiki/"+tableinfo_dir)


# Extract Table Name & Summary from each Table
# Pydantic class(to instantiate a structured LLM)
class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces and NO slash and No hyphen)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )


prompt_str = """
    Give me a summary of the table with the following JSON format.

    - The table name must be unique to the table and describe it while being concise. 
    - Do NOT output a generic table name (e.g. table, my_table).

    Do NOT make the table name one of the following: {exclude_table_name_list}

    Table:
    {table_str}

    Summary: 
"""

prompt_tmpl = ChatPromptTemplate(message_templates=[ChatMessage.from_str(prompt_str, role="user")])

def _get_tableinfo_with_index(idx: int) -> str:
    results_gen = Path(f"./data/wiki/{tableinfo_dir}").glob(f"{idx}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.parse_file(path)
    else:
        raise ValueError(
            f"More than one file matching index: {list(results_gen)}"
        )


table_names = set()
table_infos = []
for idx, df in enumerate(dfs):
    table_info = _get_tableinfo_with_index(idx)
    if table_info:
        table_infos.append(table_info)
    else:
        while True:
            df_str = df.head(10).to_csv()
            table_info = llm.structured_predict(
                TableInfo,
                prompt_tmpl,
                table_str=df_str,
                exclude_table_name_list=str(list(table_names)),
            )
            table_name = table_info.table_name
            print(f"Processed table: {table_name}")
            if table_name not in table_names:
                table_names.add(table_name)
                break
            else:
                # try again
                print(f"Table name {table_name} already exists, trying again.")
                pass

        out_file = f"./data/wiki/{tableinfo_dir}/{idx}_{table_name}.json".replace(" ", "_")
        if not os.path.isfile(out_file):
            json.dump(table_info.dict(), open(out_file, "w"))
            
    table_infos.append(table_info)


# put data into sqlite db
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    inspect,
)

def delete_all_tables(engine):
    inspector = inspect(engine)
    temp_metadata = MetaData()
    
    for table_name in inspector.get_table_names():
        try:
            table = Table(table_name, temp_metadata, autoload_with=engine)
            table.drop(engine, checkfirst=True)
            print(f"Deleted table: {table_name}")
        except Exception as e:
            print(f"Error: {e}")


# Function to create a sanitized column name
def sanitize_column_name(col_name):
    # Remove special characters and replace spaces with underscores
    return re.sub(r"\W+", "_", col_name)


# Function to create a table from a DataFrame using SQLAlchemy
def create_table_from_dataframe(df: pd.DataFrame, table_name: str, engine, metadata_obj):
    # Sanitize column names
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    # Dynamically create columns based on DataFrame columns and data types
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    # Create a table with the defined columns
    table_name = table_name.replace(" ","_")
    table = Table(table_name, metadata_obj, *columns)

    # delte all
    # delete_stmt = table.delete()
    # conn.execute(delete_stmt)

    # Create the table in the database
    metadata_obj.create_all(engine)

    # Insert data from DataFrame into the table
    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()


# engine = create_engine("sqlite:///:memory:")
engine = create_engine("sqlite:///wiki_table_questions.db")
metadata_obj = MetaData()
delete_all_tables(engine)

for idx, df in enumerate(dfs):
    tableinfo = _get_tableinfo_with_index(idx)
    print(f"Creating table: {tableinfo.table_name}")
    create_table_from_dataframe(df, tableinfo.table_name, engine, metadata_obj)


sql_database = SQLDatabase(engine)

# Build obj_retriever via ObjectIndex
# schema, node mapping and index

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    SQLTableSchema(table_name=t.table_name.replace(" ","_"), context_str=t.table_summary) for t in table_infos
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)


# build sql_retriever via SQLRetriever

from llama_index.core.retrievers import SQLRetriever
from typing import List

sql_retriever = SQLRetriever(sql_database)

def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.llms import ChatResponse


def parse_response_to_sql(chat_response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = chat_response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        # TODO: move to removeprefix after Python 3.9+
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(dialect=engine.dialect.name)
print(text2sql_prompt.template)


response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str,)


# workflow
class TableRetrieveEvent(Event):
    """Result of running table retrieval."""
    table_context_str: str
    query: str

class TextToSQLEvent(Event):
    """Text-to-SQL event."""
    sql: str
    query: str

class TextToSQLWorkflow1(Workflow):
    """Text-to-SQL Workflow that does query-time table retrieval."""

    def __init__(
        self,
        obj_retriever,
        text2sql_prompt,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        *args,
        **kwargs
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self.obj_retriever = obj_retriever
        self.text2sql_prompt = text2sql_prompt
        self.sql_retriever = sql_retriever
        self.response_synthesis_prompt = response_synthesis_prompt
        self.llm = llm

    @step
    async def retrieve_tables(self, ctx: Context, ev: StartEvent) -> TableRetrieveEvent:
        """Retrieve tables."""
        table_schema_objs = self.obj_retriever.retrieve(ev.query)
        table_context_str = get_table_context_str(table_schema_objs)
        return TableRetrieveEvent(
            table_context_str=table_context_str, query=ev.query
        )

    @step
    async def generate_sql(self, ctx: Context, ev: TableRetrieveEvent) -> TextToSQLEvent:
        """Generate SQL statement."""
        fmt_messages = self.text2sql_prompt.format_messages(
            query_str=ev.query, schema=ev.table_context_str
        )
        chat_response = self.llm.chat(fmt_messages)
        sql = parse_response_to_sql(chat_response)
        return TextToSQLEvent(sql=sql, query=ev.query)

    @step
    async def generate_response(self, ctx: Context, ev: TextToSQLEvent) -> StopEvent:
        """Run SQL retrieval and generate response."""
        retrieved_rows = self.sql_retriever.retrieve(ev.sql)
        fmt_messages = self.response_synthesis_prompt.format_messages(
            sql_query=ev.sql,
            context_str=str(retrieved_rows),
            query_str=ev.query,
        )
        chat_response = llm.chat(fmt_messages)
        return StopEvent(result=chat_response)
    
    
# from llama_index.core.workflow import draw_all_possible_flows
# from IPython.display import display, HTML

# draw_all_possible_flows(
#     TextToSQLWorkflow1, filename="text_to_sql_table_retrieval.html"
# )
# # Read the contents of the HTML file
# with open("text_to_sql_table_retrieval.html", "r") as file:
#     html_content = file.read()
# # Display the HTML content
# display(HTML(html_content))


workflow = TextToSQLWorkflow1(
    obj_retriever,
    text2sql_prompt,
    sql_retriever,
    response_synthesis_prompt,
    llm,
    verbose=True,
)


async def my_async_function():
    response = await workflow.run(
        #query="What was the year that The Notorious B.I.G was signed to Bad Boy?"
        query="What movie has the word 'ring' in its title?"
    )
    print(str(response))



asyncio.run(my_async_function())
