# ===========================================================================
# Extract PDF and use pydantic for structured LLMs
# Date: 30, Jan 2025
# Writer: Ted, Jung
# Description: Structured Data Extraction using pydantic
#              Leverage a Structured LLM
# ===========================================================================

import json

from pydantic import BaseModel, Field
from datetime import datetime
from llama_index.readers.file import PDFReader
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="deepseek-r1", request_timeout=720.0)
Settings.llm = llm

class LineItem(BaseModel):
    """A line item in an invoice."""

    item_name: str = Field(description="The name of this item")
    price: float = Field(description="The price of this item")


class Invoice(BaseModel):
    """A representation of information from an invoice."""

    invoice_id: str = Field(
        description="A unique identifier for this invoice, often a number"
    )
    date: datetime = Field(description="The date this invoice was created")
    line_items: list[LineItem] = Field(
        description="A list of all the items in this invoice"
    )
    payments: list[LineItem] = Field(
        description="A list of payment info with company name"
    )
    cab_info: str = Field(
        description="cap info"
    )

pdf_reader = PDFReader()
documents = pdf_reader.load_data(file=Path("./data/pdf/uber/receipt/uber.pdf"))
text = documents[0].text

# Define the expected structure of the output you want to receive from the LLM
# Invoide: pydantic model
# sllm: enforce a specific output structure
sllm = llm.as_structured_llm(Invoice)

response = sllm.complete(text)

print(response)

json_response = json.loads(response.text)
print(json.dumps(json_response,indent=2))