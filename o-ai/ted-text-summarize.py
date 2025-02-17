# ===========================================================================
# How to user summarize
# Date: 17, Feb 2025
# Updated: 17, Feb 2025
# Description: If you have multiple files to summarize?
#              Then user chat.completion.parse() with response structure
#              Can use tempate to guide LLM to advice the output format
# ===========================================================================
 

import os

from pydantic import BaseModel
from openai import OpenAI
from textwrap import dedent

client = OpenAI()


curr_dir = os.getcwd()
articles = [
    f"{curr_dir}/o-ai/data/cnn.md"
]


def get_article_content(path):
    with open(path, 'r') as f:
        content = f.read()
    return content
        
content = [get_article_content(path) for path in articles]
print(content)
print(100*"=")



# Create a template to be used while summaring an article
summarization_prompt = '''
    You will be provided with content from an article about an invention.
    Your goal will be to summarize the article following the schema provided.
    Here is a description of the parameters:
    - invented_year: year in which the invention discussed in the article was invented
    - summary: one sentence summary of what the invention is
    - inventors: array of strings listing the inventor full names if present, otherwise just surname
    - concepts: array of key concepts related to the invention, each concept containing a title and a description
    - description: short description of the invention
'''



# Define the structred output having fields and nested class also having fields
class ArticleSummary(BaseModel):
    invented_year: int
    summary: str
    inventors: list[str]
    description: str

    class Concept(BaseModel):
        title: str
        description: str

    concepts: list[Concept]



# Send a single article to LLM(OpenAI) to summarize it
# Use the pre-defined class(format) structured when summarize it
def get_article_summary(text: str):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": dedent(summarization_prompt)},
            {"role": "user", "content": text}
        ],
        response_format=ArticleSummary,
    )

    return completion.choices[0].message.parsed



summaries = []

for i in range(len(content)):
    print(f"Analyzing article #{i+1}...")
    summaries.append(get_article_summary(content[i]))

    print("Done.")



# print summary
def print_summary(summary):
    print(f"Invented year: {summary.invented_year}\n")
    print(f"Summary: {summary.summary}\n")
    print("Inventors:")
    for i in summary.inventors:
        print(f"- {i}")
    print("\nConcepts:")
    for c in summary.concepts:
        print(f"- {c.title}: {c.description}")
    print(f"\nDescription: {summary.description}")


for i in range(len(summaries)):
    print(f"ARTICLE {i}\n")
    print_summary(summaries[i])
    print("\n\n")



