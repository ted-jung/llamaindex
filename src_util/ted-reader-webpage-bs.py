# ===========================================================================
# Crawl web with BeautifulSoup and do a query on LLM
# Created: 19, Feb 2025
# Updated: 21, Feb 2025
# Writer: Ted, Jung
# Description: 
#   Scrap web page -> Indexing -> PromptTemplate -> Query
# ===========================================================================



from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.settings import Settings
from llama_index.core.indices import VectorStoreIndex
from llama_index.core import PromptTemplate



# Do a Query to find what you want
def ted_query(str_context, str_query, tone_name):

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5") 
    llm = OpenAI(model="gpt-4o-mini", timeout=700.0)


    # Creae a PromptTemplate
    qa_prompt_tmpl_str = """\
        Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Please write the answer in the style of {tone_name}
        Query: {query_str}
        Answer: \
    """

    # Mapping variables
    template_var_mappings = {
        "context_str": f"{str_context}", 
        "query_str": f"{str_query}", 
        "tone_name": f"{tone_name}"
    }

    prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str, template_var_mappings)
    

    # Read documents to create an index to query
    my_document = Document(text=str_context)
    documents = [my_document]
    ted_index = VectorStoreIndex.from_documents(documents)

    answer_engine = ted_index.as_query_engine(llm = llm, prompt_tmpl=prompt_tmpl)
    res = (answer_engine.query(str_query))

    return res



# Get sources from the web using headless browser
def ted_source_data(url):

    job_country_string = ""
    options = Options()
    options.add_argument("--headless")    # Run Chrome in headless mode
    options.add_argument("--disable-gpu") # Recommended for headless

    driver = webdriver.Chrome(options=options)
    driver.get(url)


    # Wait for the data to load (important!)
    # Example: Wait for an element to appear (replace with your selector, mb-16, mb-1, text-ne~~~)
    # CSS? replace with your web site
    try:
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, ".mb-16")) 
        WebDriverWait(driver, 3).until(element_present) # Wait up to 3 seconds
        html = driver.page_source                       # Get the updated HTML
        soup = BeautifulSoup(html, "html.parser")

        data_elements1 = soup.find_all("div", class_="mb-1")
        data_elements2 = soup.find_all("div", class_= "text-neutral-200 text-base font-normal")
        
        for element1, element2 in zip(data_elements1, data_elements2):
            role = element1.text.strip()
            country = element2.text.strip()
            job_country_string += f"Job: {role}, Country: {country} | "
        
        return job_country_string
    
    except Exception as e:
        print(f"Error waiting for element: {e}")

    finally:
        driver.quit()



# Main
if __name__ == "__main__":

    career_response = ted_source_data('https://xxx.com/company/careers')

    response = ted_query(career_response, "Given index is two fields opened Job, Country. Find job position each country and count jobs be opened in each contry with alphabet order", "Shakespeare")
    print(response)

