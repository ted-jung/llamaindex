# ===========================================================================
# Crawl web and do LLM (opensource scrapy)
# Created: 19, Feb 2025
# Updated: 19, Feb 2025
# Writer: Ted, Jung
# Description: 
#   Leverage crawl using scrapy
# ===========================================================================

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

from llama_index.core.indices import SummaryIndex



def ted_data(url):
    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode
    options.add_argument("--disable-gpu") # Recommended for headless

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # Wait for the data to load (important!)
    # Example: Wait for an element to appear (replace with your selector)
    try:
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, ".mb-16")) #Replace with the actual selector
        WebDriverWait(driver, 5).until(element_present) #Wait up to 10 seconds
        print(100*"=")
        html = driver.page_source  # Get the updated HTML
        # print(html)
        soup = BeautifulSoup(html, "html.parser")

        data_elements1 = soup.find_all("div", class_="mb-1") #Replace with the actual selector
        data_elements2 = soup.find_all("div", class_= "text-neutral-200 text-base font-normal") #Replace with the actual selector
        
        for element1, element2 in zip(data_elements1, data_elements2):
            role = element1.text.strip()
            country = element2.text.strip()
            combined_string = f"Role: {role}, Country: {country}"
            print(combined_string)

    except Exception as e:
        print(f"Error waiting for element: {e}")

    finally:
        driver.quit()



if __name__ == "__main__":
    ted_data('https://clickhouse.com/company/careers')

