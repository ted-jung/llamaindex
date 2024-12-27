from utils import llama32
from utils import load_env

from datetime import datetime



def main():
    print("This script is running directly.")
    
    load_env()

    current_date = datetime.now()
    formatted_date = current_date.strftime("%d %B %Y")

    weather_question = ""

    messages = [
        {
            "role": "system",
            "content": f"""
            Environment: ipython
            Tools: brave_search, wolfram_alpha
            Cutting Knowledge Date: December 2023
            Today Date: {formatted_date}
            """
        },
        {
            "role": "user",
            "content": weather_question
        }
    ]

    print(llama32(messages))

# This block ensures the script is run as a standalone program
if __name__ == "__main__":
    main()