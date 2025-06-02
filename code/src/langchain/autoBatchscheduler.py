#pip install pandas openpyxl langchain gpt4all
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All

# Load Excel data
excel_path = "data.csv"
df = pd.read_csv(excel_path)
data_str = df.to_string(index=False)

# Load local GPT4All model
llm = GPT4All(model="./models/ggml-gpt4all-j-v1.3-groovy.bin")

# Create a prompt template
template = """
You are a helpful data analyst. Given the following data from an Excel file:

{data}

Answer the user's question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["data", "question"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# User question loop
while True:
    question = input("\nAsk a question about the Excel data (or type 'exit'): ")
    if question.lower() == "exit":
        break
