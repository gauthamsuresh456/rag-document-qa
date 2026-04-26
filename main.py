from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

response = llm.invoke("What is retrival-augmentation in one sentence")
print(response.content)