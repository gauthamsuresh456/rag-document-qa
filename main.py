from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

#Loading the .env with the api key
load_dotenv()

#Loading the model we want, cus load_dotenv the model will automatically get the api key
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

#sending in the prompt and receiving the response back 
response = llm.invoke("What is retrieval-augmentation in one sentence")
print(response.content)