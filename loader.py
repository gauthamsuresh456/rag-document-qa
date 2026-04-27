from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

#loading the pdf
loader = PyPDFLoader("test.pdf")
pages = loader.load()

#inspect what you got 
print(f"Number of pages: {len(pages)}")
print(f"Contents from the first Page: \n {pages[0].page_content[:500]}")
print(f"Metadata:{pages[0].metadata}")