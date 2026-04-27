from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

#loading the pdf
loader = PyPDFLoader("test.pdf")
pages = loader.load()

#inspect what you got 
#To inspect we are gonna just take one page
print(f"Number of pages: {len(pages)}")
print(f"Contents from the first Page: \n {pages[0].page_content[:500]}")
print(f"Metadata:{pages[0].metadata}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  #Max characters per chunk
    chunk_overlap=50,  #Overlap between chunk so context isnt lost
    separators=["\n\n","\n","."," "]  #split on this order
)
chunks = splitter.split_documents(pages)

print(f"Number of chunks: {len(chunks)}")
print(f"Contents from the first Page: \n {chunks[0].page_content }") 
print(f"Contents from the second Page: \n {chunks[1].page_content }") 