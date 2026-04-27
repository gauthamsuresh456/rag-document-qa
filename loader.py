from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import logger
from src.exception import CustomException
import sys

load_dotenv()

def load_and_chunk(pdf_path:str, chunk_size:int=500, chunk_overlap:int=50):
    try:
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages= loader.load()
        logger.info(f"Loaded {len(pages)} pages")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  #Max characters per chunk
            chunk_overlap=50,  #Overlap between chunk so context isnt lost
            separators=["\n\n","\n","."," "]  #split on this order            
        )
        chunks =  splitter.split_documents(pages)
        
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.info(f"Failed to load or chunk document")
        raise CustomException(e, sys)

if __name__ == "__main__":
    chunks = load_and_chunk("test.pdf")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Contents from the first Page: \n {chunks[0].page_content }") 
    print(f"Contents from the second Page: \n {chunks[1].page_content }") 
    