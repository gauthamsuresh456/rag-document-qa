import sys
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.exception import CustomException
from src.logger import logger
from loader import load_and_chunk

load_dotenv()

CHROMA_PATH= "chroma_db"

def build_vector_store(chunks):
    try:
        logger.info("initializing embeddings model")
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        
        logger.info(f"Building Chroma vector store from {len(chunks)} chunks")
        
        vector_store =  Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            persist_directory=CHROMA_PATH
        )
        logger.info("vector store saved succesfully")
        
        return vector_store
    
    except Exception as e:
        logger.error("Faild to build vector store")
        raise CustomException(e,sys)


def load_vector_store():
    try:
        logger.info("Loading existing vector stored from disk")
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store =  Chroma(
            persist_directory = CHROMA_PATH,
            embedding_function = embedder
        )
        logger.info("vector store loaded succesfully")
        return vector_store
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    #Building it from scratch
    chunks = load_and_chunk("test.pdf")
    vector_store = build_vector_store(chunks)
    
    print(f"Vector store built and saved to '{CHROMA_PATH}/'")
    print(f"Total chunks stored: {vector_store._collection.count()}")
    
    print("\nLoading vector store back from disk...")
    vs = load_vector_store()
    
    question = "How to use neural network to generate a membership function"  # change to match your PDF
    results = vs.similarity_search(question, k=3)

    print(f"\nTop 3 results for: '{question}'")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")