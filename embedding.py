import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from src.logger import logger
from src.exception import CustomException
from loader import load_and_chunk

load_dotenv()

def embed_chunks(chunk):
    try:
        logger.info("initializing embedding model")
        embedder = OpenAIEmbeddings(model= 'text-embedding-3-small')

        #grabbing text from each chunk
        texts = [chunk.page_content for chunk in chunks]
        
        logger.info(f"Embedding {len(texts)} chunks")
        vectors =  embedder.embed_documents(texts)
        
        logger.info(f"Done. Each vector has {len(vectors[0])} dimensions")
        return vectors
    except Exception as e :
        logger.error("Embedding Faild")
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    chunks = load_and_chunk("test.pdf")
    vectors = embed_chunks(chunks)
    
    print(f"Number of vectors: {len(vectors)}")
    print(f"Dimension Per vectors: {len(vectors[0])}")
    print(f"\n First 10 vectors:\n {vectors[0][:10]}")
    
    v1 = vectors[0]
    v2 = vectors[1]
    
    dot_prod = sum(a*b for a,b in zip(v1,v2))
    mag1 = sum(a**2 for a in v1)** 0.5
    mag2 = sum(a**2 for a in v2)** 0.5
    
    similarity = dot_prod/(mag1*mag2)
    print(f"\n similarity between chunk 1 and chunk 2: {similarity:.4f}")
    # 1.0 = identical, 0.0 = unrelated, -1.0 =  opposite
    
    #Test to find a common chunk
    
    logger.info("initializing embedding model")
    embedder = OpenAIEmbeddings(model= 'text-embedding-3-small')
        
    question = "How to use neural network to generate a membership function"  
    q_vector = embedder.embed_query(question)

    similarities = []
    for i, v in enumerate(vectors):
        dot = sum(a*b for a, b in zip(q_vector, v))
        m1 = sum(a**2 for a in q_vector) ** 0.5
        m2 = sum(a**2 for a in v) ** 0.5
        sim = dot / (m1 * m2)
        similarities.append((sim, i, chunks[i].page_content[:80]))

    similarities.sort(reverse=True)
    print(f"\nTop 3 most relevant chunks for your question({question}):")
    for sim, idx, preview in similarities[:3]:
        print(f"\nChunk {idx} | similarity: {sim:.4f}")
        print(f"Preview: {preview}...")
