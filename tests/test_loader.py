import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loader import load_and_chunk
from vector_store import build_vector_store, load_vector_store

def test_load_and_chunk_returns_list():
    chunks = load_and_chunk("test.pdf")
    assert isinstance(chunks, list)

def test_chunks_not_empty():
    chunks = load_and_chunk("test.pdf")
    assert len(chunks) > 0

def test_chunk_has_content():
    chunks = load_and_chunk("test.pdf")
    assert len(chunks[0].page_content) > 0
    
def test_vector_store_builds():
    chunks = load_and_chunk("test.pdf")
    vs = build_vector_store(chunks)
    assert vs._collection.count() > 0

def test_similarity_search_returns_results():
    vs = load_vector_store()
    results = vs.similarity_search("What is this document about?", k=2)
    assert len(results) == 2
    assert len(results[0].page_content) > 0