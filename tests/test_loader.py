import pytest
from loader import load_and_chunk

def test_load_and_chunk_returns_list():
    chunks = load_and_chunk("test.pdf")
    assert isinstance(chunks, list)

def test_chunks_not_empty():
    chunks = load_and_chunk("test.pdf")
    assert len(chunks) > 0

def test_chunk_has_content():
    chunks = load_and_chunk("test.pdf")
    assert len(chunks[0].page_content) > 0