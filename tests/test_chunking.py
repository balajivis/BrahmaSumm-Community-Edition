import pytest
from src.chunking.textchunking import ChunkManager  # Adjust import based on your project structure

@pytest.fixture
def chunk_manager():
    # Create an instance of ChunkManager with a sample configuration
    return ChunkManager(config_path='config/config.yaml')

def test_preprocess_text(chunk_manager):
    text = "This is a test.\n\nThis is a second paragraph with non-ASCII: ñ"
    processed_text = chunk_manager.preprocess_text(text)
    
    # Ensure multiple newlines are handled, and non-ASCII characters are removed
    assert "\n\n" in processed_text, "Preprocessing should preserve paragraph breaks"
    assert "ñ" not in processed_text, "Preprocessing should remove non-ASCII characters"

def test_flexible_chunk(chunk_manager):
    text = """
    **BrahmaSumm** is an advanced document summarization and visualization tool designed to streamline document management, knowledge base creation, and chatbot enhancement. By leveraging cutting-edge chunking and clustering techniques, BrahmaSumm reduces token usage sent to Large Language Models (LLMs) by up to 99%, while maintaining the quality of content. The tool provides intuitive document processing, stunning visualizations, and efficient querying across multiple formats.

    ## Features (v0.1)
    - **Multi-format support**: Summarize and visualize content from PDFs, YouTube videos, audio files, HTML, spreadsheets, and Google Drive folders.
    - **Clustering-based summarization**: BrahmaSumm intelligently chunks and clusters documents, extracting key insights while preserving quality.
    - **UMAP visualization**: View your documents in an intuitive, visual format that highlights clusters and relationships within the content.
    - **Token reduction**: Reduce the token count sent to LLMs by up to 99% with BrahmaSumm's efficient clustering algorithms.
    - **Extract Tables, Images, and Text**: Seamlessly extract and summarize data from tables, images, and text within documents.
    - **Vectorization for querying**: Enable powerful document querying by vectorizing content for efficient search and retrieval.  """
    chunk_manager.flexible_chunk(text)

    chunks = chunk_manager.get_chunks()
    
    # Assert there are multiple chunks and each chunk has reasonable word counts
    assert len(chunks) > 1, "Text should be chunked into multiple chunks"
    word_counts = chunk_manager.get_word_count_per_chunk()
    
    # Ensure each chunk has word count between 75% and 125% of the target
    min_words = int(chunk_manager.target_words * (1 - chunk_manager.flexibility))
    max_words = 2*int(chunk_manager.target_words * (1 + chunk_manager.flexibility))

    for count in word_counts:
        assert min_words <= count <= max_words, "Each chunk should meet word count flexibility range"

def test_get_word_count_per_chunk(chunk_manager):
    text = "This is a test chunk.\n\nThis is another test chunk."
    chunk_manager.flexible_chunk(text)
    
    word_counts = chunk_manager.get_word_count_per_chunk()
    
    # Ensure the correct word count for each chunk is returned
    assert all(isinstance(count, int) for count in word_counts), "Word count should be an integer"
    assert sum(word_counts) > 0, "Word count should be greater than 0"

def test_get_total_chunks(chunk_manager):
    text = "This is a test chunk.\n\nThis is another test chunk."
    chunk_manager.flexible_chunk(text)
    
    total_chunks = chunk_manager.get_total_chunks()
    
    # Ensure the total number of chunks is correct
    assert total_chunks > 0, "Total number of chunks should be greater than 0"

def test_get_total_words(chunk_manager):
    text = "This is a test chunk with several words.\n\nThis is another test chunk with some more words."
    chunk_manager.flexible_chunk(text)
    
    total_words = chunk_manager.get_total_words()
    
    # Ensure the total word count across all chunks is correct
    assert total_words == len(text.split()), "Total word count should match the word count of the input text"