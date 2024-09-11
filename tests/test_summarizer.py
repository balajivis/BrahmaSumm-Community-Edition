import pytest
from src.summarize import Summarizer 

@pytest.fixture(scope="module")
def summarizer():
    # This will set up the summarizer once for all tests
    config_path = 'config/config.yaml'
    return Summarizer(config_path)

def test_find_suitable_theme(summarizer):
    result = summarizer.find_suitable_theme("Who is John Galt!")
    assert isinstance(result, str)  # Check if result is a string
    assert len(result) > 0  # Check if result is not empty

def test_summary(summarizer):
    # Run the summarizer and store the summary for later tests
    summary = summarizer('https://www.whitehouse.gov/state-of-the-union-2024/',"web")
    assert isinstance(summary, str)
    assert len(summary) > 0

def test_analysis(summarizer):
    # This test depends on `test_summary` and expects a valid summary
    chunk_words, total_chunks, total_words, total_tokens, tokens_sent_tokens = summarizer.get_analysis()
    # Ensure all elements in chunk_words are integers
    assert all(isinstance(word_count, int) for word_count in chunk_words), "All elements in chunk_words should be integers"
    
    # Ensure no element in chunk_words is zero (if that's expected)
    assert all(word_count > 0 for word_count in chunk_words[:-1]), "No word count should be zero except possibly the last element"

    # Check that the length of chunk_words is what you expect (optional)
    assert len(chunk_words) > 0, "chunk_words should not be empty"

    assert isinstance(total_chunks, int)
    assert isinstance(total_words, int)
    assert isinstance(total_tokens, int)
    assert isinstance(tokens_sent_tokens, int)