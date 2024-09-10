import os
import pytest
from src.models.models import ModelManager
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_config(tmp_path):
    # Create a temporary configuration file for testing
    config_content = """
    llm_model: "llama-3.1-70b-versatile"
    embedding_model: "mxbai-embed-large"
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return str(config_file)

@pytest.fixture
def model_manager(mock_config):
    # Initialize the ModelManager with a mock config
    return ModelManager(mock_config)

@patch("src.models.models.ChatGroq")
def test_load_llm_groq(mock_llm_groq, model_manager):
    # Mock the Groq LLM model load
    mock_llm_groq.return_value = MagicMock()

    llm_groq = model_manager.load_llm_groq()

    assert llm_groq is not None, "LLM Groq model should be loaded"
    mock_llm_groq.assert_called_once_with(
        model_name="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY")
    )

@patch("src.models.models.OllamaEmbeddings")
def test_load_embedding_model(mock_ollama_embeddings, model_manager):
    # Mock the embedding model load
    mock_ollama_embeddings.return_value = MagicMock()

    embedding_model = model_manager.load_embedding_model()

    assert embedding_model is not None, "Embedding model should be loaded"
    mock_ollama_embeddings.assert_called_once_with(model="mxbai-embed-large")

@patch("src.models.models.ChatGroq")
def test_count_tokens(mock_llm_groq, model_manager):
    # Mock the LLM token counting method
    mock_llm_groq.return_value.get_num_tokens.return_value = 5
    model_manager.llm_groq = mock_llm_groq.return_value

    token_count = model_manager.count_tokens("Hello world.")
    
    assert token_count == 5, "Token count should be 5"
    mock_llm_groq.return_value.get_num_tokens.assert_called_once_with("Hello world.")