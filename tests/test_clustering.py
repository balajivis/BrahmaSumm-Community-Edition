import pytest
import numpy as np
from src.clustering.clustering import ClusterManager
from src.models.models import ModelManager  # Import the actual ModelManager

@pytest.fixture
def model_manager():
    # Initialize the ModelManager with the config
    config_path = 'config/config.yaml'
    return ModelManager(config_path)

@pytest.fixture
def cluster_manager(model_manager):
    # Load the real embedding model
    embedding_model = model_manager.load_embedding_model()

    # Initialize the ClusterManager with the real embedding model
    config_path = 'config/config.yaml'
    return ClusterManager(embedding_model, config_path)

def test_initialization(cluster_manager):
    # Test if ClusterManager initializes properly with the given config and embedding model
    assert cluster_manager.embedding_model is not None, "Embedding model should be initialized"
    assert isinstance(cluster_manager.config, dict), "Config should be a dictionary"
    assert cluster_manager.vectors == [], "Vectors should be initialized as an empty list"

def test_embed_documents_with_progress(cluster_manager):
    # Real document chunks to embed
    chunks = ["This is the first chunk.", "This is the second chunk.", "This is the third chunk."]

    # Embed the documents
    cluster_manager.embed_documents_with_progress(chunks)

    # Ensure that vectors are populated and are NumPy arrays
    assert len(cluster_manager.vectors) == 3, "There should be 3 embedded vectors"
    assert all(isinstance(vec, list) for vec in cluster_manager.vectors), "Each vector should be a list"

def test_cluster_document(cluster_manager):
    # Embed some actual text chunks to get real vectors
    chunks = ["This is the first chunk.", "This is the second chunk.", "This is the third chunk.", 
              "This is the fourth chunk.", "This is the fifth chunk."]
    cluster_manager.embed_documents_with_progress(chunks)
    
    # Get the length of the embedding (list length) dynamically
    embedding_dim = len(cluster_manager.vectors[0])  # Extract the embedding dimension from list length

    # Perform clustering
    labels, centers = cluster_manager.cluster_document(n_clusters=2)

    # Ensure the correct number of clusters and centers are returned
    assert len(labels) == len(chunks), "There should be a label for each vector"
    
    # Ensure that the shape of the cluster centers matches the embedding dimension
    assert len(centers) == 2, "There should be 2 cluster centers"
    assert all(len(center) == embedding_dim for center in centers), f"Each cluster center should have {embedding_dim} dimensions"

def test_find_n_closest_representatives(cluster_manager):
    # Embed some actual text chunks to get real vectors
    chunks = ["This is the first chunk.", "This is the second chunk.", "This is the third chunk.",
              "This is the fourth chunk.", "This is the fifth chunk."]
    cluster_manager.embed_documents_with_progress(chunks)

    # Perform clustering
    cluster_manager.cluster_document(n_clusters=2)

    # Find the closest representatives
    representatives = cluster_manager.find_n_closest_representatives(n=2)

    # Ensure the correct number of representatives are returned
    assert len(representatives) == 2, "There should be representatives for each of the 2 clusters"
    for cluster_label, closest_indices in representatives:
        assert len(closest_indices) == 2, "Each cluster should have 2 closest representatives"