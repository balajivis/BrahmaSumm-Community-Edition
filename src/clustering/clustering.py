import logging
from tqdm import tqdm
import yaml
from sklearn.cluster import KMeans
import numpy as np

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterManager:
    """
    The ClusterManager class is responsible for embedding document chunks
    and clustering them to identify themes or groups. It uses KMeans clustering
    and provides utilities for finding representative chunks close to the cluster centers.
    """

    def __init__(self, embedding_model, config_path):
        """
        Initializes the ClusterManager with an embedding model and configuration file.

        :param embedding_model: Model used for embedding document chunks.
        :param config_path: Path to the YAML configuration file.
        """
        self.embedding_model = embedding_model
        self.config = yaml.safe_load(open(config_path, 'r'))
        self.vectors = []
        logger.info("ClusterManager initialized with config from %s", config_path)

    def embed_documents_with_progress(self, chunks, batch_size=None):
        """
        Embed document chunks with progress tracking, using batch processing.

        :param chunks: List of document chunks to embed.
        :param batch_size: Number of chunks to process in each batch. If None, uses the config value.
        """
        if batch_size is None:
            batch_size = self.config.get('embed_batch_size', 10)
        
        logger.info("Embedding %d document chunks in batches of %d", len(chunks), batch_size)

        # Embed chunks in batches with progress tracking
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding documents"):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch_chunks)
            self.vectors.extend(batch_embeddings)

        logger.info("Completed embedding for %d chunks", len(chunks))

    def get_vectors(self):
        """
        Returns the embedded vectors generated from document chunks.

        :return: List of embedded vectors.
        """
        return self.vectors

    def cluster_document(self, n_clusters=None):
        """
        Clusters the embedded document vectors using KMeans.

        :param n_clusters: Number of clusters to form. If None, uses the config value.
        :return: Tuple of cluster labels and cluster centers.
        """
        if n_clusters is None:
            n_clusters = self.config.get('n_clusters', 5)
        
        if n_clusters > len(self.vectors):
            logger.warning("Requested %d clusters, but only %d vectors are available. Adjusting number of clusters.", n_clusters, len(self.vectors))
            n_clusters = len(self.vectors)

        logger.info("Clustering %d vectors into %d clusters", len(self.vectors), n_clusters)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        self.labels = self.kmeans.fit_predict(self.vectors)

        logger.info("Clustering completed. %d clusters formed.", n_clusters)
        return self.labels, self.kmeans.cluster_centers_

    def find_n_closest_representatives(self, n=None):
        """
        Finds the 'n' closest document chunks to each cluster center.

        :param n: Number of closest chunks to return for each cluster. If None, uses the config value.
        :return: List of tuples (cluster_label, indices of closest chunks for each cluster).
        """
        cluster_centers = self.kmeans.cluster_centers_
        
        if n is None:
            n = self.config.get('n_closest_representatives', 3)
        
        if n > len(cluster_centers):
            logger.warning("Requested %d representatives, but only %d clusters available. Adjusting number of representatives.", n, len(cluster_centers))
            n = len(cluster_centers)

        logger.info("Finding %d closest representatives for each of the %d clusters", n, len(cluster_centers))

        num_clusters = cluster_centers.shape[0]  # Number of clusters
        representatives_with_labels = []

        # Loop through each cluster center and find closest chunks
        for i in range(num_clusters):
            distances = np.linalg.norm(self.vectors - cluster_centers[i], axis=1)
            closest_indices = np.argsort(distances)[:n]

            logger.debug("Cluster %d: Closest %d chunks found", i, n)
            representatives_with_labels.append((i, closest_indices))

        logger.info("Closest representatives found for all clusters.")
        return representatives_with_labels