from tqdm import tqdm
import yaml
from sklearn.cluster import KMeans
import numpy as np

class ClusterManager:
    def __init__(self, embedding_model, config_path):
        self.embedding_model = embedding_model
        self.config = yaml.safe_load(open(config_path, 'r'))
        self.vectors = []

    def embed_documents_with_progress(self,chunks, batch_size=None):
        """
        Embed documents with progress reporting.
        
        :param chunks: List of document chunks to embed
        :param batch_size: Number of chunks to process in each batch
        :return: List of embeddings
        """        
        if batch_size is None:
            batch_size = self.config['embed_batch_size']
            
        # Loop through the chunks and embed them in batches
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding documents"):
            # Get the current batch of chunks
            batch_chunks = chunks[i:i + batch_size]
            
            # Embed the batch and store the results
            batch_embeddings = self.embedding_model.embed_documents(batch_chunks)
            self.vectors.extend(batch_embeddings)
            
            #print(f"Processed {i + len(batch_chunks)} of {len(chunks)} chunks.")
    
    def get_vectors(self):
        return self.vectors
    
    #Lets cluster our results to find beautiful themes
    
    def cluster_document(self,n_clusters=None):
        if n_clusters is None:
            n_clusters = self.config['n_clusters']
            
        if n_clusters > len(self.vectors):
            n_clusters = len(self.vectors)
            
        self.kmeans = KMeans(n_clusters, random_state=0, n_init="auto")
        self.labels = self.kmeans.fit_predict(self.vectors)
        return self.labels, self.kmeans.cluster_centers_

    
    def find_n_closest_representatives(self, n=None):
        cluster_centers = self.kmeans.cluster_centers_
        if n is None:
            n = self.config['n_closest_representatives']
            
        if n > len(cluster_centers):
            n = len(cluster_centers)
            
        """
        Find the `n` closest chunks to the center for each cluster and associate them with their cluster labels.

        :param vectors: Array of vector embeddings for the chunks
        :param labels: Array of cluster labels for each chunk
        :param cluster_centers: Array of cluster center coordinates (centroids)
        :param n: Number of closest chunks to return for each cluster
        :return: List of tuples (cluster_label, indices of `n` closest chunks for each cluster)
        """
        num_clusters = cluster_centers.shape[0]  # Number of clusters
        representatives_with_labels = []  # List to hold (label, indices of representative chunks)

        # Loop through each cluster center
        for i in range(num_clusters):
            # Compute distances between all vectors and the current cluster center
            distances = np.linalg.norm(self.vectors - cluster_centers[i], axis=1)

            # Find the indices of the `n` closest points
            closest_indices = np.argsort(distances)[:n]  # Sort and take the top `n`

            # Get the cluster label (this corresponds to the cluster `i`)
            cluster_label = i

            # Append the label and closest indices to the result list
            representatives_with_labels.append((cluster_label, closest_indices))

        return representatives_with_labels