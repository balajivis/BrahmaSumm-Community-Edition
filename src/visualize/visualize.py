import umap
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, config_path):
        pass
    
    def print_labels_in_grid(self,labels):
        """
        Print the cluster labels in a grid format.
        :param labels: The list of labels.
        :param row_length: Number of labels to display per row.
        """
        print("Visualizing the document by topic clusters")
        row_length=30
        if len(labels) > 500:
            row_length=50
            
        if len(labels) <= row_length:
            print(labels)
            return
        
        for i in range(0, len(labels), row_length):
            # Slice the labels into rows and print them
            print(labels[i:i+row_length])
            

    def plot_clusters_with_umap(self, vectors, themes, labels, n_neighbors=25, min_dist=0.001, spread=0.8, length=12, width=5, output_image='umap_clusters.png'):
        """
        Plot clusters using UMAP and label them with their corresponding themes, then save to PNG.

        :param vectors: The vector embeddings of the chunks
        :param themes: A dictionary with cluster labels as keys and themes as values
        :param labels: The cluster labels for each vector embedding
        :param n_neighbors: UMAP parameter that controls the number of neighbors to consider
        :param min_dist: UMAP parameter that controls how closely UMAP packs points together
        :param spread: UMAP parameter to control how spread out the clusters are
        :param length: The length of the plot figure
        :param width: The width of the plot figure
        :param output_image: Path to save the PNG file of the plot
        """
        # Step 1: Apply UMAP to reduce the dimensionality of vectors
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, spread=spread, random_state=42)
        embedding = umap_model.fit_transform(vectors)  # This gives a 2D embedding

        # Step 2: Prepare to plot the clusters
        plt.figure(figsize=(length, width))

        # Step 3: Plot each cluster with its corresponding theme label
        unique_labels = list(set(labels))  # Get unique cluster labels

        for cluster_label in unique_labels:
            # Get the indices of vectors that belong to this cluster
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_label]

            # Get the 2D UMAP coordinates for this cluster
            cluster_embedding = embedding[cluster_indices]

            # Get the theme for this cluster, fall back to cluster number if missing
            theme_key = f"Cluster {cluster_label}"
            theme_label = themes.get(cluster_label, f"Cluster {cluster_label}")

            # Debugging: Check if the theme matches the cluster
            #print(f"Plotting cluster {cluster_label}: Theme = {theme_label}")
            #print(f"Cluster {cluster_label} theme exists: {theme_key in themes}")

            # Plot the cluster points with the correct theme label
            plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], label=theme_label, s=50)

        # Step 4: Add labels and title to the plot
        plt.title('Clusters Visualized with UMAP', fontsize=12)
        plt.legend(loc='best', title="Themes")
        plt.grid(True)

        # Step 5: Save the plot to a PNG file
        plt.savefig(output_image, format='png')

        # Step 6: Show the plot (optional)
        plt.show()