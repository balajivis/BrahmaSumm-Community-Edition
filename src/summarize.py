import logging
import yaml
from src.models.models import ModelManager
from src.chunking.chunking import ChunkManager
from src.doc_loaders.doc_loader import DocumentLoader
from src.clustering.clustering import ClusterManager
from src.visualize.visualize import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, config_path):
        """
        Initializes the Summarizer class with models, chunking manager, clustering, and visualization.
        Loads the prompts and the necessary models as per the configuration.
        """
        self.model_manager = ModelManager(config_path)
        self.prompts = self.load_prompts()
        self.model_manager.load_llm_groq()
        self.model_manager.load_embedding_model()
        self.chunk_manager = ChunkManager(config_path)
        self.cluster_manager = ClusterManager(self.model_manager.embedding_model, config_path)
        self.visualizer = Visualizer(config_path)

    def load_prompts(self):
        """
        Loads prompts from the specified YAML configuration file.
        
        :param config_path: Path to the YAML file containing prompts
        :return: Loaded prompts dictionary
        """
        with open('config/prompts.yaml', 'r') as file:
            return yaml.safe_load(file)

    def __call__(self, source: str) -> str:
        """
        Processes the input document through loading, chunking, clustering, and summarizing.

        :param source: The source document (URL or file path)
        :return: Final summary of the document
        """
        logger.info("Loading document...")
        doc_loader = DocumentLoader(source)
        text = doc_loader()

        logger.info("Chunking text...")
        self.processed_text = self.chunk_manager.preprocess_text(text)
        self.chunk_manager.flexible_chunk(self.processed_text)
        chunks = self.chunk_manager.get_chunks()

        logger.info("Embedding and clustering...")
        self.cluster_manager.embed_documents_with_progress(chunks)
        labels, cluster_centers = self.cluster_manager.cluster_document()
        logger.info(f"Number of clusters: {len(cluster_centers)}")
        
        representatives = self.cluster_manager.find_n_closest_representatives()
        logger.info("Finding themes for each cluster...")

        themes, cluster_content = self.find_themes_for_clusters(chunks, representatives)

        # Print labels in grid format
        self.visualizer.print_labels_in_grid(labels)
        for cluster_label, theme in themes.items():
            logger.info(f"Cluster {cluster_label}: Theme = {theme}")

        logger.info("Creating the final summary...")
        self.combined_content = " ".join(cluster_content.values())
        self.final_summary = self.model_manager.llm_groq.invoke(
            f"Summarize this content in a fairly detailed manner without oversimplification {self.combined_content}"
        ).content

        return self.final_summary

    def get_analysis(self):
        """
        Provides detailed analysis of the processed document, including chunk sizes, total tokens, and word counts.

        :return: Tuple containing chunk words, total chunks, total words, total tokens, and tokens sent to LLM
        """
        total_tokens = self.model_manager.count_tokens(self.processed_text)
        chunk_words = self.chunk_manager.get_word_count_per_chunk()
        total_chunks = self.chunk_manager.get_total_chunks()
        total_words = self.chunk_manager.get_total_words()
        tokens_sent_tokens = self.model_manager.count_tokens(self.combined_content)
        
        return chunk_words, total_chunks, total_words, total_tokens, tokens_sent_tokens

    def find_suitable_theme(self, chunk_text):
        """
        Uses LLM to extract the most relevant theme from the given chunk.

        :param chunk_text: A chunk of text from the document
        :return: The extracted theme for the given chunk
        """
        prompt = self.prompts['find_suitable_theme_prompt'].format(chunk_text=chunk_text)
        return self.model_manager.llm_groq.invoke(prompt).content

    def find_themes_for_clusters(self, chunks, representatives):
        """
        Finds a suitable theme for each cluster and combines the chunks for each representative.

        :param chunks: The chunked text from the document
        :param representatives: The representative chunks closest to the cluster centers
        :return: A dictionary of themes for each cluster and combined content for each cluster
        """
        themes = {}
        cluster_content = {}

        for cluster_label, representative_indices in representatives:
            first_representative_chunk = chunks[representative_indices[0]]
            theme = self.find_suitable_theme(first_representative_chunk)
            themes[cluster_label] = theme  # Store the theme in the dictionary

            # Combine chunks for this cluster
            combined_chunks = " ".join([chunks[index] for index in representative_indices])
            cluster_content[cluster_label] = combined_chunks

        return themes, cluster_content
      
def main():
    config_path = 'config/config.yaml'
    summarizer = Summarizer(config_path)

    themes = summarizer.find_suitable_theme("BrahmaSumm is an advanced document summarization and visualization tool designed to streamline document management, knowledge base creation, and chatbot enhancement. By leveraging cutting-edge chunking and clustering techniques, BrahmaSumm reduces token usage sent to Large Language Models (LLMs) by up to 99%, while maintaining the quality of content. The tool provides intuitive document processing, stunning visualizations, and efficient querying across multiple formats.")
    print(themes)

    summary = summarizer('https://www.whitehouse.gov/state-of-the-union-2024/')
    print(summary)

    chunk_words, total_chunks, total_words, total_tokens, tokens_sent_tokens = summarizer.get_analysis()
    print(f"Chunk words: {chunk_words}\n Total chunks: {total_chunks}\n Total words: {total_words}\n Total tokens in original text: {total_tokens}\n Total tokens sent to LLM: {tokens_sent_tokens}")

if __name__ == "__main__":
    main()


