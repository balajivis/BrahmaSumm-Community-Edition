import logging
import yaml
import json
from models.models import ModelManager
from chunking.textchunking import ChunkManager
from doc_loaders.doc_loader import DocumentLoader
from clustering.clustering import ClusterManager
from visualize.visualize import Visualizer
from outputs.report_generate import create_final_report

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
        # self.model_manager.load_llm()
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

    def __call__(self, source: str, type: str) -> dict:
        """
        Processes the input document through loading, chunking, clustering, and summarizing.
        It returns a dictionary with all necessary data for report generation.

        :param source: The source document (URL or file path)
        :return: A dictionary containing the final summary, analysis, UMAP cluster details, and themes.
        """
        # Step 1: Load the document
        logger.info("Loading document...")
        doc_loader = DocumentLoader(source,type)
        text = doc_loader()

        # Step 2: Preprocess and chunk the document
        logger.info("Chunking text...")
        self.processed_text = self.chunk_manager.preprocess_text(text)
        self.chunk_manager.flexible_chunk(self.processed_text)
        chunks = self.chunk_manager.get_chunks()

        # Step 3: Embed the document and run clustering
        logger.info("Embedding and clustering...")
        self.cluster_manager.embed_documents_with_progress(chunks)
        labels, cluster_centers = self.cluster_manager.cluster_document()
        logger.info(f"Number of clusters: {len(cluster_centers)}")

        # Step 4: Find representatives and themes for each cluster
        representatives = self.cluster_manager.find_n_closest_representatives()
        logger.info("Finding themes for each cluster...")
        themes, cluster_content = self.find_themes_for_clusters_slow(chunks, representatives)
        
      
        # Step 5: Generate UMAP visualization
        logger.info("Creating the visualization...")
        #print("Labels:", labels)
        #print("Themes keys:", themes.keys())
        
        self.visualizer.plot_clusters_with_umap(
            self.cluster_manager.vectors, 
            themes, 
            labels, 
            n_neighbors=25, 
            min_dist=0.001, 
            spread=0.8, 
            length=12, 
            width=8, 
            output_image='reports/umap_clusters.png'
        )

        # Step 6: Generate the final summary using LLM
        logger.info("Creating the final summary...")
        self.combined_content = " ".join(cluster_content.values())
        prompt = self.prompts['create_summary_prompt'].format(combined_content=self.combined_content)
        
        final_summary = self.model_manager.llm.invoke(prompt).content
      
        # Step 7: Perform analysis on the document
        chunk_words, total_chunks, total_words, total_tokens, tokens_sent_tokens = self.get_analysis()

        # Step 8: Populate the data dictionary
        data = {
            'summary': final_summary,
            'labels': labels,
            'chunk_words': chunk_words,
            'total_chunks': total_chunks,
            'total_words': total_words,
            'total_tokens': total_tokens,
            'tokens_sent_tokens': tokens_sent_tokens,
            'themes': themes,
            'umap_image_path': 'reports/umap_clusters.png'
        }
        
        

        return data

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
        logger.info("Finding suitable theme for chunk: %s", chunk_text)
        return self.model_manager.llm.invoke(prompt).content

    def find_themes_for_clusters_slow(self, chunks, representatives):
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
            logger.info("Found theme for cluster %s: %s", cluster_label, theme)

            # Combine chunks for this cluster
            combined_chunks = " ".join([chunks[index] for index in representative_indices])
            cluster_content[cluster_label] = combined_chunks

        print(themes)
        return themes, cluster_content
      
    # TODO: Fix this -- current unused due to various issues in formatting
    def find_themes_for_clusters(self, chunks, representatives):
        """
        Finds suitable themes for all clusters in a single LLM call and combines the chunks for each representative.

        :param chunks: The chunked text from the document
        :param representatives: The representative chunks closest to the cluster centers
        :return: A dictionary of themes for each cluster and combined content for each cluster
        """
        # Step 1: Prepare the text chunks for the LLM prompt
        clusters_data = {}
        cluster_content = {}
        themes = {}
        
        for cluster_label, representative_indices in representatives:
            # Get the first representative chunk to represent the cluster
            first_representative_chunk = chunks[representative_indices[0]]
            
            # For theme we just need the first representative, but the full list of chunks for the cluster
            combined_chunks = " ".join([chunks[index] for index in representative_indices])
            cluster_content[cluster_label] = combined_chunks
            
            # Prepare the data for each cluster
            clusters_data[cluster_label] = {
                "representative_text": first_representative_chunk,
                "combined_text": " ".join([chunks[index] for index in representative_indices])
            }

        # Step 2: Build the LLM prompt
        prompt = self.prompts['find_suitable_theme_prompt_multiple'].format(first_representative_chunk=first_representative_chunk)
        
        # Step 3: Call the LLM once for all clusters
        response = self.model_manager.llm.invoke(prompt).content

        
        print(response)

        # Step 4: Process and clean the response
        # LLM might return extra text alongside JSON, so let's clean it
        start_idx = response.find("{")  # Find the start of the JSON
        end_idx = response.find("}")  # Find the end of the JSON
        if start_idx == -1 or end_idx == -1:
            print("No valid JSON found in the LLM response")
            return {}, {}
        
        # Extract the JSON part of the response
        json_response = response[start_idx:end_idx+1]

        # Step 5: Parse the JSON response
        try:
            parsed_response = json.loads(json_response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return {}, {}


         # Step 6: Initialize dictionaries for themes and summaries
        # Iterate over the parsed response and store the themes and summaries
        for cluster_label, cluster_data in parsed_response.items():
            theme = cluster_data.get("theme", "No theme available")
            themes[cluster_label] = theme

        
        print(themes)
        return themes, cluster_content
      

def main():
    config_path = 'config/config.yaml'
    summarizer = Summarizer(config_path)
    print(summarizer.find_suitable_theme("Who is John Galt!"))

    data = summarizer('https://medium.com/@balajivis/whats-so-challenging-about-building-chatbots-drawing-lessons-from-the-trenches-1ca7343c6e3d',"web")
    # data = summarizer('https://abc7.com/read-harris-trump-presidential-debate-transcript/15289001/','web')
    # data = summarizer('https://www.whitehouse.gov/state-of-the-union-2024/',"web")
    #data = summarizer('https://d18rn0p25nwr6d.cloudfront.net/CIK-0001921963/77018dae-bae9-4c33-8eaf-fa6685991719.pdf',"pdf")
    
    create_final_report(data,report_path='reports/final_report.pdf')
    
    print(data["summary"])

    chunk_words, total_chunks, total_words, total_tokens, tokens_sent_tokens = summarizer.get_analysis()
    print(f"Total chunks: {total_chunks}\n Total words: {total_words}\n Total tokens in original text: {total_tokens}\n Total tokens sent to LLM: {tokens_sent_tokens}")

if __name__ == "__main__":
    main()