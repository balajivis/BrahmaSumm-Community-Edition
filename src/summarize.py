from models.models import ModelManager
from chunking.chunking import ChunkManager
from doc_loaders.doc_loader import DocumentLoader
from clustering.clustering import ClusterManager
from visualize.visualize import Visualizer

class Summarizer:

  def __init__(self, config_path):
    self.model_manager = ModelManager(config_path)
    self.model_manager.load_llm_groq()
    self.model_manager.load_embedding_model()
    self.chunk_manager = ChunkManager(config_path)
    self.cluster_manager = ClusterManager(self.model_manager.embedding_model, config_path)
    self.Visualizer = Visualizer(config_path)
    
  def __call__(self, source: str) -> str:
    print("Loading document...")
    doc_loader = DocumentLoader(source)
    text = doc_loader()
    print("Chunking text...")
    self.processed_text = self.chunk_manager.preprocess_text(text)
    self.chunk_manager.flexible_chunk(self.processed_text)
    chunks = self.chunk_manager.get_chunks()
    
    self.cluster_manager.embed_documents_with_progress(chunks)
    labels, cluster_centers = self.cluster_manager.cluster_document()
    print("Number of clusters:", len(cluster_centers))
    representatives = self.cluster_manager.find_n_closest_representatives()
    
    print("Finding themes...")
    themes, cluster_content = self.find_themes_for_clusters(chunks, representatives)
    
    # Print labels properly
    Visualizer.print_labels_in_grid(labels)
    for cluster_label, theme in themes.items():
      print(f"Cluster {cluster_label}: Theme = {theme}")
      
    print("Creating the final summary...")
    self.combined_content = " ".join(cluster_content.values())
    self.final_summary = self.model_manager.llm_groq.invoke(f"Summarize this content in a fairly detailed manner without over simplification {self.combined_content}").content
    
    return self.final_summary
  
  def get_analysis(self) -> str:
    total_tokens = self.model_manager.count_tokens(self.processed_text)
    chunk_words = self.chunk_manager.get_word_count_per_chunk()
    total_chunks = self.chunk_manager.get_total_chunks()
    total_words = self.chunk_manager.get_total_words()
    tokens_sent_tokens = self.model_manager.count_tokens(self.combined_content)
    
    return chunk_words, total_chunks, total_words, total_tokens, tokens_sent_tokens
  
  def find_suitable_theme(self,chunk_text):
    #We will use Groq with this
    return self.model_manager.llm_groq.invoke(f"Extract a key theme from this chunk from a web page: {chunk_text} that is used for topic modeling. It might contain boiler plate, \
    elements from web and other irrelevant content and if that is so, say so. Anything related to engagement metrics with number of likes \
    or other website part such as    Contact Us Privacy Policy Copyright Policy Accessibility Statement   InstagramOpens in a new window \
  should be called out as unrelated. Be very concise limiting to less than 5 words").content
  
  def find_themes_for_clusters(self,chunks, representatives):
    themes = {}
    cluster_content = {}
    #print(representatives)
    for cluster_label, representative_indices in representatives:
        first_representative_chunk = chunks[representative_indices[0]]
        theme = self.find_suitable_theme(first_representative_chunk)
        themes[cluster_label] = theme  # Store the theme in the dictionary
        #print(f"Theme for Cluster {cluster_label}: {theme}")

        combined_chunks = " ".join([chunks[index] for index in representative_indices])
        # Store the combined chunks in the cluster_content dictionary
        cluster_content[cluster_label] = combined_chunks

        #print(f"Cluster Content for Cluster {cluster_label}: {combined_chunks}\n")

    return themes, cluster_content
    

if __name__ == '__main__':
  config_path = 'config.yaml'
  summarizer = Summarizer(config_path)
  summary = summarizer('https://www.whitehouse.gov/state-of-the-union-2024/')
  print(summary)
  chunk_words, total_chunks, total_words, total_tokens, tokens_sent_tokens = summarizer.get_analysis()
  print(f"Chunk words: {chunk_words},\n Total chunks: {total_chunks}, Total words: {total_words}, Total tokens in original text: {total_tokens}, Total tokens sent to llm: {tokens_sent_tokens}")
  



