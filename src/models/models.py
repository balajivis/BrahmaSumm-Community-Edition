import os
import yaml
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config_path):
        """
        Initialize ModelManager by loading configuration and setting up models.
        :param config_path: Path to the YAML configuration file.
        """
        load_dotenv()
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully from %s", config_path)
        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

        # Initialize models as None to be lazily loaded
        self.llm_groq = None
        self.embedding_model = None

    def load_llm_groq(self):
        """
        Lazily loads the LLM Groq model based on the configuration if it hasn't been loaded yet.
        :return: The loaded LLM Groq model.
        """
        if not self.llm_groq:
            try:
                logger.info("Loading Groq LLM model...")
                self.llm_groq = ChatGroq(
                    model_name=self.config['llm_model'],
                    api_key=os.getenv("GROQ_API_KEY")
                )
                logger.info("Groq LLM model loaded successfully.")
            except KeyError as e:
                logger.error("Missing required config key for LLM: %s", e)
                raise
            except Exception as e:
                logger.error("Error loading Groq LLM model: %s", e)
                raise
        return self.llm_groq

    def load_embedding_model(self):
        """
        Lazily loads the Hugging Face embedding model based on the configuration if it hasn't been loaded yet.
        :return: The loaded embedding model.
        """
        if not self.embedding_model:
            try:
                logger.info("Loading embedding model...")
                self.embedding_model = OllamaEmbeddings(model=self.config['embedding_model'])
                logger.info("Embedding model loaded successfully.")
            except KeyError as e:
                logger.error("Missing required config key for embedding model: %s", e)
                raise
            except Exception as e:
                logger.error("Error loading embedding model: %s", e)
                raise
        return self.embedding_model

    def count_tokens(self, text):
        """
        Counts the number of tokens in the given text using the Groq LLM.
        :param text: The text to count tokens for.
        :return: Number of tokens in the text.
        """
        if not self.llm_groq:
            logger.warning("Groq LLM model is not loaded. Loading the model first...")
            self.load_llm_groq()
        
        try:
            num_tokens = self.llm_groq.get_num_tokens(text)
            logger.info("Successfully counted %d tokens for the given text.", num_tokens)
            return num_tokens
        except Exception as e:
            logger.error("Error counting tokens: %s", e)
            raise


# Main function for testing the ModelManager class
if __name__ == '__main__':
    try:
        model_manager = ModelManager('config/config.yaml')
        model_manager.load_llm_groq()
        model_manager.load_embedding_model()

        print("Token count:", model_manager.count_tokens("Hello world."))
        print("Groq LLM Response:", model_manager.llm_groq.invoke("Hello world."))
        print("Embedding Result:", model_manager.embedding_model.embed_documents(["Hello world."]))
    except Exception as e:
        logger.error("An error occurred: %s", e)