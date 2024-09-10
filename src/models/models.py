import os
import yaml
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings

class ModelManager:
    def __init__(self, config_path):
        load_dotenv()
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize models
        self.llm_groq = None
        self.embedding_model = None

    def load_llm_groq(self):
        """
        Load the LLM Groq model. If it has not been loaded, construct it.
        Returns the model.
        """
        if not self.llm_groq:
            self.llm_groq = ChatGroq(
                model_name=self.config['llm_model'],
                api_key=os.getenv("GROQ_API_KEY")
            )
        return self.llm_groq

    def load_embedding_model(self):
        """
        Load the Hugging Face embedding model. If it has not been loaded, construct it.
        Returns the model.
        """
        if not self.embedding_model:
            self.embedding_model = OllamaEmbeddings(model=self.config['embedding_model'] )
        return self.embedding_model
    
    def count_tokens(self, text):
        """
        Count the number of tokens in the text.
        """
        return self.llm_groq.get_num_tokens(text,)
    
# main function for testing this
if __name__ == '__main__':
    model_manager = ModelManager('config.yaml')
    model_manager.load_llm_groq()
    model_manager.load_embedding_model()
    
    print(model_manager.count_tokens("Hello world."))
    print(model_manager.llm_groq.invoke("Hello world."))
    print(model_manager.embedding_model.embed_documents(["Hello world."]))
    