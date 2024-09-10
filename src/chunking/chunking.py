import re
import yaml
# Explain what this does in detail in a docstring
"""
    The ChunkManager class is designed to manage the chunking of text documents based on word count 
    and sentence/paragraph boundaries.
"""

class ChunkManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.flexibility = self.config['chunk_flexbility']
        self.target_words = self.config['target_words']
        self.chunks = []
            
    def preprocess_text(self,text):
        """
        Preprocess the text by removing extra paragraph breaks (\n\n)
        and non-English characters.
        """
        # Step 1: Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n\n', text)
        text = re.sub(r'\t+', '\t', text)

        # Step 2: Remove non-ASCII characters (non-English characters)
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        return text

    def flexible_chunk(self,text, target_words=None, flexibility=None):
        """
        Chunk the document based on word count and sentence/paragraph boundaries,
        ensuring that chunks are not finalized until at least 75% of the target word count.
        """
        if target_words is None:
            target_words = self.target_words
        if flexibility is None:
            flexibility = self.flexibility
        
        min_words = int(target_words * (1 - flexibility))  # 75% of target
        max_words = int(target_words * (1 + flexibility))  # 125% of target

        # Step 1: Split the text into paragraphs and sentences
        paragraphs = re.split(r'\n\n', text)  # Paragraphs based on \n\n

        chunks = []
        current_chunk = []
        current_word_count = 0

        def finalize_chunk(force=False):
            """Helper function to finalize and reset the current chunk."""
            if current_chunk and (force or current_word_count >= min_words):  # Ensure last chunk is added even if under 75%
                chunks.append(' '.join(current_chunk))

        def process_paragraph(para):
            """Helper function to handle each paragraph."""
            nonlocal current_chunk, current_word_count

            sentences = re.split(r'(?<=[.!?]) +', para)  # Split paragraph by sentences
            for sentence in sentences:
                sentence_word_count = len(sentence.split())

                # If the current chunk + this sentence stays under 125% of the target:
                if current_word_count + sentence_word_count <= max_words:
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count

                # Finalize if we reach the max word count
                if current_word_count >= min_words:
                    finalize_chunk()
                    current_chunk = []
                    current_word_count = 0
                elif current_word_count + sentence_word_count > max_words:
                    finalize_chunk()
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count

        for paragraph in paragraphs:
            para_word_count = len(paragraph.split())

            # If the entire paragraph is smaller than the target, add it directly
            if para_word_count <= max_words:
                current_chunk.append(paragraph)
                current_word_count += para_word_count
                if current_word_count >= min_words:
                    finalize_chunk()
                    current_chunk = []
                    current_word_count = 0
            else:
                # Process paragraph sentence by sentence
                process_paragraph(paragraph)

        # Add any remaining words in the final chunk (force finalization)
        finalize_chunk(force=True)

        self.chunks = chunks
        
    def get_word_count_per_chunk(self):
        word_counts = [len(chunk.split()) for chunk in self.chunks]
        return word_counts
    
    def get_total_chunks(self):
        return len(self.chunks)
    
    def get_chunks(self):
        return self.chunks

    def get_total_words(self):
        word_counts = [len(chunk.split()) for chunk in self.chunks]
        return sum(word_counts)

if __name__ == '__main__':
    text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Nulla euismod, nisl eget aliquam ultricies, nunc nisl ultricies 
    nunc, sit amet aliquam nisl nunc eget nisl.
    
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Nulla euismod, nisl eget aliquam ultricies, nunc nisl ultricies 
    nunc, sit amet aliquam nisl nunc eget nisl.
    """
    chunk_manager = ChunkManager('config.yaml')
    
    processed_text = chunk_manager.preprocess_text(text)
    print(processed_text)
    chunk_manager.flexible_chunk(processed_text)
    print(chunk_manager.get_word_count_per_chunk())
    print(chunk_manager.get_total_chunks())
    print(chunk_manager.get_total_words())
    print(chunk_manager.get_chunks())