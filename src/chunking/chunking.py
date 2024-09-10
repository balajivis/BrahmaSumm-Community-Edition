import re
import yaml
import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkManager:
    """
    The ChunkManager class is responsible for dividing large text documents into chunks
    based on word count, while preserving sentence and paragraph boundaries.
    It ensures that chunks are created flexibly based on a target word count, 
    allowing for slight variations in chunk size.
    """

    def __init__(self, config_path):
        """
        Initializes ChunkManager by loading the configuration.

        :param config_path: Path to the configuration file containing chunking parameters.
        """
        logger.info("Initializing ChunkManager with configuration from %s", config_path)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.flexibility = self.config.get('chunk_flexibility', 0.25)  # Default to 25% flexibility
        self.target_words = self.config.get('target_words', 100)       # Default to 100 words per chunk
        self.chunks = []
        logger.info("ChunkManager initialized with target_words=%d and flexibility=%.2f", self.target_words, self.flexibility)

    def preprocess_text(self, text):
        """
        Cleans the input text by replacing multiple newlines and removing non-ASCII characters.
        
        :param text: The raw text to be preprocessed.
        :return: Cleaned text.
        """
        logger.info("Preprocessing text: removing extra newlines and non-ASCII characters")
        text = re.sub(r'\n+', '\n\n', text)
        text = re.sub(r'\t+', '\t', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        logger.debug("Preprocessed text: %s", text[:100])  # Log a preview of the preprocessed text
        return text

    def flexible_chunk(self, text, target_words=None, flexibility=None):
        """
        Divides the text into chunks based on a target word count and flexible chunk size.
        Ensures that each chunk contains a minimum of 75% and a maximum of 125% of the target word count.

        :param text: The preprocessed text to be chunked.
        :param target_words: Optional; target word count per chunk. If not provided, the default is used.
        :param flexibility: Optional; the percentage flexibility for chunk size. If not provided, the default is used.
        """
        if target_words is None:
            target_words = self.target_words
        if flexibility is None:
            flexibility = self.flexibility

        min_words = int(target_words * (1 - flexibility))  # Minimum words per chunk (75%)
        max_words = int(target_words * (1 + flexibility))  # Maximum words per chunk (125%)

        logger.info("Chunking text with target_words=%d, flexibility=%.2f, min_words=%d, max_words=%d", 
                    target_words, flexibility, min_words, max_words)

        paragraphs = re.split(r'\n\n', text)  # Split by paragraphs
        chunks = []
        current_chunk = []
        current_word_count = 0

        def finalize_chunk(force=False):
            """Finalizes the current chunk if it meets the minimum word count or if forced."""
            if current_chunk and (force or current_word_count >= min_words):
                logger.debug("Finalizing chunk with %d words", current_word_count)
                chunks.append(' '.join(current_chunk))

        def process_paragraph(paragraph):
            """Processes a paragraph, splitting it into sentences and adding to chunks."""
            nonlocal current_chunk, current_word_count
            sentences = re.split(r'(?<=[.!?]) +', paragraph)  # Split by sentence
            for sentence in sentences:
                sentence_word_count = len(sentence.split())

                # If adding the sentence keeps the chunk under the max limit, add it
                if current_word_count + sentence_word_count <= max_words:
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count

                # If the chunk is at or above the minimum, finalize it
                if current_word_count >= min_words:
                    finalize_chunk()
                    current_chunk = []
                    current_word_count = 0

                # If adding the sentence exceeds the max, finalize and start a new chunk
                elif current_word_count + sentence_word_count > max_words:
                    finalize_chunk()
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count

        # Process each paragraph and chunk the text accordingly
        for paragraph in paragraphs:
            para_word_count = len(paragraph.split())
            logger.debug("Processing paragraph with %d words", para_word_count)

            # If the paragraph itself is smaller than the max size, add it as a chunk
            if para_word_count <= max_words:
                current_chunk.append(paragraph)
                current_word_count += para_word_count
                if current_word_count >= min_words:
                    finalize_chunk()
                    current_chunk = []
                    current_word_count = 0
            else:
                process_paragraph(paragraph)

        # Finalize any remaining text as the last chunk
        finalize_chunk(force=True)
        self.chunks = chunks
        logger.info("Chunking completed with %d chunks", len(chunks))

    def get_word_count_per_chunk(self):
        """
        Returns the word count for each chunk.

        :return: List of word counts per chunk.
        """
        word_counts = [len(chunk.split()) for chunk in self.chunks]
        logger.debug("Word counts per chunk: %s", word_counts)
        return word_counts

    def get_total_chunks(self):
        """
        Returns the total number of chunks created.

        :return: Total number of chunks.
        """
        total_chunks = len(self.chunks)
        logger.info("Total chunks: %d", total_chunks)
        return total_chunks

    def get_chunks(self):
        """
        Returns the list of text chunks.

        :return: List of chunks.
        """
        return self.chunks

    def get_total_words(self):
        """
        Returns the total word count across all chunks.

        :return: Total word count.
        """
        total_words = sum(self.get_word_count_per_chunk())
        logger.info("Total word count: %d", total_words)
        return total_words


if __name__ == '__main__':
    # Example usage of ChunkManager
    text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Nulla euismod, nisl eget aliquam ultricies, nunc nisl ultricies 
    nunc, sit amet aliquam nisl nunc eget nisl.
    
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Nulla euismod, nisl eget aliquam ultricies, nunc nisl ultricies 
    nunc, sit amet aliquam nisl nunc eget nisl.
    """
    chunk_manager = ChunkManager('config/config.yaml')
    
    processed_text = chunk_manager.preprocess_text(text)
    chunk_manager.flexible_chunk(processed_text)
    
    logger.info("Word count per chunk: %s", chunk_manager.get_word_count_per_chunk())
    logger.info("Total chunks: %d", chunk_manager.get_total_chunks())
    logger.info("Total words: %d", chunk_manager.get_total_words())
    logger.debug("Chunks: %s", chunk_manager.get_chunks())