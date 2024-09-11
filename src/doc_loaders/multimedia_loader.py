# We will assume that the models are cached already

import os
import tempfile
from tqdm import tqdm
import logging
from src.chunking.audiochunking import split_audio
import whisper
from pytubefix import YouTube

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimediaLoader:
    def __init__(self, source: str):
        """
        Initialize the DocumentLoader with a source.

        Parameters
        ----------
        source : str
            The path to a PDF file or a URL to a webpage.
        """
        self.source = source
        self.text = ""
        self.model = whisper.load_model("tiny")
        
    def __call__(self) -> str:
        """
        Call the object to load the text from the source.

        Returns
        -------
        str
            The loaded text content.
        """
        self.load()
        return self.text

    def load(self) -> None:
        """
        Load text content from the source. Detects if the source is a PDF or a webpage.

        Raises
        ------
        ValueError
            If the source is not a valid PDF file or a URL.
        """
        if self.source.lower().endswith('.mp3'):
            logger.info("Loading audio...")
            self._load_audio(self.source)
        elif self.source.startswith("https://www.youtube.com/watch?v="):
            logger.info("Loading youtube...")
            self._load_yt()
        else:
            raise ValueError("Invalid source. Must be a PDF file or a URL.")

    def _load_yt(self) -> None:
        """
        Load YouTube video content from the source.

        Raises
        ------
        ValueError
            If the source is not a valid YT URL.
        """
        # check for valid yt url
        if not self.source.startswith("https://www.youtube.com/watch?v="):
            raise ValueError("Invalid YT URL. Must be a valid YouTube URL.")
        
        logger.info("Loading YouTube video...")
        yt = YouTube(self.source)
        audio = yt.streams.filter(only_audio=True).first()
        
        # Create a temporary directory
        temp_dir = tempfile.gettempdir()
        yt_audio_path = os.path.join(temp_dir, f"yt_audio.wav")
        #store audio in wav format
        audio.download(filename=yt_audio_path)
        
        logger.info("Sending the audio...")
        self._load_audio(yt_audio_path)
        os.remove(yt_audio_path)
        
    def _load_audio(self,audio) -> None:
        """
        Trascribe audio content from the source.
        """
        logger.info("Splitting audio into chunks...")
        audio_chunks = split_audio(audio)
        print(f"Number of audio chunks: {len(audio_chunks)}")
        # Give a progress bar
        for i, chunk in tqdm(enumerate(audio_chunks), desc="Transcribing audio chunks", total=len(audio_chunks)):
            wav_file = self.audio_chunk_to_wav(chunk, i)
            
            result = self.model.transcribe(wav_file)
            #self.text += result["start"] + "-" + result["end"] + ":" + result["text"] + "\n"
            self.text += result["text"]
        
            print(f"Chunk {i+1}: {result['text']}")
        os.remove(wav_file)
    
    def audio_chunk_to_wav(self,chunk, chunk_index):
        """
        Converts an audio chunk to a WAV file.

        :param chunk: An AudioSegment chunk.
        :param chunk_index: Index of the chunk to generate a unique file name.
        :return: The path to the WAV file.
        """
        # Create a temporary directory
        temp_dir = tempfile.gettempdir()
        wav_file_path = os.path.join(temp_dir, f"chunk_{chunk_index}.wav")
        
        # Export the chunk to a wav file
        chunk.export(wav_file_path, format="wav")
        
        return wav_file_path

if __name__ == "__main__":
    # We will be using a sample from librivox for the demo
    # https://librivox.org/julius-caesar-by-william-shakespeare/
    #loader = MultimediaLoader("samples/juliuscaesar_01_shakespeare_64kb.mp3")
    loader =  MultimediaLoader("https://www.youtube.com/watch?v=4Prc1UfuokY")
    text = loader()
    print(text)

