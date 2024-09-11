from pydub import AudioSegment

def split_audio(audio_file_path, chunk_length_ms=30000):  # Default is 30 seconds
    """
    Splits the audio file into chunks of specified length.
    
    :param audio_file_path: Path to the audio file.
    :param chunk_length_ms: Length of each chunk in milliseconds.
    :return: A list of audio chunks.
    """
    audio = AudioSegment.from_file(audio_file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks