import whisper
from pydub import AudioSegment
model = whisper.load_model("medium")


# We will be using a sample from librivox for the demo
# https://librivox.org/julius-caesar-by-william-shakespeare/

# Function to split the audio file into chunks
def split_audio(audio_file_path, chunk_length_ms=30000): # chunk_length_ms is 30 seconds by default
    audio = AudioSegment.from_file(audio_file_path)
    chunks = []
    
    for i in range(0, len(audio), chunk_length_ms):
        chunks.append(audio[i:i+chunk_length_ms])
    return chunks

# Function to transcribe audio
def transcribe_audio_chunks(chunks):
    transcription = ""
    for i, chunk in enumerate(chunks):
        # Export chunk to a temporary file
        chunk_file = f'audio/chunk{i}.wav'
        chunk.export(chunk_file, format="wav")
        
        with open(chunk_file, 'rb') as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
            transcription += response['text'] + " "  # Add a space between chunks to separate words
    return transcription

# Main function to transcribe an audio file
def transcribe_audio(audio_file_path):
    chunks = split_audio(audio_file_path)
    transcription = transcribe_audio_chunks(chunks)
    return transcription