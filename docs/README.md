# BrahmaSumm

**BrahmaSumm** is a tool for document summarization, clustering, and visualization, designed to reduce token count and enable better document querying and knowledge base creation.

Make sure you have the requirements such as Groq and embedding models from Ollama as defined in the config.yaml file.

##
For audio transcription install ffmpeg (brew install ffmpeg in Mac or apt-get install ffmpeg in Linux)
# !brew install libmagic
# !brew install poppler
# !brew install tesseract

Running the tests
pytest (from root folder)

To find out sizes of all pip packages installed
pip list \
  | tail -n +3 \
  | awk '{print $1}' \
  | xargs pip show \
  | grep -E 'Location:|Name:' \
  | cut -d ' ' -f 2 \
  | paste -d ' ' - - \
  | awk '{print $2 "/" tolower($1)}' \
  | xargs du -sh 2> /dev/null \
  | sort -hr


