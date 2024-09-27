from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, DirectoryLoader
# from .multimedia_loader import MultimediaLoader
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, source: str, type: str, filter: str = None):
        """
        Initialize the DocumentLoader with a source.

        Parameters
        ----------
        source : str
            The path to a document file or a URL to a webpage.
        """
        self.source = source
        self.type = type
        self.glob = filter
        self.text = ""

    def __call__(self) -> str:
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
        if self.source.lower().endswith('.pdf'):
            self._load_pdf()
        elif self.source.lower().startswith('http'):
            self._load_webpage()
        elif self.type == "directory":
            self._load_directory()
        # else:
            # loader = MultimediaLoader(self.source)
            # loader()
        
       

    def _load_pdf(self) -> None:
        """
        Load text content from a PDF file.
        """
        loader = PyPDFLoader(self.source)
        docs = loader.load()
        self.text = ''.join(doc.page_content for doc in docs)

    def _load_webpage(self) -> None:
        """
        Load text content from a webpage.
        """
        loader = WebBaseLoader(self.source)
        docs = loader.load()
        self.text = docs[0].page_content if docs else ""
    
    def _load_directory(self) -> None:
        loader = DirectoryLoader(self.source, self.glob)
        docs = loader.load()
        print(docs)



if __name__ == "__main__":
    # Load text from a PDF file
    #pdf_loader = DocumentLoader("https://arxiv.org/pdf/2105.01697.pdf","pdf")
    #print(pdf_loader())  

    # Load text from a webpage
    #web_loader = DocumentLoader("https://arxiv.org/abs/2105.01697","web")
    #print(web_loader())
    
    dir_loader = DocumentLoader("/Users/balajiviswanathan/Invento/pathak","directory","**/*.xlsx")
    print(dir_loader())