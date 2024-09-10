from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

class DocumentLoader:
    """
    Load text from a source, either a PDF file or a webpage.

    Parameters
    ----------
    source : str
        Either a path to a PDF file or a URL to a webpage.

    Attributes
    ----------
    source : str
        The source from which the document is loaded.
    text : str
        The text content of the source.
    """

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
        if self.source.lower().endswith('.pdf'):
            self._load_pdf()
        elif self.source.lower().startswith('http'):
            self._load_webpage()
        else:
            raise ValueError("Invalid source. Must be a PDF file or a URL.")

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


if __name__ == "__main__":
    # Load text from a PDF file
    pdf_loader = DocumentLoader("https://arxiv.org/pdf/2105.01697.pdf")
    print(pdf_loader())  

    # Load text from a webpage
    web_loader = DocumentLoader("https://arxiv.org/abs/2105.01697")
    print(web_loader())