from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path: str):
    """
    Takes a path to a PDF file.
    Returns a list of Document objects, one per page.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages


def split_documents(pages):
    """
    Takes the list of pages from load_document().
    Returns a list of smaller Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)
    return chunks