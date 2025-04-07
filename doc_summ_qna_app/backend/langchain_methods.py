
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Union, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split text into chunks

def load_docs_file(file_path)->List[Document]:
    """
    ## Description
    Loads a DOCX file and extracts its contents into a list of Document objects.

    #Arguments
    file_path : str  
        Path to the DOCX file to be loaded.

    #Returns
    List[Document]  
        A list containing the extracted Document objects from the DOCX file.
    """
    loader = Docx2txtLoader(file_path)
    data = loader.load()
    return data


def load_pdf_file(file_path)->List[Document]:
    """
    ## Description
    Loads a PDF file and extracts its contents into a list of Document objects.

    ## Arguments
    file_path : str  
        Path to the PDF file to be loaded.

    ## Returns
    List[Document]  
        A list containing the extracted Document objects from the PDF file.
    """
    loader = PyPDFLoader(file_path)
    data = loader.load()
    return data


def load_txt_file(file_path)->List[Document]:
    """
    ## Description
    Loads a text file and extracts its contents into a list of Document objects.

    ## Arguments
    file_path : str  
        Path to the text file to be loaded.

    ## Returns
    List[Document]  
        A list containing the extracted Document objects from the text file.
    """
    loader = TextLoader(file_path)
    data = loader.load()
    return data

def load_txt(text)->List[Document]:
    """
    ## Description
    Creates a Document object from a raw text string.

    ## Arguments
    text : str  
        The raw text to be converted into a Document.

    ## Returns
    List[Document]  
        A list containing a single Document object constructed from the text.
    """
    
    data = Document(
        page_content=text,
        metadata={}
    )
    return List[data]

def document_chunking(document: List[Document], chunk_size: int = 800, chunk_overlap : int = 150)->List[Document]:
    """
    ## Description
    Splits a list of Document objects into smaller chunks using recursive character splitting.

    ## Arguments
    - document : List[Document]  
        - A list of Document objects to be chunked.

    - chunk_size : int, optional (default=800)  
        - Maximum size of each chunk in characters.

    - chunk_overlap : int, optional (default=150)  
        - Number of overlapping characters between consecutive chunks.

    ## Returns
    - List[Document]  
        A list of chunked Document objects preserving context using overlaps.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Each chunk will have a max of 500 characters
        chunk_overlap=chunk_overlap  # Overlapping characters between consecutive chunks to maintain context
    )
    chunks = text_splitter.split_documents(document)

    return chunks

