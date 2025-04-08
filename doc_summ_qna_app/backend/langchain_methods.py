
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Union, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split text into chunks
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import requests
# from transformers import AutoTokenizer
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4

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

def initialize_embeddings(model_name: str = "BAAI/bge-large-en-v1.5", 
                          model_kwargs: Dict[str, str] = {"device": "cpu"}, 
                          encode_kwargs: Dict[str, bool] = {"normalize_embeddings": True})->HuggingFaceBgeEmbeddings:
    """
    ## Description
    Initializes a HuggingFaceBgeEmbeddings object with specified model and encoding parameters.

    ## Arguments
    - model_name : str, optional (default="BAAI/bge-large-en-v1.5")  
        - Name of the HuggingFace model to use for generating embeddings.

    - model_kwargs : Dict[str, str], optional (default={"device": "cpu"})  
        - Dictionary of model configuration settings such as device type.

    - encode_kwargs : Dict[str, bool], optional (default={"normalize_embeddings": True})  
        - Dictionary of encoding options such as normalization of embeddings.

    ## Returns
    - HuggingFaceBgeEmbeddings  
        An instance of HuggingFaceBgeEmbeddings initialized with given parameters.
    """
    print("response generated")
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    print("response generated")
    return hf

def embed_query(embedding_model:HuggingFaceBgeEmbeddings, query:str)->List[float]:
    """
    ## Description
    Generates an embedding vector for a given query string using the specified embedding model.

    ## Arguments
    - embedding_model : HuggingFaceBgeEmbeddings  
        - A pre-initialized HuggingFaceBgeEmbeddings object.

    - query : str  
        - The query string to be embedded.

    ## Returns
    - List[float]  
        A list representing the embedding vector of the query.
    """

    query_embedding = embedding_model.embed_query(query)

    return query_embedding

def embed_document(embedding_model:HuggingFaceBgeEmbeddings, texts: List[str])->List[List[float]]:
    """
    ## Description
    Generates embedding vectors for a list of document texts using the specified embedding model.

    ## Arguments
    - embedding_model : HuggingFaceBgeEmbeddings  
        - A pre-initialized HuggingFaceBgeEmbeddings object.

    - texts : List[str]  
        - A list of document texts to be embedded.

    ## Returns
    - List[List[float]]  
        A list of embedding vectors corresponding to the input texts.
    """

    document_embedding = embedding_model.embed_query(texts)

    return document_embedding


def initialize_vector_store(embedding_model:HuggingFaceBgeEmbeddings)->FAISS:
    """
    ## Description
    Initializes a FAISS vector store with the given embedding model for similarity search.

    ## Arguments
    - embedding_model : HuggingFaceBgeEmbeddings  
        - A pre-initialized HuggingFaceBgeEmbeddings object used for embedding documents.

    ## Returns
    - FAISS  
        An instance of FAISS vector store ready for adding and searching documents.
    """

    index = faiss.IndexFlatL2(len(embed_query(embedding_model = embedding_model, query = "Hello World")))

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return vector_store

def add_document_to_vector_store(documents: List[Document], vector_store:FAISS)->None:
    """
    ## Description
    Adds a list of Document objects to the FAISS vector store with randomly generated UUIDs.

    ## Arguments
    - documents : List[Document]  
        - A list of Document objects to be added to the vector store.

    - vector_store : FAISS  
        - An instance of the FAISS vector store where the documents will be added.

    ## Returns
    - None  
        This function performs an in-place operation and does not return any value.
    """
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

def retrive_similar_chunks(vector_store:FAISS, query:str, k_most_similar = 3, filter_query: Dict[str,Any] | None = None)->List:
    """
    ## Description
    Retrieves the most similar document chunks from the vector store for a given query.

    ## Arguments
    - vector_store : FAISS  
        - An instance of the FAISS vector store used for similarity search.

    - query : str  
        - The query string to search for similar documents.

    - k_most_similar : int, optional (default=3)  
        - Number of most similar document chunks to retrieve.

    - filter_query : Optional[Dict[str, Any]], optional (default=None)  
        - A dictionary of filtering conditions to apply during the search.

    ## Returns
    - List  
        A list of the most similar document chunks retrieved from the vector store.
    """
    
    results = vector_store.similarity_search(
        query=query,
        k=k_most_similar,
        filter=filter_query,
    )

    return results
