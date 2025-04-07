
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split text into chunks
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Tuple, Dict, Union, Optional
from PyPDF2 import PdfReader
from langchain_methods import load_docs_file, load_pdf_file, load_txt_file, load_txt, document_chunking
from langchain_core.documents import Document


class DocSummAndQnA:
    def __init__(self):
        self.file_path: str = None
        self.text_input: str = None
        self.document_type: str = None
        self.document: List[Document] = None
        self.vectorise_document: List = None
        self.summary_chunk: List[Document] = None
        self.qna_chunks: List[Document] = None

    def load_document(self, file_path=None, document_type=None, text=None)->Optional[List[Document]]:

        docs = None
        if document_type == "PDF":
            docs = load_pdf_file(file_path=file_path)
            self.document_type = document_type
        elif document_type == "DOCS":
            docs = load_docs_file(file_path=file_path)
            self.document_type = document_type
        elif document_type == "TXT":
            docs = load_txt_file(file_path=file_path)
            self.document_type = document_type
        elif document_type == "Text":
            docs = load_txt(text=text)
            self.document_type = document_type

        self.document = docs

        return docs
    
    # def document_summarize(self,document: List[Document])->str:
        

    # def document_qna(self,document: List[Document])->str:
    #     pass

    def document_initial_setup(self, file_path=None, document_type=None, text=None)-> None:

        #get document

        # Load document
        docs = self.load_document(file_path=file_path, document_type=document_type)

        #create chunks for summarization
        summary_chunk = document_chunking(document=docs, chunk_size=2000, chunk_overlap=150)

        #create chunks for QnA
        summary_chunk = document_chunking(document=docs, chunk_size=800, chunk_overlap=100)

        #Embed chunks and store in memory




if __name__ == "__main__":
    file_path = "sample_document.pdf"
    document_type = "PDF"

    doc_sum_qna = DocSummAndQnA()

    doc = doc_sum_qna.load_document(file_path=file_path, document_type=document_type)
    print(doc)