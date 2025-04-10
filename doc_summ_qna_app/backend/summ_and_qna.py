
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split text into chunks
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Tuple, Dict, Union, Optional
from PyPDF2 import PdfReader
from langchain_methods import load_docs_file, load_pdf_file, load_txt_file, load_txt, document_chunking, initialize_embeddings, initialize_vector_store
from langchain_methods import embed_query, retrive_similar_chunks, add_document_to_vector_store
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from openai import OpenAI

INSTRUCTION = """ You are an AI assistant designed to answer user queries based only on the provided context. 
        If the context contains relevant information."""



class DocSummAndQnA:

    def __init__(self):
        self.file_path: str = None
        self.text_input: str = None
        self.document_type: str = None
        self.document: List[Document] = None
        self.vectorise_document: List = None
        self.summary_chunk: List[Document] = None
        self.qna_chunks: List[Document] = None
        self.embedding_model: HuggingFaceBgeEmbeddings = None
        self.vector_store: FAISS = None

    def load_document(self, file_path = None, document_type = None, text = None)->Optional[List[Document]]:

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

    def create_prompt(self, instruction: str = "", query:str = "", context:str = "")->str:

        prompt = """
        {}
        
        {}

        {}

        """.format(instruction, query, context)

        return prompt
        

        

    def document_qna(self, query:str)->str:

        similar_chunks = retrive_similar_chunks(self.vector_store, query=query)

        context = ""
        for chunk in similar_chunks:
            context = context + "\n" + chunk.page_content
        
        prompt = self.create_prompt(instruction=INSTRUCTION, query=query, context=context)

        # formatted_prompt = [
        #     {"role": "developer", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": str(prompt)},
        # ]

        system_prompt = "You are a helpful assistant. Please response to the user queries"

        prompt=[
            ("system",system_prompt),
            ("user",prompt)
        ]

        model_name = ""
        api_base = ""
        api_key = ""

        # openAI LLM
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,  # if you prefer to pass api key in directly
            openai_api_base=api_base
        )

        response = llm.invoke(prompt)

        # response = llm.invoke(prompt)

        return response


        

    def document_initial_setup(self, file_path=None, document_type=None, text=None)-> None:

        #get document

        # Load document
        docs = self.load_document(file_path=file_path, document_type=document_type)

        print("\n\n\n - - - document loaded - - - - - -- - \n\n")

        #create chunks for summarization
        self.summary_chunk = document_chunking(document=docs, chunk_size=2000, chunk_overlap=150)

        print("\n\n\n - - - summary chunk - - - - - -- - \n\n")

        #create chunks for QnA
        self.qna_chunks = document_chunking(document=docs, chunk_size=800, chunk_overlap=100)

        print("\n\n\n - - - qna chunk done - - - - - -- - \n\n")

        #Embed chunks and store in memory

        self.embedding_model = initialize_embeddings()

        print("\n\n\n - - - embedding model initialized - - - - - -- - \n\n")

        self.vector_store = initialize_vector_store(embedding_model=self.embedding_model)

        print("\n\n\n - - - vector store model initialized - - - - - -- - \n\n")

        add_document_to_vector_store(documents=self.qna_chunks, vector_store=self.vector_store)




if __name__ == "__main__":
    file_path = "sample_document.pdf"
    document_type = "PDF"

    doc_sum_qna = DocSummAndQnA()
    # print(1)
    print("\n\n\n - - - 1 - - - - - -- - \n\n")

    doc_sum_qna.document_initial_setup(file_path=file_path, document_type=document_type)
    # print(doc)
    print("\n\n\n - - - 2 - - - - - -- - \n\n")
    ans = doc_sum_qna.document_qna(query="what is khan academy making?")
    print(ans)

