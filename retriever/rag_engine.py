import os
import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self, model_type="Gemini"):
        """
        Initialize the RAG engine
        
        Args:
            model_type (str): The model type to use for embeddings (Gemini or OpenAI)
        """
        self.model_type = model_type
        self.persist_directory = f"./chroma_db_health_{model_type.lower()}"
        
        # Set up embeddings based on model type
        if model_type == "Gemini":
            # Check if GEMINI_API_KEY is set
            if not os.getenv("GEMINI_API_KEY"):
                raise ValueError("GEMINI_API_KEY not found in environment variables")
                
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        else:  # OpenAI
            # Check if OPENAI_API_KEY is set
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment variables")
                
            self.embeddings = OpenAIEmbeddings()
        
        # Create vector store if it doesn't exist
        if not os.path.exists(self.persist_directory):
            self._create_vector_store()
        else:
            # Load existing vector store
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    
    def _create_vector_store(self):
        """
        Create a vector store from documents in the data/rag_corpus directory
        """
        # Load documents
        documents = self._load_documents()
        
        if not documents:
            print("No documents found in data/rag_corpus directory")
            # Create an empty vector store
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"Vector store created with {len(splits)} chunks")
    
    def _load_documents(self):
        """
        Load documents from the data/rag_corpus directory
        
        Returns:
            list: List of Document objects
        """
        documents = []
        
        # Get all files in the corpus directory
        pdf_files = glob.glob("data/rag_corpus/**/*.pdf", recursive=True)
        txt_files = glob.glob("data/rag_corpus/**/*.txt", recursive=True)
        csv_files = glob.glob("data/rag_corpus/**/*.csv", recursive=True)
        
        # Load PDF files
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                documents.extend(loader.load())
                print(f"Loaded PDF: {pdf_file}")
            except Exception as e:
                print(f"Error loading PDF {pdf_file}: {str(e)}")
        
        # Load text files
        for txt_file in txt_files:
            try:
                loader = TextLoader(txt_file)
                documents.extend(loader.load())
                print(f"Loaded text file: {txt_file}")
            except Exception as e:
                print(f"Error loading text file {txt_file}: {str(e)}")
        
        # Load CSV files
        for csv_file in csv_files:
            try:
                loader = CSVLoader(csv_file)
                documents.extend(loader.load())
                print(f"Loaded CSV file: {csv_file}")
            except Exception as e:
                print(f"Error loading CSV file {csv_file}: {str(e)}")
        
        return documents
    
    def get_relevant_documents(self, query, k=5):
        """
        Get relevant documents for a query
        
        Args:
            query (str): The query to search for
            k (int): Number of documents to retrieve
            
        Returns:
            list: List of relevant documents
        """
        # Search for relevant documents
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        documents = retriever.invoke(query)
        return documents
    
    def rebuild_vector_store(self):
        """
        Rebuild the vector store from documents in the data/rag_corpus directory
        """
        # Remove existing vector store
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
        
        # Create new vector store
        self._create_vector_store()
        
        return len(self.vectorstore.get()["ids"]) 