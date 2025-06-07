import os
import glob
import time
import shutil
import uuid
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from google import genai

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
        
        # Veritabanını proje dizini içinde oluştur
        self.base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db")
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Her model için benzersiz bir dizin oluşturuyoruz
        self.persist_directory = os.path.join(self.base_dir, f"chroma_db_health_{model_type.lower()}")
        
        # Set up embeddings based on model type
        if model_type == "Gemini":
            # Check if GEMINI_API_KEY is set
            if not os.getenv("GEMINI_API_KEY"):
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            # Configure Google Generative AI with API key
            api_key = os.getenv("GEMINI_API_KEY")
            # Doğrudan API anahtarını GoogleGenerativeAIEmbeddings'e geçir
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
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
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"Veritabanı açılırken hata oluştu: {str(e)}")
                print("Veritabanı yeniden oluşturuluyor...")
                self._relocate_database()
                self._create_vector_store()
    
    def _relocate_database(self):
        """
        Veritabanını tamamen yeni bir konuma taşı
        """
        # Benzersiz bir ID oluştur
        unique_id = str(uuid.uuid4())[:8]
        timestamp = str(int(time.time()))
        
        # Yeni dizin yolu oluştur (proje dizini içinde)
        new_directory = os.path.join(
            self.base_dir, 
            f"chroma_db_health_{self.model_type.lower()}_{timestamp}_{unique_id}"
        )
        
        print(f"Veritabanı yeni konuma taşınıyor: {new_directory}")
        self.persist_directory = new_directory
        
        # Yeni dizini oluştur
        os.makedirs(self.persist_directory, exist_ok=True)
        
        return self.persist_directory
    
    def _create_vector_store(self):
        """
        Create a vector store from documents in the data/rag_corpus directory
        """
        # Dizini oluştur (izin sorunlarını önlemek için)
        os.makedirs(self.persist_directory, exist_ok=True)
        
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
        
        print(f"Vector store created with {len(splits)} chunks in {self.persist_directory}")
    
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
                loader = TextLoader(txt_file, encoding="utf-8")
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
        try:
            # Veritabanını tamamen yeni bir konuma taşı
            new_directory = self._relocate_database()
            print(f"Yeni veritabanı dizini: {new_directory}")
            
            # Yeni vector store oluştur
            self._create_vector_store()
            
            try:
                chunk_count = len(self.vectorstore.get()["ids"])
                print(f"Veritabanı başarıyla oluşturuldu, {chunk_count} adet chunk içeriyor.")
                return chunk_count
            except Exception as e:
                print(f"Veritabanı oluşturuldu fakat boyut bilgisi alınamadı: {str(e)}")
                return "Veritabanı oluşturuldu fakat boyut bilgisi alınamadı"
        
        except Exception as e:
            print(f"Veritabanı yeniden oluşturulurken hata: {str(e)}")
            raise e 