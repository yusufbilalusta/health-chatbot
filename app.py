import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Marmara Akademik Chatbot",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Marmara Akademik Chatbot")
st.write("Bu chatbot, Marmara Üniversitesi akademik dokümanları hakkında sorularınızı yanıtlamak için geliştirilmiştir.")

# Sidebar for model selection
st.sidebar.title("Model Seçimi")
model_option = st.sidebar.radio(
    "Kullanmak istediğiniz modeli seçin:",
    ["Gemini", "OpenAI"]
)

# Function to load and process the document
@st.cache_resource
def load_and_process_document():
    # Load the PDF document
    loader = PyPDFLoader("data/marmara_yonerge.pdf")
    data = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    
    return docs

# Function to create vector store based on model selection
@st.cache_resource
def create_vector_store(_docs, model_type):
    if model_type == "Gemini":
        # Check if GOOGLE_API_KEY is set
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY bulunamadı! Lütfen .env dosyasında API anahtarınızı tanımlayın.")
            return None
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        persist_directory = "./chroma_db_gemini"
    else:  # OpenAI
        # Check if OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY bulunamadı! Lütfen .env dosyasında API anahtarınızı tanımlayın.")
            return None
        
        embeddings = OpenAIEmbeddings()
        persist_directory = "./chroma_db_openai"
    
    # Create or load the vector store
    try:
        vectorstore = Chroma.from_documents(
            documents=_docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return vectorstore
    except Exception as e:
        st.error(f"Vektör veri tabanı oluşturulurken hata: {str(e)}")
        return None

# Function to create and run the RAG chain
def run_rag_chain(query, model_type, vectorstore):
    if vectorstore is None:
        return "Vektör veri tabanı oluşturulamadı."
    
    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5}
    )
    
    # Create the language model based on the selected option
    if model_type == "Gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.3,
            max_tokens=800
        )
    else:  # OpenAI
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=800
        )
    
    # Create a prompt template
    system_prompt = (
        "Sen Marmara Üniversitesi hakkında bilgi veren bir asistansın. "
        "Aşağıda verilen döküman parçalarını kullanarak kullanıcının sorusunu yanıtla. "
        "Eğer cevabı bilmiyorsan, bilmediğini söyle ve tahmin etme. "
        "Cevabını Türkçe olarak ver ve mümkün olduğunca kısa ve öz tut. "
        "Ayrıca, cevap verirken akademik ve resmi bir dil kullan. "
        "\n\n"
        "Verilen döküman parçaları: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Create the question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Run the chain
    try:
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# Display proper messages for API keys
if model_option == "Gemini" and not os.getenv("GOOGLE_API_KEY"):
    st.warning("Gemini API anahtarı bulunamadı. Lütfen .env dosyasında GOOGLE_API_KEY tanımlayın.")

if model_option == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
    st.warning("OpenAI API anahtarı bulunamadı. Lütfen .env dosyasında OPENAI_API_KEY tanımlayın.")

# Process the document
docs = load_and_process_document()

# Load chat history from session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
query = st.chat_input("Marmara Üniversitesi hakkında bir soru sorun...")

# When the user submits a query
if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Düşünüyorum...")
        
        # Create vector store based on model selection
        vectorstore = create_vector_store(docs, model_option)
        
        # Run the RAG chain
        if vectorstore:
            response = run_rag_chain(query, model_option, vectorstore)
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            message_placeholder.markdown("API anahtarı eksik olduğu için yanıt üretilemedi.")

# Display information about the project
with st.expander("Proje Hakkında"):
    st.write("""
    Bu chatbot, Marmara Üniversitesi akademik dokümanları üzerinde Retrieval Augmented Generation (RAG) 
    teknolojisi kullanılarak geliştirilmiştir. Gemini ve OpenAI LLM modellerini kullanarak sorularınızı yanıtlar.
    
    **Nasıl Çalışır?**
    1. PDF dokümanları okunur ve işlenir
    2. Metin parçalarına ayrılır ve vektör veritabanına kaydedilir
    3. Soru sorduğunuzda en ilgili metin parçaları bulunur
    4. Seçilen LLM modeli bu parçaları kullanarak sorunuzu yanıtlar
    
    **Kullanılan Teknolojiler:**
    - LangChain
    - ChromaDB
    - Streamlit
    - Google Gemini / OpenAI
    """) 