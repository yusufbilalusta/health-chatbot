import os
import pandas as pd
import time
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()

# Check if required API keys are set
if not os.getenv("GOOGLE_API_KEY"):
    print("GOOGLE_API_KEY is not set. Please set it in the .env file.")

if not os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY is not set. Please set it in the .env file.")

# Sample evaluation questions
eval_questions = [
    "Marmara Üniversitesi'nde akademik personel görevlendirme kriterleri nelerdir?",
    "Öğretim üyeleri yurt dışına nasıl görevlendirilebilir?",
    "Rektörlük yurt dışı görevlendirmelerde hangi yetkilere sahiptir?",
    "Akademik personel yabancı dil eğitimi için yurt dışına nasıl gönderilebilir?",
    "Marmara Üniversitesi'nde bir araştırma görevlisi hangi şartlar altında görevlendirilebilir?",
    "Kısa süreli görevlendirme nedir ve süresi ne kadardır?",
    "Uzun süreli görevlendirme nedir ve süresi ne kadardır?",
    "Doktora eğitimi için yurt dışına görevlendirme süreci nasıl işler?",
    "Marmara Üniversitesi'nde mali hak ve sosyal güvenlik hakları nasıl düzenlenir?",
    "Yurt dışı görevlendirmelerde üniversitedeki ders yükümlülükleri nasıl düzenlenir?"
]

# Function to load and process the document
def load_and_process_document():
    print("Loading and processing document...")
    # Load the PDF document
    loader = PyPDFLoader("data/marmara_yonerge.pdf")
    data = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    
    print(f"Document processed into {len(docs)} chunks")
    return docs

# Function to create vector store based on model selection
def create_vector_store(docs, model_type):
    print(f"Creating vector store for {model_type}...")
    
    if model_type == "Gemini":
        # Check if GOOGLE_API_KEY is set
        if not os.getenv("GOOGLE_API_KEY"):
            print("GOOGLE_API_KEY not found! Please set it in your .env file.")
            return None
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        persist_directory = "./chroma_db_gemini"
        
    else:  # OpenAI
        # Check if OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not found! Please set it in your .env file.")
            return None
        
        embeddings = OpenAIEmbeddings()
        persist_directory = "./chroma_db_openai"
    
    # Create or load the vector store
    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

# Function to create and run the RAG chain
def run_rag_chain(query, model_type, vectorstore):
    if vectorstore is None:
        return "Vector store could not be created."
    
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
        start_time = time.time()
        response = rag_chain.invoke({"input": query})
        end_time = time.time()
        
        execution_time = end_time - start_time
        return response["answer"], execution_time
    except Exception as e:
        return f"Error: {str(e)}", 0

def evaluate_model(model_type):
    # Load and process document
    docs = load_and_process_document()
    
    # Create vector store
    vectorstore = create_vector_store(docs, model_type)
    
    if vectorstore is None:
        print(f"Could not create vector store for {model_type}")
        return None
    
    results = []
    
    # Evaluate each question
    for idx, question in enumerate(eval_questions):
        print(f"Evaluating question {idx+1}/{len(eval_questions)} on {model_type}...")
        answer, response_time = run_rag_chain(question, model_type, vectorstore)
        
        # Save result
        results.append({
            "Question": question,
            "Answer": answer,
            "Response Time (s)": response_time,
            "Model": model_type
        })
    
    return pd.DataFrame(results)

def main():
    print("Starting evaluation...")
    
    # Evaluate Gemini model
    gemini_results = evaluate_model("Gemini")
    
    # Evaluate OpenAI model
    openai_results = evaluate_model("OpenAI")
    
    # Combine results
    if gemini_results is not None and openai_results is not None:
        all_results = pd.concat([gemini_results, openai_results])
        
        # Save results to CSV
        all_results.to_csv("model_evaluation_results.csv", index=False)
        print("Evaluation completed and results saved to model_evaluation_results.csv")
        
        # Print average response times
        print("\nAverage Response Times:")
        gemini_avg = gemini_results["Response Time (s)"].mean()
        openai_avg = openai_results["Response Time (s)"].mean()
        print(f"Gemini: {gemini_avg:.2f} seconds")
        print(f"OpenAI: {openai_avg:.2f} seconds")
        
        # Print evaluation summary
        print("\nNumber of questions evaluated:", len(eval_questions))
        print("Models compared: Gemini vs OpenAI")
    else:
        print("Evaluation could not be completed. Please check your API keys.")

if __name__ == "__main__":
    main() 