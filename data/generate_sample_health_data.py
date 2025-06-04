import os
import csv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from google import genai
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Define topics for health data
HEALTH_TOPICS = {
    "Diyabet": [
        "Diyabet nedir?",
        "Diyabet belirtileri nelerdir?",
        "Diyabet tedavisi nasıl yapılır?",
        "Diyabet komplikasyonları nelerdir?",
        "Diyabet ve beslenme nasıl olmalıdır?"
    ],
    "Hipertansiyon": [
        "Hipertansiyon nedir?",
        "Hipertansiyon belirtileri nelerdir?",
        "Hipertansiyon tedavisi nasıl yapılır?",
        "Hipertansiyon komplikasyonları nelerdir?",
        "Hipertansiyon için yaşam tarzı değişiklikleri nelerdir?"
    ],
    "Migren": [
        "Migren nedir?",
        "Migren belirtileri nelerdir?",
        "Migren tedavisi nasıl yapılır?",
        "Migren tetikleyicileri nelerdir?",
        "Migren atağını önlemek için neler yapılabilir?"
    ],
    "Astım": [
        "Astım nedir?",
        "Astım belirtileri nelerdir?",
        "Astım tedavisi nasıl yapılır?",
        "Astım tetikleyicileri nelerdir?",
        "Astım atağını önlemek için neler yapılabilir?"
    ]
}

def generate_health_info_with_openai(topic, question):
    """
    Generate health information using OpenAI
    
    Args:
        topic (str): Health topic
        question (str): Question about the topic
        
    Returns:
        str: Generated health information
    """
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not found. Skipping OpenAI generation.")
        return None
    
    try:
        # Create the OpenAI client
        client = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Create the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""
            Sen bir sağlık profesyonelisin. "{topic}" hakkında bir tıbbi makale yazıyorsun.
            Aşağıdaki soruya kapsamlı, doğru ve Türkçe olarak yanıt ver. 
            Bilimsel terminolojiyi kullan ancak genel okuyucunun anlayabileceği şekilde açıkla.
            Yanıtın açık, bilgilendirici ve yapılandırılmış olmalı.
            En güncel tıbbi bilgiler ve kanıta dayalı uygulamaları kullan.
            """),
            ("human", question)
        ])
        
        # Generate information
        chain = prompt_template | client
        response = chain.invoke({})
        
        return response.content
    
    except Exception as e:
        print(f"Error generating health information with OpenAI: {str(e)}")
        return None

def generate_health_info_with_gemini(topic, question):
    """
    Generate health information using Gemini
    
    Args:
        topic (str): Health topic
        question (str): Question about the topic
        
    Returns:
        str: Generated health information
    """
    # Check if the API key is set
    if not os.getenv("GEMINI_API_KEY"):
        print("Gemini API key not found. Skipping Gemini generation.")
        return None
    
    try:
        # Initialize the Gemini client
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Create the prompt
        system_prompt = f"""
        Sen bir sağlık profesyonelisin. "{topic}" hakkında bir tıbbi makale yazıyorsun.
        Aşağıdaki soruya kapsamlı, doğru ve Türkçe olarak yanıt ver. 
        Bilimsel terminolojiyi kullan ancak genel okuyucunun anlayabileceği şekilde açıkla.
        Yanıtın açık, bilgilendirici ve yapılandırılmış olmalı.
        En güncel tıbbi bilgiler ve kanıta dayalı uygulamaları kullan.
        """
        
        prompt = f"{system_prompt}\n\nSoru: {question}"
        
        # Generate information
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return response.text
    
    except Exception as e:
        print(f"Error generating health information with Gemini: {str(e)}")
        return None

def generate_sample_health_data():
    """
    Generate sample health data for the RAG corpus
    """
    # Create the RAG corpus directory if it doesn't exist
    corpus_dir = "data/rag_corpus"
    os.makedirs(corpus_dir, exist_ok=True)
    
    # Generate health information for each topic
    for topic, questions in HEALTH_TOPICS.items():
        print(f"Generating information for topic: {topic}")
        
        # Create a text file for the topic
        file_path = os.path.join(corpus_dir, f"{topic.lower()}.txt")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {topic}\n\n")
            
            # Generate information for each question
            for question in questions:
                print(f"  - Question: {question}")
                
                # Try to generate with OpenAI
                openai_info = generate_health_info_with_openai(topic, question)
                
                # If OpenAI fails, try Gemini
                if openai_info is None:
                    gemini_info = generate_health_info_with_gemini(topic, question)
                    if gemini_info is None:
                        print(f"  - Skipping question: {question} (both APIs failed)")
                        continue
                    info = gemini_info
                else:
                    info = openai_info
                
                # Write the information to the file
                f.write(f"## {question}\n\n")
                f.write(f"{info}\n\n")
        
        print(f"  - Information generated and saved to {file_path}")

if __name__ == "__main__":
    generate_sample_health_data() 