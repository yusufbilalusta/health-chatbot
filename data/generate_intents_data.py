import os
import csv
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from google import genai
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Define intents and sample prompts
INTENTS = {
    "Greeting": [
        "Merhaba",
        "Selam",
        "İyi günler",
        "Nasılsınız",
        "Merhaba doktor"
    ],
    "Goodbye": [
        "Hoşça kal",
        "Görüşürüz",
        "İyi günler dilerim",
        "Teşekkür ederim, iyi günler",
        "Sağlıcakla kalın"
    ],
    "Refusal": [
        "Bu konuda bilgi vermek istemiyorum",
        "Özel bir durum, paylaşmak istemiyorum",
        "Bunu söylemek istemiyorum",
        "Başka bir soru sorabilir miyim",
        "Bu konuyu geçelim"
    ],
    "Symptoms": [
        "Başım çok ağrıyor ne yapmalıyım",
        "Sürekli öksürüyorum",
        "Ateşim var ve boğazım ağrıyor",
        "İki gündür karnım ağrıyor",
        "Dizimde ağrı ve şişlik var"
    ],
    "Disease_Info": [
        "Diyabet nedir",
        "Hipertansiyon hakkında bilgi alabilir miyim",
        "Migren belirtileri nelerdir",
        "Sedef hastalığı nasıl bir hastalıktır",
        "Alzheimer hastalığı hakkında bilgi verir misiniz"
    ],
    "Treatment_Info": [
        "Grip için hangi ilaçlar kullanılır",
        "Migren tedavisi nasıl yapılır",
        "Astım için en etkili tedavi nedir",
        "Yüksek tansiyon nasıl kontrol edilir",
        "Diyabet tedavisinde yeni yöntemler nelerdir"
    ],
    "Appointment_Info": [
        "Doktor randevusu nasıl alabilirim",
        "En yakın hastane nerede",
        "MHRS üzerinden randevu alma",
        "Göz doktoru için nasıl randevu alabilirim",
        "Acil servise gitmem gerekir mi"
    ]
}

def generate_variations_with_openai(prompt, intent, num_variations=10):
    """
    Generate variations of a prompt using OpenAI
    
    Args:
        prompt (str): The original prompt
        intent (str): The intent of the prompt
        num_variations (int): Number of variations to generate
        
    Returns:
        list: List of prompt variations
    """
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not found. Skipping OpenAI generation.")
        return []
    
    try:
        # Create the OpenAI client
        client = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Create the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""
            Sen bir sağlık chatbotu için veri üreticisin. 
            Aşağıdaki kullanıcı mesajını aynı anlamı koruyarak ama farklı şekillerde yeniden yazmanı istiyorum.
            Bu mesaj "{intent}" niyetine sahip. Benzer niyet ama farklı ifadelerle {num_variations} farklı mesaj üret.
            Sadece yeniden yazılmış mesajları döndür, her biri ayrı bir satırda olsun, başka bir açıklama ekleme.
            """),
            ("human", prompt)
        ])
        
        # Generate variations
        chain = prompt_template | client
        response = chain.invoke({})
        
        # Split the response into lines and clean up
        variations = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
        
        return variations[:num_variations]  # Limit to the requested number of variations
    
    except Exception as e:
        print(f"Error generating variations with OpenAI: {str(e)}")
        return []

def generate_variations_with_gemini(prompt, intent, num_variations=10):
    """
    Generate variations of a prompt using Gemini
    
    Args:
        prompt (str): The original prompt
        intent (str): The intent of the prompt
        num_variations (int): Number of variations to generate
        
    Returns:
        list: List of prompt variations
    """
    # Check if the API key is set
    if not os.getenv("GEMINI_API_KEY"):
        print("Gemini API key not found. Skipping Gemini generation.")
        return []
    
    try:
        # Initialize the Gemini client
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Create the prompt
        system_prompt = f"""
        Sen bir sağlık chatbotu için veri üreticisin. 
        Aşağıdaki kullanıcı mesajını aynı anlamı koruyarak ama farklı şekillerde yeniden yazmanı istiyorum.
        Bu mesaj "{intent}" niyetine sahip. Benzer niyet ama farklı ifadelerle {num_variations} farklı mesaj üret.
        Sadece yeniden yazılmış mesajları döndür, her biri ayrı bir satırda olsun, başka bir açıklama ekleme.
        """
        
        full_prompt = f"{system_prompt}\n\nMesaj: {prompt}"
        
        # Generate variations
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt
        )
        
        # Split the response into lines and clean up
        variations = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        
        return variations[:num_variations]  # Limit to the requested number of variations
    
    except Exception as e:
        print(f"Error generating variations with Gemini: {str(e)}")
        return []

def generate_dataset(output_file, variations_per_prompt=20):
    """
    Generate an intent dataset and save it to a CSV file
    
    Args:
        output_file (str): Output file path
        variations_per_prompt (int): Number of variations to generate per prompt
    """
    dataset = []
    
    # For each intent
    for intent, prompts in INTENTS.items():
        print(f"Generating variations for intent: {intent}")
        
        # For each prompt
        for prompt in prompts:
            # Add the original prompt
            dataset.append((intent, prompt))
            
            # Generate variations with OpenAI
            openai_variations = generate_variations_with_openai(prompt, intent, variations_per_prompt // 2)
            for variation in openai_variations:
                dataset.append((intent, variation))
            
            # Generate variations with Gemini
            gemini_variations = generate_variations_with_gemini(prompt, intent, variations_per_prompt // 2)
            for variation in gemini_variations:
                dataset.append((intent, variation))
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Write to CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Intent', 'Text'])
        writer.writerows(dataset)
    
    print(f"Dataset generated with {len(dataset)} examples and saved to {output_file}")

if __name__ == "__main__":
    output_file = "data/intents_dataset/health_intents.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generate_dataset(output_file) 