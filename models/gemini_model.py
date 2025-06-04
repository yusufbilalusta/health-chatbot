import os
from dotenv import load_dotenv
from google import genai
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

class GeminiModel:
    def __init__(self):
        """
        Initialize the Gemini model
        """
        # Check if the API key is set
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        # Initialize the Gemini client
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.0-flash"
    
    def get_response(self, query, intent, context=None):
        """
        Get a response from the Gemini model
        
        Args:
            query (str): User query
            intent (str): Detected intent
            context (list, optional): List of context documents
            
        Returns:
            str: Response from the model
        """
        # Create a prompt based on the intent and context
        if intent in ["Greeting", "Goodbye", "Refusal"]:
            # For simple intents, use a basic prompt
            system_prompt = self._get_system_prompt(intent)
            prompt = f"{system_prompt}\n\nUser: {query}"
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text
            
        elif intent in ["Symptoms", "Disease_Info", "Treatment_Info", "Appointment_Info"]:
            # For RAG-based intents
            system_prompt = self._get_system_prompt(intent, context)
            prompt = f"{system_prompt}\n\nUser: {query}"
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text
            
        else:
            # For unknown intents
            return "Özür dilerim, ne demek istediğinizi anlayamadım. Lütfen sağlık ile ilgili bir soru sorun."
    
    def _get_system_prompt(self, intent, context=None):
        """
        Get a system prompt based on the intent
        
        Args:
            intent (str): Detected intent
            context (list, optional): List of context documents
            
        Returns:
            str: System prompt
        """
        if intent == "Greeting":
            return (
                "Sen bir sağlık bilgilendirme asistanısın. Kullanıcıyı kibarca selamla ve "
                "onlara nasıl yardımcı olabileceğini sor. Sağlık semptomları, hastalıklar, "
                "tedaviler ve randevular hakkında bilgi verebileceğini belirt."
            )
            
        elif intent == "Goodbye":
            return (
                "Sen bir sağlık bilgilendirme asistanısın. Kullanıcıya kibarca veda et ve "
                "sağlıklı günler dile. Gerekirse tekrar gelmeleri için onları teşvik et."
            )
            
        elif intent == "Refusal":
            return (
                "Sen bir sağlık bilgilendirme asistanısın. Kullanıcının sorusuna yanıt vermek "
                "istemediğini kibarca belirt. Yetkin dışındaki konularda veya tıbbi teşhis/tedavi "
                "gerektiren durumlarda doktora başvurmalarını öner."
            )
            
        elif intent == "Symptoms":
            base_prompt = (
                "Sen bir sağlık bilgilendirme asistanısın. Kullanıcının belirttiği semptomlara "
                "göre olası durumları genel hatlarıyla açıkla. ANCAK kesinlikle tanı koyma ve "
                "doktor olmadığını belirt. Kullanıcıya mutlaka bir sağlık kuruluşuna başvurmasını öner. "
            )
            
            if context:
                return base_prompt + "\n\nReferans bilgiler:\n" + "\n".join(
                    [doc.page_content for doc in context]
                )
            return base_prompt
            
        elif intent == "Disease_Info":
            base_prompt = (
                "Sen bir sağlık bilgilendirme asistanısın. Kullanıcının sorduğu hastalık "
                "hakkında genel bilgiler ver. Belirtiler, risk faktörleri ve genel önlemler "
                "hakkında bilgi verebilirsin. Ancak, her durumda kişisel sağlık tavsiyesi vermekten "
                "kaçın ve gerekirse doktora başvurmalarını öner."
            )
            
            if context:
                return base_prompt + "\n\nReferans bilgiler:\n" + "\n".join(
                    [doc.page_content for doc in context]
                )
            return base_prompt
            
        elif intent == "Treatment_Info":
            base_prompt = (
                "Sen bir sağlık bilgilendirme asistanısın. Kullanıcının sorduğu tedavi "
                "hakkında genel bilgiler ver. Tedavi yöntemleri, olası yan etkiler ve genel "
                "beklentiler hakkında bilgi verebilirsin. Ancak, her durumda kişisel sağlık tavsiyesi "
                "vermekten kaçın ve gerekirse doktora başvurmalarını öner."
            )
            
            if context:
                return base_prompt + "\n\nReferans bilgiler:\n" + "\n".join(
                    [doc.page_content for doc in context]
                )
            return base_prompt
            
        elif intent == "Appointment_Info":
            return (
                "Sen bir sağlık bilgilendirme asistanısın. Kullanıcıya randevu alma süreçleri "
                "hakkında genel bilgi ver. e-Nabız, MHRS gibi platformlardan bahsedebilir ve "
                "randevu almanın önemini vurgulayabilirsin."
            )
            
        else:
            return (
                "Sen bir sağlık bilgilendirme asistanısın. Kullanıcının sorusuna "
                "yardımcı olmaya çalış, ancak sağlık konularında genel bilgiler verdiğini "
                "ve bir doktor olmadığını belirt."
            )
    
    def classify_intent_with_prompt(self, query):
        """
        Classify the intent of a query using a prompt-based approach with Gemini
        
        Args:
            query (str): User query
            
        Returns:
            str: Detected intent
        """
        system_prompt = """
        Sen bir sağlık chatbotu için niyet sınıflandırıcısısın. Kullanıcının mesajını aşağıdaki kategorilerden birine sınıflandır:
        
        - Greeting: Selamlama
        - Goodbye: Vedalaşma
        - Refusal: Yanıtlama istememe
        - Symptoms: Semptom sorgulama ("baş ağrım var ne olabilir?")
        - Disease_Info: Hastalık hakkında bilgi isteme ("diyabet nedir?")
        - Treatment_Info: Tedavi hakkında bilgi isteme ("astım için hangi ilaçlar kullanılır?")
        - Appointment_Info: Randevu/klinik işlemleri hakkında soru
        
        SADECE bu kategorilerden BİRİNİ seç ve yalnızca kategori adını döndür (açıklama olmadan).
        """
        
        prompt = f"{system_prompt}\n\nUser: {query}"
        
        # Generate response
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        # Extract the intent (should be just the category name)
        intent = response.text.strip()
        
        # Make sure the intent is one of the valid categories
        valid_intents = ["Greeting", "Goodbye", "Refusal", "Symptoms", 
                         "Disease_Info", "Treatment_Info", "Appointment_Info"]
        
        if intent in valid_intents:
            return intent
        else:
            # Default to most similar intent based on simple matching
            for valid_intent in valid_intents:
                if valid_intent.lower() in intent.lower():
                    return valid_intent
            
            # If no match, return a default
            return "Disease_Info"
