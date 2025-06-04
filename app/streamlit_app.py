import streamlit as st
import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv

# Add the project root to the path to import from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the models
from models.intent_classifier import IntentClassifier
from models.gemini_model import GeminiModel
from models.gpt_model import GPTModel
from retriever.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Sağlık Bilgilendirme Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Sağlık Bilgilendirme Chatbot")
st.write("Bu chatbot, sağlık konularında bilgilendirme yapmak için geliştirilmiştir. Sorularınızı Türkçe olarak sorabilirsiniz.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_option" not in st.session_state:
    st.session_state.model_option = "Gemini"

if "intent_classifier" not in st.session_state:
    st.session_state.intent_classifier = None

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "openai_model" not in st.session_state:
    st.session_state.openai_model = None

if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None

if "rag_engine_openai" not in st.session_state:
    st.session_state.rag_engine_openai = None

if "rag_engine_gemini" not in st.session_state:
    st.session_state.rag_engine_gemini = None

# Sidebar for model selection
with st.sidebar:
    st.title("Model Seçimi")
    
    model_option = st.radio(
        "Kullanmak istediğiniz modeli seçin:",
        ["Gemini", "OpenAI", "Intent Classifier"]
    )
    
    st.session_state.model_option = model_option
    
    # Option to retrain intent classifier
    if st.button("Intent Classifier'ı Yeniden Eğit"):
        with st.spinner("Intent Classifier eğitiliyor..."):
            try:
                classifier = IntentClassifier()
                metrics, X_test, y_test, y_pred = classifier.train("data/intents_dataset/health_intents.csv")
                
                st.session_state.intent_classifier = classifier
                
                # Display metrics
                st.success("Intent Classifier başarıyla eğitildi!")
                st.write(f"Precision: {metrics['precision']:.4f}")
                st.write(f"Recall: {metrics['recall']:.4f}")
                st.write(f"F1 Score: {metrics['f1_score']:.4f}")
                
                # Create a confusion matrix visualization
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    import numpy as np
                    
                    # Get unique intents
                    intents = classifier.intents
                    
                    # Create the confusion matrix plot
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        metrics['confusion_matrix'], 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=intents,
                        yticklabels=intents
                    )
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title('Confusion Matrix')
                    
                    # Display the plot
                    st.pyplot(plt)
                    
                    # Create a detailed analysis dataframe
                    analysis_data = []
                    for i, (x, y_true, y_predicted) in enumerate(zip(X_test, y_test, y_pred)):
                        analysis_data.append({
                            'Text': x,
                            'True Intent': y_true,
                            'Predicted Intent': y_predicted,
                            'Correct': y_true == y_predicted
                        })
                    
                    analysis_df = pd.DataFrame(analysis_data)
                    st.write("Örnek tahminler:")
                    st.dataframe(analysis_df.head(10))
                    
                    # Calculate accuracy per intent
                    accuracy_per_intent = {}
                    for intent in intents:
                        intent_samples = analysis_df[analysis_df['True Intent'] == intent]
                        if len(intent_samples) > 0:
                            accuracy = intent_samples['Correct'].mean()
                            accuracy_per_intent[intent] = accuracy
                    
                    # Display accuracy per intent
                    acc_df = pd.DataFrame.from_dict(
                        accuracy_per_intent, 
                        orient='index', 
                        columns=['Accuracy']
                    ).sort_values('Accuracy', ascending=False)
                    
                    st.write("Her intent için doğruluk oranı:")
                    st.dataframe(acc_df)
                    
                except Exception as e:
                    st.error(f"Görselleştirme oluşturulurken hata: {str(e)}")
            
            except Exception as e:
                st.error(f"Eğitim sırasında hata oluştu: {str(e)}")
    
    # Option to rebuild RAG index
    if st.button("RAG Veritabanını Yeniden Oluştur"):
        with st.spinner("RAG veritabanı yeniden oluşturuluyor..."):
            try:
                # Create/rebuild Gemini RAG engine
                rag_engine_gemini = RAGEngine(model_type="Gemini")
                chunks_count_gemini = rag_engine_gemini.rebuild_vector_store()
                st.session_state.rag_engine_gemini = rag_engine_gemini
                
                # Create/rebuild OpenAI RAG engine
                rag_engine_openai = RAGEngine(model_type="OpenAI")
                chunks_count_openai = rag_engine_openai.rebuild_vector_store()
                st.session_state.rag_engine_openai = rag_engine_openai
                
                st.success(f"RAG veritabanları başarıyla yeniden oluşturuldu!\nGemini: {chunks_count_gemini} chunk\nOpenAI: {chunks_count_openai} chunk")
            except Exception as e:
                st.error(f"RAG veritabanı oluşturulurken hata: {str(e)}")

    # Display API key information
    st.subheader("API Anahtarları")
    
    if os.getenv("GEMINI_API_KEY"):
        st.success("✅ GEMINI_API_KEY tanımlı")
    else:
        st.error("❌ GEMINI_API_KEY tanımlı değil")
    
    if os.getenv("OPENAI_API_KEY"):
        st.success("✅ OPENAI_API_KEY tanımlı")
    else:
        st.error("❌ OPENAI_API_KEY tanımlı değil")

# Load the models if not already loaded
if not st.session_state.model_loaded:
    with st.spinner("Modeller yükleniyor..."):
        try:
            # Load intent classifier
            intent_classifier = IntentClassifier()
            if not intent_classifier.load_model():
                st.warning("Intent Classifier modeli bulunamadı. Eğitmeniz gerekebilir.")
            else:
                st.session_state.intent_classifier = intent_classifier
            
            # Load Gemini model
            if os.getenv("GEMINI_API_KEY"):
                gemini_model = GeminiModel()
                st.session_state.gemini_model = gemini_model
                
                # Load Gemini RAG engine
                rag_engine_gemini = RAGEngine(model_type="Gemini")
                st.session_state.rag_engine_gemini = rag_engine_gemini
            
            # Load OpenAI model
            if os.getenv("OPENAI_API_KEY"):
                openai_model = GPTModel()
                st.session_state.openai_model = openai_model
                
                # Load OpenAI RAG engine
                rag_engine_openai = RAGEngine(model_type="OpenAI")
                st.session_state.rag_engine_openai = rag_engine_openai
            
            st.session_state.model_loaded = True
        
        except Exception as e:
            st.error(f"Modeller yüklenirken hata: {str(e)}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "intent" in message and message["role"] == "assistant":
            st.caption(f"Intent: {message['intent']}")

# Chat input
query = st.chat_input("Sağlık ile ilgili bir soru sorun...")

# Process user input
if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Düşünüyorum...")
        
        try:
            start_time = time.time()
            
            # Determine intent
            if st.session_state.intent_classifier:
                intent = st.session_state.intent_classifier.predict(query)
            else:
                # Use model's own intent classification if classifier not available
                if st.session_state.model_option == "Gemini" and st.session_state.gemini_model:
                    intent = st.session_state.gemini_model.classify_intent_with_prompt(query)
                elif st.session_state.model_option == "OpenAI" and st.session_state.openai_model:
                    intent = st.session_state.openai_model.classify_intent_with_prompt(query)
                else:
                    intent = "Unknown"
            
            # Get relevant documents for RAG-based intents
            context = None
            if intent in ["Symptoms", "Disease_Info", "Treatment_Info"]:
                if st.session_state.model_option == "Gemini" and st.session_state.rag_engine_gemini:
                    context = st.session_state.rag_engine_gemini.get_relevant_documents(query)
                elif st.session_state.model_option == "OpenAI" and st.session_state.rag_engine_openai:
                    context = st.session_state.rag_engine_openai.get_relevant_documents(query)
            
            # Generate response based on selected model
            if st.session_state.model_option == "Gemini":
                if st.session_state.gemini_model:
                    response = st.session_state.gemini_model.get_response(query, intent, context)
                else:
                    response = "Gemini modeli yüklenemedi. Lütfen API anahtarınızı kontrol edin."
            
            elif st.session_state.model_option == "OpenAI":
                if st.session_state.openai_model:
                    response = st.session_state.openai_model.get_response(query, intent, context)
                else:
                    response = "OpenAI modeli yüklenemedi. Lütfen API anahtarınızı kontrol edin."
            
            elif st.session_state.model_option == "Intent Classifier":
                response = f"Tespit edilen niyet: {intent}\n\nBu modda sadece niyet sınıflandırması yapılmaktadır."
            
            else:
                response = "Lütfen bir model seçin."
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update placeholder with response
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "intent": intent,
                "response_time": response_time
            })
            
            # Display response metadata
            st.caption(f"Intent: {intent} | Response Time: {response_time:.2f}s")
        
        except Exception as e:
            message_placeholder.markdown(f"Hata oluştu: {str(e)}")

# Display information about the project
with st.expander("Proje Hakkında"):
    st.write("""
    ## Sağlık Bilgilendirme Chatbot
    
    Bu chatbot, kullanıcının sağlıkla ilgili niyetlerini (soru tiplerini) sınıflandırarak, ya hazır cevaplar döndürür ya da bir tıbbi makale veri tabanından RAG ile içerik getirir.
    
    ### Olası Niyet Sınıfları:
    
    - **Greeting**: Selamlama
    - **Goodbye**: Vedalaşma
    - **Refusal**: Yanıtlama istememe
    - **Symptoms**: Semptom sorgulama ("baş ağrım var ne olabilir?")
    - **Disease_Info**: Hastalık hakkında bilgi isteme ("diyabet nedir?")
    - **Treatment_Info**: Tedavi hakkında bilgi isteme ("astım için hangi ilaçlar kullanılır?")
    - **Appointment_Info**: Randevu/klinik işlemleri hakkında soru
    
    ### Kullanılan Teknolojiler:
    
    - LangChain
    - Sentence Transformers
    - ChromaDB
    - Streamlit
    - Google Gemini
    - OpenAI
    
    ### Proje Yapısı:
    
    - **Intent Classifier**: Kullanıcı mesajlarını niyet sınıflarına ayırır
    - **RAG Engine**: Tıbbi makalelerden ilgili içeriği getirir
    - **Model Entegrasyonu**: Gemini ve OpenAI modelleri ile entegrasyon
    """)

# Add performance comparison section
with st.expander("Model Performans Karşılaştırması"):
    st.write("""
    ### Model Performansı Karşılaştırması
    
    Aşağıdaki metriklerle değerlendirme yapılmıştır:
    
    - Precision (Kesinlik)
    - Recall (Duyarlılık)
    - F1 Score
    - Confusion Matrix (Karışıklık Matrisi)
    
    Intent Classifier'ı yeniden eğiterek güncel metrikleri görebilirsiniz.
    """)
    
    # Create a placeholder for performance metrics
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = {
            "Intent Classifier": {"precision": 0, "recall": 0, "f1_score": 0},
            "OpenAI": {"precision": 0, "recall": 0, "f1_score": 0},
            "Gemini": {"precision": 0, "recall": 0, "f1_score": 0}
        }
    
    # Display performance metrics
    metrics_df = pd.DataFrame.from_dict(st.session_state.performance_metrics, orient='index')
    st.dataframe(metrics_df)
    
    st.write("""
    Not: OpenAI ve Gemini modellerinin performans metrikleri, Intent Classifier'ın eğitim verileri üzerindeki performanslarını göstermektedir.
    """)
    
    # Button to run intent classification comparison test
    if st.button("Model Karşılaştırma Testi Yap"):
        with st.spinner("Model karşılaştırma testi yapılıyor..."):
            try:
                # Load test data
                test_data = pd.read_csv("data/intents_dataset/health_intents.csv")
                
                # Select a sample of the data for testing (to save API calls)
                test_sample = test_data.sample(min(30, len(test_data)), random_state=42)
                
                # Initialize results
                results = {
                    "Intent Classifier": {"true": [], "pred": []},
                    "OpenAI": {"true": [], "pred": []},
                    "Gemini": {"true": [], "pred": []}
                }
                
                # Test Intent Classifier
                if st.session_state.intent_classifier:
                    for _, row in test_sample.iterrows():
                        results["Intent Classifier"]["true"].append(row["Intent"])
                        results["Intent Classifier"]["pred"].append(
                            st.session_state.intent_classifier.predict(row["Text"])
                        )
                
                # Test OpenAI
                if st.session_state.openai_model:
                    for _, row in test_sample.iterrows():
                        results["OpenAI"]["true"].append(row["Intent"])
                        results["OpenAI"]["pred"].append(
                            st.session_state.openai_model.classify_intent_with_prompt(row["Text"])
                        )
                
                # Test Gemini
                if st.session_state.gemini_model:
                    for _, row in test_sample.iterrows():
                        results["Gemini"]["true"].append(row["Intent"])
                        results["Gemini"]["pred"].append(
                            st.session_state.gemini_model.classify_intent_with_prompt(row["Text"])
                        )
                
                # Calculate metrics
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                for model in results:
                    if results[model]["true"] and results[model]["pred"]:
                        st.session_state.performance_metrics[model] = {
                            "precision": precision_score(
                                results[model]["true"], 
                                results[model]["pred"], 
                                average='weighted'
                            ),
                            "recall": recall_score(
                                results[model]["true"], 
                                results[model]["pred"], 
                                average='weighted'
                            ),
                            "f1_score": f1_score(
                                results[model]["true"], 
                                results[model]["pred"], 
                                average='weighted'
                            )
                        }
                
                # Update the dataframe
                metrics_df = pd.DataFrame.from_dict(st.session_state.performance_metrics, orient='index')
                st.dataframe(metrics_df)
                
                st.success("Model karşılaştırma testi tamamlandı!")
            
            except Exception as e:
                st.error(f"Model karşılaştırma testi sırasında hata: {str(e)}")
