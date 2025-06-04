import os
import pandas as pd
import time
import json
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import the models
from models.intent_classifier import IntentClassifier
from models.gemini_model import GeminiModel
from models.gpt_model import GPTModel
from retriever.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Define test queries for each intent
TEST_QUERIES = {
    "Greeting": [
        "Merhaba doktor",
        "İyi günler sağlık asistanı",
        "Selam, nasılsın?",
        "Merhaba, sağlık konusunda yardım almak istiyorum"
    ],
    "Goodbye": [
        "Teşekkür ederim, görüşürüz",
        "İyi günler dilerim",
        "Sağlıcakla kalın",
        "Verdiğiniz bilgiler için teşekkürler, hoşça kalın"
    ],
    "Refusal": [
        "Bu konuda bilgi vermek istemiyorum",
        "Özel bir durum, paylaşmak istemiyorum",
        "Başka bir soru sorabilir miyim",
        "Bu soruya cevap vermek istemiyorum"
    ],
    "Symptoms": [
        "Son iki gündür başım çok ağrıyor",
        "Öksürüğüm ve yüksek ateşim var",
        "Eklemlerimde ağrı ve şişlik hissediyorum",
        "Sabahları uyandığımda baş dönmesi yaşıyorum"
    ],
    "Disease_Info": [
        "Diyabet hastalığı nedir?",
        "Hipertansiyon hakkında bilgi alabilir miyim?",
        "Migren nedir ve belirtileri nelerdir?",
        "Alzheimer hastalığı hakkında bilgi verir misiniz?"
    ],
    "Treatment_Info": [
        "Grip için hangi ilaçlar kullanılabilir?",
        "Migren ağrısını geçirmek için ne yapabilirim?",
        "Astım için kullanılan ilaçlar nelerdir?",
        "Yüksek tansiyon tedavisi nasıl yapılır?"
    ],
    "Appointment_Info": [
        "Doktor randevusu nasıl alabilirim?",
        "En yakın hastane nerede?",
        "MHRS üzerinden randevu alma",
        "Göz doktoru için nasıl randevu alabilirim?"
    ]
}

def evaluate_intent_classification():
    """
    Evaluate the intent classification performance of different models
    """
    print("Evaluating intent classification performance...")
    
    # Initialize the models
    try:
        intent_classifier = IntentClassifier()
        if not intent_classifier.load_model():
            print("Training a new intent classifier model...")
            intent_classifier.train("data/intents_dataset/health_intents.csv")
    except Exception as e:
        print(f"Error loading/training intent classifier: {str(e)}")
        return
    
    try:
        openai_model = GPTModel()
    except Exception as e:
        print(f"Error loading OpenAI model: {str(e)}")
        openai_model = None
    
    try:
        gemini_model = GeminiModel()
    except Exception as e:
        print(f"Error loading Gemini model: {str(e)}")
        gemini_model = None
    
    # Create a test dataset
    test_data = []
    for intent, queries in TEST_QUERIES.items():
        for query in queries:
            test_data.append({"Intent": intent, "Text": query})
    
    test_df = pd.DataFrame(test_data)
    
    # Initialize results
    results = {
        "Intent Classifier": {"true": [], "pred": [], "time": []},
        "OpenAI": {"true": [], "pred": [], "time": []},
        "Gemini": {"true": [], "pred": [], "time": []}
    }
    
    # Test intent classifier
    print("Testing Intent Classifier...")
    for _, row in test_df.iterrows():
        true_intent = row["Intent"]
        query = row["Text"]
        
        start_time = time.time()
        pred_intent = intent_classifier.predict(query)
        end_time = time.time()
        
        results["Intent Classifier"]["true"].append(true_intent)
        results["Intent Classifier"]["pred"].append(pred_intent)
        results["Intent Classifier"]["time"].append(end_time - start_time)
    
    # Test OpenAI
    if openai_model:
        print("Testing OpenAI model...")
        for _, row in test_df.iterrows():
            true_intent = row["Intent"]
            query = row["Text"]
            
            start_time = time.time()
            pred_intent = openai_model.classify_intent_with_prompt(query)
            end_time = time.time()
            
            results["OpenAI"]["true"].append(true_intent)
            results["OpenAI"]["pred"].append(pred_intent)
            results["OpenAI"]["time"].append(end_time - start_time)
    
    # Test Gemini
    if gemini_model:
        print("Testing Gemini model...")
        for _, row in test_df.iterrows():
            true_intent = row["Intent"]
            query = row["Text"]
            
            start_time = time.time()
            pred_intent = gemini_model.classify_intent_with_prompt(query)
            end_time = time.time()
            
            results["Gemini"]["true"].append(true_intent)
            results["Gemini"]["pred"].append(pred_intent)
            results["Gemini"]["time"].append(end_time - start_time)
    
    # Calculate metrics
    metrics = {}
    for model in results:
        if results[model]["true"] and results[model]["pred"]:
            # Calculate metrics
            metrics[model] = {
                "precision": precision_score(results[model]["true"], results[model]["pred"], average='weighted'),
                "recall": recall_score(results[model]["true"], results[model]["pred"], average='weighted'),
                "f1_score": f1_score(results[model]["true"], results[model]["pred"], average='weighted'),
                "avg_response_time": sum(results[model]["time"]) / len(results[model]["time"]),
                "confusion_matrix": confusion_matrix(
                    results[model]["true"], 
                    results[model]["pred"],
                    labels=list(TEST_QUERIES.keys())
                ).tolist()
            }
    
    # Create a comparison DataFrame
    comparison_df = pd.DataFrame({
        "Model": list(metrics.keys()),
        "Precision": [metrics[model]["precision"] for model in metrics],
        "Recall": [metrics[model]["recall"] for model in metrics],
        "F1 Score": [metrics[model]["f1_score"] for model in metrics],
        "Avg Response Time (s)": [metrics[model]["avg_response_time"] for model in metrics]
    })
    
    # Display results
    print("\nIntent Classification Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save results to CSV
    comparison_df.to_csv("intent_classification_comparison.csv", index=False)
    print("Results saved to intent_classification_comparison.csv")
    
    # Create and save confusion matrices
    for model in metrics:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            metrics[model]["confusion_matrix"], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(TEST_QUERIES.keys()),
            yticklabels=list(TEST_QUERIES.keys())
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model}')
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{model.replace(' ', '_')}.png")
        print(f"Confusion matrix saved to confusion_matrix_{model.replace(' ', '_')}.png")
    
    # Save detailed results as JSON
    with open("intent_classification_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("Detailed results saved to intent_classification_results.json")
    
    return metrics

def evaluate_rag_performance():
    """
    Evaluate the RAG performance
    """
    print("\nEvaluating RAG performance...")
    
    # Initialize the RAG engines
    try:
        rag_engine_openai = RAGEngine(model_type="OpenAI")
    except Exception as e:
        print(f"Error loading OpenAI RAG engine: {str(e)}")
        rag_engine_openai = None
    
    try:
        rag_engine_gemini = RAGEngine(model_type="Gemini")
    except Exception as e:
        print(f"Error loading Gemini RAG engine: {str(e)}")
        rag_engine_gemini = None
    
    # Initialize the models
    try:
        openai_model = GPTModel()
    except Exception as e:
        print(f"Error loading OpenAI model: {str(e)}")
        openai_model = None
    
    try:
        gemini_model = GeminiModel()
    except Exception as e:
        print(f"Error loading Gemini model: {str(e)}")
        gemini_model = None
    
    # Create test queries for RAG-based intents
    rag_test_queries = {
        "Disease_Info": TEST_QUERIES["Disease_Info"],
        "Treatment_Info": TEST_QUERIES["Treatment_Info"],
        "Symptoms": TEST_QUERIES["Symptoms"]
    }
    
    # Flatten the queries
    test_queries = []
    for intent, queries in rag_test_queries.items():
        for query in queries:
            test_queries.append({"intent": intent, "query": query})
    
    # Initialize results
    results = {
        "OpenAI": {"response_time": [], "response_length": []},
        "Gemini": {"response_time": [], "response_length": []}
    }
    
    # Test OpenAI
    if openai_model and rag_engine_openai:
        print("Testing OpenAI RAG...")
        for item in test_queries:
            intent = item["intent"]
            query = item["query"]
            
            # Get relevant documents
            start_time = time.time()
            context = rag_engine_openai.get_relevant_documents(query)
            
            # Generate response
            response = openai_model.get_response(query, intent, context)
            end_time = time.time()
            
            results["OpenAI"]["response_time"].append(end_time - start_time)
            results["OpenAI"]["response_length"].append(len(response))
            
            print(f"OpenAI - Query: {query}")
            print(f"Response: {response[:100]}...")
            print(f"Response time: {end_time - start_time:.2f}s")
            print("---")
    
    # Test Gemini
    if gemini_model and rag_engine_gemini:
        print("Testing Gemini RAG...")
        for item in test_queries:
            intent = item["intent"]
            query = item["query"]
            
            # Get relevant documents
            start_time = time.time()
            context = rag_engine_gemini.get_relevant_documents(query)
            
            # Generate response
            response = gemini_model.get_response(query, intent, context)
            end_time = time.time()
            
            results["Gemini"]["response_time"].append(end_time - start_time)
            results["Gemini"]["response_length"].append(len(response))
            
            print(f"Gemini - Query: {query}")
            print(f"Response: {response[:100]}...")
            print(f"Response time: {end_time - start_time:.2f}s")
            print("---")
    
    # Calculate metrics
    rag_metrics = {}
    for model in results:
        if results[model]["response_time"]:
            rag_metrics[model] = {
                "avg_response_time": sum(results[model]["response_time"]) / len(results[model]["response_time"]),
                "avg_response_length": sum(results[model]["response_length"]) / len(results[model]["response_length"]),
                "min_response_time": min(results[model]["response_time"]),
                "max_response_time": max(results[model]["response_time"])
            }
    
    # Create a comparison DataFrame
    rag_comparison_df = pd.DataFrame({
        "Model": list(rag_metrics.keys()),
        "Avg Response Time (s)": [rag_metrics[model]["avg_response_time"] for model in rag_metrics],
        "Avg Response Length": [rag_metrics[model]["avg_response_length"] for model in rag_metrics],
        "Min Response Time (s)": [rag_metrics[model]["min_response_time"] for model in rag_metrics],
        "Max Response Time (s)": [rag_metrics[model]["max_response_time"] for model in rag_metrics]
    })
    
    # Display results
    print("\nRAG Performance Comparison:")
    print(rag_comparison_df.to_string(index=False))
    
    # Save results to CSV
    rag_comparison_df.to_csv("rag_performance_comparison.csv", index=False)
    print("Results saved to rag_performance_comparison.csv")
    
    # Create and save response time comparison plot
    plt.figure(figsize=(10, 6))
    for model in rag_metrics:
        plt.bar(model, rag_metrics[model]["avg_response_time"])
    
    plt.xlabel('Model')
    plt.ylabel('Avg Response Time (s)')
    plt.title('RAG Response Time Comparison')
    plt.savefig("rag_response_time_comparison.png")
    print("Response time comparison plot saved to rag_response_time_comparison.png")
    
    return rag_metrics

def main():
    print("Starting health chatbot evaluation...")
    
    # Create output directory for evaluation results
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Evaluate intent classification
    intent_metrics = evaluate_intent_classification()
    
    # Evaluate RAG performance
    rag_metrics = evaluate_rag_performance()
    
    # Save all metrics to a JSON file
    all_metrics = {
        "intent_classification": intent_metrics,
        "rag_performance": rag_metrics
    }
    
    with open("evaluation_results/all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    print("\nEvaluation completed! All metrics saved to evaluation_results/all_metrics.json")

if __name__ == "__main__":
    main() 