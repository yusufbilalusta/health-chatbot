# 🩺 Sağlık Bilgilendirme Chatbot

Bu proje, kullanıcının sağlıkla ilgili sorularını anlayarak uygun bilgilendirme yapan bir chatbot uygulamasıdır. Sistem, kullanıcının niyetini (intent) sınıflandırır ve ya hazır cevaplar döndürür ya da bir tıbbi makale veri tabanından içerik getirir.

## 🎯 Proje Özellikleri

- **Intent Classification**: Kullanıcı mesajlarını 7 farklı niyet kategorisine sınıflandırır
- **RAG (Retrieval Augmented Generation)**: Tıbbi makalelerden ilgili içeriği getirir
- **Model Karşılaştırması**: OpenAI ve Gemini modellerinin performans karşılaştırması
- **Streamlit Arayüzü**: Kullanıcı dostu bir arayüz

## 🧠 Intent Sınıfları

- **Greeting**: Selamlama
- **Goodbye**: Vedalaşma
- **Refusal**: Yanıtlama istememe
- **Symptoms**: Semptom sorgulama ("baş ağrım var ne olabilir?")
- **Disease_Info**: Hastalık hakkında bilgi isteme ("diyabet nedir?")
- **Treatment_Info**: Tedavi hakkında bilgi isteme ("astım için hangi ilaçlar kullanılır?")
- **Appointment_Info**: Randevu/klinik işlemleri hakkında soru

## 🏗️ Proje Yapısı

```
chatbot-health/
├── data/
│   ├── intents_dataset/
│   │   └── health_intents.csv
│   ├── rag_corpus/
│   │   ├── diabet.pdf
│   │   └── migren.txt
│   └── generate_intents_data.py
├── models/
│   ├── gpt_model.py
│   ├── gemini_model.py
│   └── intent_classifier.py
├── retriever/
│   └── rag_engine.py
├── app/
│   └── streamlit_app.py
├── evaluation_results/
│   └── all_metrics.json
├── evaluate_health_chatbot.py
├── README.md
└── requirements.txt
```

## 🚀 Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

2. API anahtarlarınızı `.env` dosyasında tanımlayın:

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

3. Intent sınıflandırıcısı için veri seti oluşturun:

```bash
python data/generate_intents_data.py
```

4. RAG için tıbbi makaleleri `data/rag_corpus/` dizinine ekleyin.

## 📊 Eğitim ve Değerlendirme

Intent sınıflandırıcıyı eğitmek ve değerlendirmek için:

```bash
python evaluate_health_chatbot.py
```

Bu komut, intent sınıflandırma ve RAG performansını değerlendirir ve sonuçları `evaluation_results/` dizinine kaydeder.

## 🖥️ Uygulamayı Çalıştırma

```bash
streamlit run app/streamlit_app.py
```

Uygulama http://localhost:8501 adresinde çalışacaktır.

## 📝 Model Performans Karşılaştırması

Modeller aşağıdaki metriklerle değerlendirilmiştir:

- Precision
- Recall
- F1 Score
- Confusion Matrix

Intent sınıflandırma performansı için model karşılaştırma sonuçları `intent_classification_comparison.csv` dosyasında, RAG performans sonuçları ise `rag_performance_comparison.csv` dosyasında bulunabilir.

## 🛠️ Kullanılan Teknolojiler

- LangChain
- Sentence Transformers
- ChromaDB
- Streamlit
- Google Gemini API
- OpenAI API
- Scikit-learn
- Pandas
- Matplotlib & Seaborn

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakınız. 