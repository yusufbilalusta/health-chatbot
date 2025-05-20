# Marmara Akademik Chatbot

Bu proje, Marmara Üniversitesi akademik dokümanları üzerinde çalışan bir RAG (Retrieval Augmented Generation) tabanlı chatbot uygulamasıdır. Bu chatbot, kullanıcıların üniversite ile ilgili sorularını, dokümanlardan bilgi çekerek yanıtlamaktadır.

## Özellikler

- Marmara Üniversitesi dokümanlarını vektör veritabanına dönüştürme
- Google Gemini veya OpenAI modellerini kullanabilme seçeneği
- Kullanıcı dostu Streamlit arayüzü
- Sohbet geçmişini kaydetme ve görüntüleme
- Türkçe sorulara Türkçe yanıtlar

## Kurulum

1. Bu repository'yi klonlayın:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. API anahtarlarınızı ayarlayın:
   - `env_sample` dosyasını `.env` olarak kopyalayın
   - `.env` dosyasını açın ve Google Gemini ve/veya OpenAI API anahtarlarınızı ekleyin

## Kullanım

1. Streamlit uygulamasını başlatın:
```bash
streamlit run app.py
```

2. Tarayıcınızda otomatik olarak açılacak olan Streamlit arayüzünden:
   - Sidebar'dan kullanmak istediğiniz modeli seçin (Gemini veya OpenAI)
   - Marmara Üniversitesi ile ilgili sorularınızı sorun

## Proje Yapısı

- `app.py`: Ana Streamlit uygulaması
- `data/`: PDF dokümanlarının bulunduğu klasör
- `requirements.txt`: Gerekli Python paketleri
- `env_sample`: Örnek .env dosyası

## Nasıl Çalışır?

1. PDF dokümanları okunur ve küçük parçalara ayrılır
2. Her parça, seçilen modele göre vektör temsillere dönüştürülür
3. Kullanıcı bir soru sorduğunda, soru ile en alakalı parçalar vektör veritabanından çekilir
4. LLM modeli, bu belge parçalarını kullanarak soruyu yanıtlar

## Model Seçimi

### Gemini
- Google'ın en son dil modeli
- Türkçe dil desteği güçlü
- Uzun belge anlama yeteneği

### OpenAI
- GPT-3.5-turbo modeli kullanılır
- Güçlü doğal dil anlama yetenekleri
- İyi yapılandırılmış yanıtlar

## Notlar

- İlk çalıştırmada vektör veritabanını oluşturmak biraz zaman alabilir
- En iyi performans için spesifik sorular sorun 