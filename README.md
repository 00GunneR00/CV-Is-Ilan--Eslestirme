# 🎯 CV Odaklı Akıllı İş Bulma Platformu

**Yapay Zeka Destekli İki Aşamalı İş Öneri Sistemi**

Bu proje, kullanıcıların CV'lerini analiz ederek en uygun iş ilanlarını öneren akıllı bir platform sunmaktadır. BERT tabanlı NLP modelleri ve Faiss vektör arama motoru kullanarak yüksek performanslı eşleştirme yapar.

---

## 🌟 Özellikler

### Temel Özellikler
- ✅ **BERT Tabanlı Metin Analizi**: Transformers (Hugging Face) ile derin anlamsal analiz
- ✅ **Faiss Vektör Arama**: Milyonlarca vektör üzerinde milisaniyeler içinde arama
- ✅ **İki Aşamalı Öneri Sistemi**:
  - **Birincil Öneriler**: Hedef sektörde en uygun pozisyonlar
  - **Çapraz Sektör Önerileri**: Farklı sektörlerde keşfedilecek fırsatlar
- ✅ **5000+ Sentetik İş İlanı**: Gerçekçi test verisi
- ✅ **Streamlit Arayüzü**: Modern ve kullanıcı dostu web arayüzü

### Ayırt Edici Özellik 🚀
**Çapraz Sektör Önerileri**: Kullanıcının beceri setinin beklenmedik sektörlerde de değer bulabileceği pozisyonları keşfeder. Bu özellik, kariyer değişikliği veya yeni fırsatlar arayanlar için benzersiz bir değer sunar.

---

## 📁 Proje Yapısı

```
cv_job_matcher/
│
├── app.py                      # Ana Streamlit uygulaması
├── pipeline.py                 # End-to-end pipeline yönetimi
├── requirements.txt            # Python bağımlılıkları
│
├── data/                       # Veri dosyaları (otomatik oluşturulur)
│   ├── job_postings.csv
│   └── sample_cvs.csv
│
├── models/                     # Modeller ve indeksler (otomatik oluşturulur)
│   ├── embedder.py            # BERT embedding modülü
│   ├── vector_search.py       # Faiss arama motoru
│   ├── recommender.py         # Öneri sistemi
│   ├── job_embeddings.npy
│   ├── cv_embeddings.npy
│   ├── faiss_index.bin
│   └── job_metadata.pkl
│
└── utils/                      # Yardımcı modüller
    ├── data_generator.py      # Sentetik veri üretimi
    └── __init__.py
```

---

## 🚀 Kurulum

### Gereksinimler
- Python 3.8+
- 4GB+ RAM (BERT modeli için)
- 2GB+ Disk alanı

### Adım 1: Depoyu Klonlayın
```bash
git clone <repo-url>
cd cv_job_matcher
```

### Adım 2: Sanal Ortam Oluşturun (Önerilen)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### Adım 3: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

**Not**: İlk kurulumda BERT modeli (~400MB) otomatik olarak indirilecektir.

---

## 💻 Kullanım

### Streamlit Uygulamasını Çalıştırma
```bash
streamlit run app.py
```

Uygulama `http://localhost:8501` adresinde açılacaktır.

### İlk Çalışma
İlk çalıştırmada sistem otomatik olarak:
1. 5000 sentetik iş ilanı üretir
2. 10 örnek CV oluşturur
3. BERT modelini yükler
4. Tüm metinleri vektörleştirir (3-5 dakika sürebilir)
5. Faiss indeksini oluşturur

Sonraki çalıştırmalarda cache'lenmiş veriler kullanılır ve uygulama anında başlar.

---

## 🎯 Kullanım Senaryoları

### 1. Örnek CV ile Test
1. Sidebar'dan "📁 Örnek CV Seç" modunu seçin
2. Listeden bir CV seçin
3. Hedef sektörü belirleyin
4. Öneri sayılarını ayarlayın
5. "Bu CV için Öneri Al" butonuna tıklayın

### 2. Kendi CV'nizi Yükleyin
1. Sidebar'dan "📤 CV Yükle (Metin)" modunu seçin
2. CV metninizi text area'ya yapıştırın
3. Hedef sektörü seçin
4. "🔍 İş Önerilerini Getir" butonuna tıklayın

---

## 🔧 Teknik Detaylar

### NLP Pipeline
```python
# BERT tabanlı multilingual model
Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
Embedding Boyutu: 768 dimension
Normalizasyon: L2 normalized (Kosinüs benzerliği için)
```

### Faiss İndeks
```python
İndeks Tipi: IndexFlatIP (Inner Product)
Arama Karmaşıklığı: O(n) - Exact search
Vektör Sayısı: 5000 iş ilanı
```

### Öneri Algoritması
```python
1. CV Vektörleştirme: cv_vec = BERT(cv_text)
2. Faiss Arama: top_k = Faiss.search(cv_vec, k=100)
3. Sektör Filtreleme:
   - Primary: top_k.filter(sector == primary_sector)[:20]
   - Cross: top_k.filter(sector != primary_sector)[:15]
4. Skor Hesaplama: cosine_similarity(cv_vec, job_vec)
```

---

## 📊 Test ve Benchmark

### Pipeline Testi
```bash
python pipeline.py
```

### Modül Testleri
```bash
# Veri üretimi testi
python utils/data_generator.py

# Embedder testi
python models/embedder.py

# Faiss arama testi
python models/vector_search.py

# Öneri sistemi testi
python models/recommender.py
```

### Performans Metrikleri
- **Vektörleştirme Hızı**: ~50-100 metin/saniye
- **Faiss Arama**: <10ms (5000 vektör üzerinde)
- **End-to-End Öneri**: ~500ms
- **Bellek Kullanımı**: ~2GB (model + embeddings)

---

## 🎨 Streamlit Arayüz Özellikleri

### Ana Özellikler
- 📊 **Canlı Metrikler**: Toplam öneri, ortalama benzerlik, sektör dağılımı
- 📈 **Sektör Analizi**: CV'ye en uygun sektörlerin istatistiksel analizi
- 🎯 **İnteraktif Filtreler**: Sidebar'dan dinamik ayarlar
- 📋 **Detaylı İlan Görünümü**: Expandable cards ile tam bilgi
- 🎨 **Modern Tasarım**: Custom CSS ile profesyonel görünüm

### Sidebar Kontrolleri
- **Mod Seçimi**: Örnek CV / Kendi CV'niz
- **Hedef Sektör**: 10 farklı sektör seçeneği
- **Öneri Sayıları**: Birincil (5-50) ve Çapraz (5-30)
- **Sistem İstatistikleri**: Canlı metrikler

---

## 🔬 Algoritmik Yaklaşım

### 1. Vektörleştirme
```python
# Her metin için
text = "Senior Data Scientist - Python, ML, TensorFlow"
embedding = BERT(text)  # → [768,] float32 vektör
normalized = embedding / ||embedding||  # L2 normalizasyon
```

### 2. Benzerlik Hesaplama
```python
# Kosinüs benzerliği (normalize edilmiş vektörler için)
similarity = dot(cv_vec, job_vec)  # Inner Product
# Değer aralığı: [-1, 1], tipik: [0.3, 0.9]
```

### 3. İki Aşamalı Filtreleme
```python
# Aşama 1: Faiss ile top-k bulma
candidates = faiss_index.search(cv_vec, k=100)

# Aşama 2: Sektör bazlı ayırma
primary_jobs = [j for j in candidates if j.sector == target_sector][:20]
cross_jobs = [j for j in candidates if j.sector != target_sector][:15]
```

---

## 📈 Gelecek Geliştirmeler

- [ ] **Açıklama Mekanizması**: Neden bu ilanlar önerildi?
- [ ] **Kullanıcı Geri Bildirimi**: Beğenilen/beğenilmeyen ilanlar
- [ ] **Dinamik Model Fine-tuning**: Geri bildirimlerle model iyileştirme
- [ ] **Grafik Analizler**: Plotly ile interaktif görselleştirmeler
- [ ] **PDF CV Yükleme**: OCR ile PDF'den metin çıkarma
- [ ] **Gerçek İş İlanı Scraping**: LinkedIn, Indeed entegrasyonu
- [ ] **Multi-language Support**: İngilizce iş ilanları desteği
- [ ] **Email Bildirimleri**: Yeni uygun ilanlar için
- [ ] **API Endpoint'leri**: RESTful API ile entegrasyon

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen:
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir ve MIT lisansı altındadır.

---

## 👨‍💻 Geliştirici

**Yapay Zeka Mühendisliği Öğrencisi-Güner Bektaş**

Portfolio Odak Alanları:
- Natural Language Processing (NLP)
- Öneri Sistemleri
- Makine Öğrenmesi

---

## 📞 İletişim

Sorularınız veya önerileriniz için:
- GitHub Issues
- Email: [bektasguner4@gmail.com]

---

## 🙏 Teşekkürler

- **Hugging Face**: BERT modelleri için
- **Faiss**: Facebook AI Research - Yüksek performanslı vektör arama
- **Streamlit**: Modern web arayüzü framework'ü
- **Anthropic Claude**: Kod geliştirme desteği için

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐**


</div>
