# 📘 Detaylı Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### 1. Projeyi İndirin
```bash
# Proje dosyalarını bilgisayarınıza indirin
cd cv_job_matcher
```

### 2. Ortamı Hazırlayın

#### Linux/Mac:
```bash
chmod +x run.sh
./run.sh
```

#### Windows:
```bash
run.bat
```

#### Manuel Kurulum:
```bash
# 1. Sanal ortam oluştur
python -m venv venv

# 2. Aktifleştir
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Uygulamayı çalıştır
streamlit run app.py
```

---

## 📚 Modül Detayları

### 1️⃣ Data Generator (`utils/data_generator.py`)

**Amaç**: Sentetik iş ilanları ve CV'ler oluşturur.

**Özellikler**:
- 10 farklı sektör
- Sektör başına 6 farklı pozisyon tipi
- Deneyim seviyesi (Junior, Mid, Senior, Lead, Principal)
- Gerçekçi iş tanımları

**Kullanım**:
```python
from utils.data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator()
jobs_df = generator.generate_job_postings(n=5000)
cvs_df = generator.generate_sample_cvs(n=10)
generator.save_data(jobs_df, cvs_df, output_dir="data")
```

**Çıktı**:
- `data/job_postings.csv`: 5000 iş ilanı
- `data/sample_cvs.csv`: 10 örnek CV

---

### 2️⃣ Text Embedder (`models/embedder.py`)

**Amaç**: BERT ile metinleri sayısal vektörlere dönüştürür.

**Model**: `paraphrase-multilingual-mpnet-base-v2`
- Türkçe desteği
- 768 boyutlu embedding
- L2 normalize (kosinüs benzerliği için)

**Kullanım**:
```python
from models.embedder import TextEmbedder

embedder = TextEmbedder()
embeddings = embedder.encode_texts(["Metin 1", "Metin 2"])
print(embeddings.shape)  # (2, 768)
```

**Performans**:
- ~50-100 metin/saniye
- GPU varsa 5-10x daha hızlı

---

### 3️⃣ Vector Search (`models/vector_search.py`)

**Amaç**: Faiss ile hızlı benzerlik araması.

**İndeks Tipi**: IndexFlatIP (Inner Product)
- Exact search (kesin sonuçlar)
- Normalize vektörler için optimal

**Kullanım**:
```python
from models.vector_search import VectorSearchEngine

search_engine = VectorSearchEngine(embedding_dim=768)
search_engine.index_documents(embeddings, index_type="flatip")

# Arama
indices, scores = search_engine.search_similar(query_vec, k=20)
```

**Performans**:
- <10ms arama süresi (5000 vektör)
- Bellek: ~30MB (5000 vektör, 768 dim)

---

### 4️⃣ Recommender (`models/recommender.py`)

**Amaç**: İki aşamalı öneri sistemi.

**Özellikler**:
1. **Birincil Öneriler**: Hedef sektördeki en uygun ilanlar
2. **Çapraz Sektör**: Diğer sektörlerdeki uygun ilanlar

**Kullanım**:
```python
from models.recommender import TwoStageRecommender

recommender = TwoStageRecommender(job_df, search_engine)
recommendations = recommender.recommend(
    cv_embedding,
    primary_sector="Yazılım Geliştirme",
    k_total=100,
    k_primary=20,
    k_cross=15
)

# Sonuç
print(recommendations['primary'])      # Liste[JobRecommendation]
print(recommendations['cross_sector']) # Liste[JobRecommendation]
```

---

### 5️⃣ Pipeline (`pipeline.py`)

**Amaç**: Tüm sistemi yönetir, end-to-end iş akışı.

**Görevler**:
1. Veri yükleme/üretme
2. Model kurulumu
3. Embedding oluşturma
4. İndeks yapılandırma
5. Öneri servisi

**Kullanım**:
```python
from pipeline import JobMatcherPipeline

# Pipeline oluştur
pipeline = JobMatcherPipeline()

# İlk kurulum (bir kez çalışır)
pipeline.setup(force_regenerate=False)

# Öneri al
recommendations = pipeline.get_recommendations_for_cv(
    cv_text="Python ve ML deneyimi olan...",
    primary_sector="Veri Bilimi",
    k_primary=20,
    k_cross=15
)
```

---

## 🎨 Streamlit Arayüzü Kullanımı

### Ana Ekran

#### Sidebar (Sol Panel)
1. **Mod Seçimi**:
   - 📁 Örnek CV Seç: Hazır örneklerle test
   - 📤 CV Yükle: Kendi CV'nizi girin

2. **Hedef Sektör**: Öncelikli sektör seçimi
3. **Öneri Ayarları**: Kaç öneri gösterilsin?
4. **Sistem İstatistikleri**: Canlı metrikler

#### Ana Panel
- Sektör uyum analizi
- Birincil sektör önerileri
- Çapraz sektör önerileri
- Detaylı ilan bilgileri

---

## 🔬 Teknik Optimizasyonlar

### Embedding Cache
```python
# İlk çalıştırma: Embedding'ler oluşturulur ve kaydedilir
pipeline.setup(force_regenerate=True)

# Sonraki çalıştırmalar: Cache'ten yüklenir (10x daha hızlı)
pipeline.setup(force_regenerate=False)
```

### Faiss İndeks Optimizasyonu
```python
# Küçük veri setleri için (< 10K)
index_type = "flatip"  # Exact search

# Büyük veri setleri için (> 100K)
index_type = "ivf"     # Approximate search
```

### Batch Processing
```python
# Tek tek işleme (yavaş)
for text in texts:
    embedding = embedder.encode_single(text)

# Batch işleme (hızlı)
embeddings = embedder.encode_texts(texts, batch_size=32)
```

---

## 📊 Benchmark ve Test

### Test Çalıştırma
```bash
python test.py
```

**Test Kapsamı**:
1. ✅ Veri üretimi
2. ✅ Text embedding
3. ✅ Faiss arama
4. ✅ Öneri sistemi
5. ✅ End-to-end pipeline

### Manuel Test
```python
# Modül modül test
python utils/data_generator.py
python models/embedder.py
python models/vector_search.py
python models/recommender.py
python pipeline.py
```

---

## 🐛 Sorun Giderme

### Problem 1: Model İndirme Hatası
**Hata**: `OSError: Can't load model...`

**Çözüm**:
```python
# Manuel model indirme
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
```

### Problem 2: Bellek Hatası
**Hata**: `RuntimeError: CUDA out of memory`

**Çözüm**:
```python
# CPU'da çalıştır
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Problem 3: Faiss Kurulum Hatası
**Hata**: `ImportError: cannot import name 'faiss'`

**Çözüm**:
```bash
# CPU versiyonu
pip uninstall faiss-gpu
pip install faiss-cpu

# GPU versiyonu (CUDA gerekli)
pip install faiss-gpu
```

### Problem 4: Streamlit Port Hatası
**Hata**: `Address already in use`

**Çözüm**:
```bash
# Farklı port kullan
streamlit run app.py --server.port 8502
```

---

## 💡 İpuçları

### 1. İlk Kurulum Hızlandırma
```python
# Küçük veri seti ile test
generator = SyntheticDataGenerator()
jobs_df = generator.generate_job_postings(n=500)  # 5000 yerine
```

### 2. GPU Kullanımı
```bash
# PyTorch GPU kurulumu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Özel Model Kullanımı
```python
# Farklı bir BERT modeli
embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### 4. Batch Size Ayarlama
```python
# Düşük bellek
embeddings = embedder.encode_texts(texts, batch_size=8)

# Yüksek bellek
embeddings = embedder.encode_texts(texts, batch_size=64)
```

---

## 📈 Performans İyileştirmeleri

### Öneri Kalitesi Artırma
```python
# Daha fazla aday ile başla
recommendations = recommender.recommend(
    cv_embedding,
    primary_sector=sector,
    k_total=200,      # 50 yerine 200
    k_primary=30,     # Daha fazla birincil
    k_cross=20        # Daha fazla çapraz
)
```

### Arama Hızı Artırma
```python
# IVF indeks kullan (approximate ama hızlı)
search_engine.build_index(embeddings, index_type="ivf")
```

---

## 🎓 Gelişmiş Kullanım

### Özel Veri Seti Kullanma
```python
# Kendi iş ilanlarınızı yükleyin
import pandas as pd

custom_jobs = pd.read_csv("my_jobs.csv")
# Gerekli sütunlar: title, description, required_skills, sector

pipeline = JobMatcherPipeline()
pipeline.job_df = custom_jobs
pipeline.setup(force_regenerate=True)
```

### API Endpoint Oluşturma
```python
from fastapi import FastAPI
from pipeline import JobMatcherPipeline

app = FastAPI()
pipeline = JobMatcherPipeline()
pipeline.setup()

@app.post("/recommend")
def get_recommendations(cv_text: str, sector: str):
    return pipeline.get_recommendations_for_cv(cv_text, sector)
```

---

## 📞 Destek

Sorularınız için:
- 📧 Email: your-email@example.com
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

---

## ✅ Kontrol Listesi

İlk kurulum için:
- [ ] Python 3.8+ yüklü mü?
- [ ] pip güncel mi? (`pip install --upgrade pip`)
- [ ] requirements.txt yüklendi mi?
- [ ] İnternet bağlantısı var mı? (model indirme için)
- [ ] 4GB+ RAM var mı?
- [ ] 2GB+ disk alanı var mı?

Çalıştırma öncesi:
- [ ] Sanal ortam aktif mi?
- [ ] Port 8501 boş mu?
- [ ] data/ ve models/ dizinleri oluştu mu?

---

**🎉 Başarılar dileriz!**