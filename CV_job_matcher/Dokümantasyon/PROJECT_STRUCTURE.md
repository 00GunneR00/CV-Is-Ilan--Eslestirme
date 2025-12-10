# 📂 Proje Yapısı Dokümantasyonu

## 🏗️ Genel Mimari

```
cv_job_matcher/
│
├── 📱 FRONTEND (Streamlit)
│   └── app.py                          # Ana kullanıcı arayüzü
│
├── 🔧 CORE PIPELINE
│   └── pipeline.py                     # End-to-end sistem orkestratörü
│
├── 🧠 MODELS (AI/ML Bileşenleri)
│   ├── embedder.py                     # BERT text embedding
│   ├── vector_search.py                # Faiss vektör arama
│   └── recommender.py                  # İki aşamalı öneri sistemi
│
├── 🛠️ UTILS (Yardımcı Modüller)
│   └── data_generator.py               # Sentetik veri üretimi
│
├── 📊 DATA (Otomatik oluşturulur)
│   ├── job_postings.csv                # 5000 iş ilanı
│   └── sample_cvs.csv                  # 10 örnek CV
│
├── 💾 MODELS (Otomatik oluşturulur)
│   ├── job_embeddings.npy              # İş ilanı vektörleri
│   ├── cv_embeddings.npy               # CV vektörleri
│   ├── faiss_index.bin                 # Faiss indeks dosyası
│   └── job_metadata.pkl                # İlan metadata'sı
│
├── 📝 DOCUMENTATION
│   ├── README.md                       # Ana dokümantasyon
│   ├── USAGE_GUIDE.md                  # Detaylı kullanım kılavuzu
│   └── PROJECT_STRUCTURE.md            # Bu dosya
│
├── ⚙️ CONFIGURATION
│   ├── requirements.txt                # Python bağımlılıkları
│   ├── .gitignore                      # Git ignore kuralları
│   ├── run.sh                          # Linux/Mac başlatma scripti
│   └── run.bat                         # Windows başlatma scripti
│
└── 🧪 TESTING
    └── test.py                         # Test suite
```

---

## 📄 Dosya Detayları

### Frontend Katmanı

#### `app.py` - Streamlit Uygulaması
**Satır Sayısı**: ~500  
**Ana Bileşenler**:
```python
# 1. Sayfa konfigürasyonu ve CSS
st.set_page_config(...)
st.markdown("""<style>...</style>""")

# 2. Pipeline yükleme ve cache
@st.cache_resource
def load_pipeline()

# 3. Öneri gösterimi
def display_recommendations(...)

# 4. Ana uygulama mantığı
def main()
```

**Özellikler**:
- Modern, responsive tasarım
- İki mod: Örnek CV / Kendi CV'niz
- Dinamik sektör filtreleme
- Gerçek zamanlı öneri üretimi
- Detaylı ilan görünümleri

---

### Core Pipeline

#### `pipeline.py` - Sistem Orkestratörü
**Satır Sayısı**: ~400  
**Sınıf**: `JobMatcherPipeline`

**Metodlar**:
```python
__init__(data_dir, models_dir)          # Başlatma
setup(force_regenerate)                 # Kurulum
_setup_data()                           # Veri yükleme
_setup_embedder()                       # Model yükleme
_setup_embeddings()                     # Vektör oluşturma
_setup_search_engine()                  # İndeks kurulumu
_setup_recommender()                    # Öneri sistemi
get_recommendations_for_cv()            # CV için öneri
get_recommendations_for_sample_cv()     # Örnek CV için öneri
get_sector_analysis()                   # Sektör analizi
```

**İş Akışı**:
```
1. Başlatma → 2. Veri Yükleme → 3. Model Setup
                    ↓
4. Embedding Üretimi ← 5. Cache Kontrolü
                    ↓
6. Faiss İndeks → 7. Öneri Sistemi → 8. Hazır!
```

---

### Model Katmanı

#### `models/embedder.py` - Text Embedding
**Satır Sayısı**: ~180  
**Sınıflar**:
- `TextEmbedder`: BERT tabanlı embedding
- `EmbeddingCache`: Önbellek yönetimi

**Metodlar**:
```python
encode_texts(texts, batch_size)         # Batch encoding
encode_single(text)                     # Tek metin
prepare_job_embeddings(job_df)          # İş ilanları
prepare_cv_embeddings(cv_df)            # CV'ler
```

**Model**:
- Adı: `paraphrase-multilingual-mpnet-base-v2`
- Boyut: 768
- Diller: 50+ (Türkçe dahil)

---

#### `models/vector_search.py` - Faiss Arama
**Satır Sayısı**: ~250  
**Sınıflar**:
- `FaissVectorSearch`: Low-level Faiss operasyonları
- `VectorSearchEngine`: High-level arama API

**Metodlar**:
```python
build_index(embeddings, index_type)     # İndeks oluştur
search(query_vector, k)                 # Arama
search_batch(query_vectors, k)          # Batch arama
save_index(filepath)                    # Kaydet
load_index(filepath)                    # Yükle
```

**İndeks Tipleri**:
- `flatip`: Exact search, Inner Product (varsayılan)
- `flatl2`: Exact search, L2 distance
- `ivf`: Approximate search (büyük veri setleri)

---

#### `models/recommender.py` - Öneri Sistemi
**Satır Sayısı**: ~300  
**Sınıflar**:
- `TwoStageRecommender`: Ana öneri motoru
- `JobRecommendation`: Öneri dataclass
- `RecommendationFormatter`: Görüntüleme yardımcısı

**Metodlar**:
```python
recommend(cv_embedding, primary_sector, k_primary, k_cross)
get_sector_distribution(cv_embedding, k)
explain_match(cv_skills, job_skills)
```

**Algoritma**:
```python
# Pseudo-code
def recommend(cv_vec, sector):
    # 1. Faiss'ten top-k al
    candidates = faiss.search(cv_vec, k=100)
    
    # 2. Birincil sektör filtresi
    primary = filter(candidates, sector=sector)[:20]
    
    # 3. Çapraz sektör filtresi
    cross = filter(candidates, sector!=sector)[:15]
    
    return {"primary": primary, "cross_sector": cross}
```

---

### Utilities

#### `utils/data_generator.py` - Veri Üretimi
**Satır Sayısı**: ~350  
**Sınıf**: `SyntheticDataGenerator`

**Veri Havuzları**:
- 10 sektör
- 60 farklı pozisyon
- 180+ teknik beceri
- 5 deneyim seviyesi

**Metodlar**:
```python
generate_job_postings(n)                # İş ilanları üret
generate_sample_cvs(n)                  # CV'ler üret
_generate_job_description()             # İlan açıklaması
_generate_cv_text()                     # CV metni
save_data()                             # Diske kaydet
```

---

## 🔄 Veri Akışı

### 1. Başlangıç (İlk Çalıştırma)
```
User → Streamlit App → Pipeline.setup()
                          ↓
                    Data Generator
                          ↓
                    [5000 jobs, 10 CVs]
                          ↓
                    Text Embedder (BERT)
                          ↓
                    [768-dim vectors]
                          ↓
                    Faiss Index Builder
                          ↓
                    [Indexed & Saved]
```

### 2. Öneri Üretimi (Runtime)
```
User CV Input → Text Embedder → CV Vector [768-dim]
                                      ↓
                              Faiss Search
                                      ↓
                              Top-100 Jobs
                                      ↓
                              Recommender
                              ↙         ↘
                    Primary (20)    Cross (15)
                              ↘         ↙
                            Streamlit Display
```

---

## 💾 Veri Formatları

### Job Posting CSV
```csv
job_id,title,sector,description,required_skills,experience_level,location
JOB_00001,Senior Data Scientist,Veri Bilimi,"...",Python|TensorFlow|...,Senior,İstanbul
```

**Sütunlar**:
- `job_id`: Benzersiz tanımlayıcı
- `title`: İş pozisyonu
- `sector`: Sektör adı
- `description`: Detaylı açıklama
- `required_skills`: Virgülle ayrılmış beceriler
- `experience_level`: Junior|Mid|Senior|Lead|Principal
- `location`: Şehir veya Remote/Hybrid

### CV CSV
```csv
cv_id,primary_sector,cv_text,skills,years_of_experience
CV_001,Yazılım Geliştirme,"ÖZET: ...",Python|JavaScript|...,5
```

### Embedding NPY
```python
# Shape: (n_samples, 768)
embeddings = np.load("job_embeddings.npy")
print(embeddings.shape)  # (5000, 768)
print(embeddings.dtype)  # float32
```

### Faiss Index BIN
```python
# Binary format - Faiss spesifik
index = faiss.read_index("faiss_index.bin")
print(index.ntotal)      # 5000
print(index.d)           # 768
```

---

## 🔌 API ve Entegrasyonlar

### Mevcut Bileşenler
```python
# Transformers (Hugging Face)
from sentence_transformers import SentenceTransformer

# Faiss (Facebook AI)
import faiss

# Streamlit
import streamlit as st
```

### Potansiyel Entegrasyonlar
```python
# FastAPI endpoint
from fastapi import FastAPI
app = FastAPI()

@app.post("/api/recommend")
def recommend(cv: str, sector: str):
    return pipeline.get_recommendations_for_cv(cv, sector)

# Slack bot
from slack_bolt import App
app = App(token=SLACK_TOKEN)

@app.command("/job-recommend")
def job_recommend_command(ack, say, command):
    # ...

# Email notifier
from sendgrid import SendGridAPIClient
# ...
```

---

## 📊 Bellek Kullanımı

### Yükleme Aşaması
```
BERT Model:           ~400 MB
Job Embeddings:       ~15 MB (5000 × 768 × 4 bytes)
CV Embeddings:        ~0.03 MB (10 × 768 × 4 bytes)
Faiss Index:          ~15 MB
Python Runtime:       ~100 MB
Streamlit:           ~50 MB
─────────────────────────────
TOPLAM:              ~580 MB
```

### Runtime
```
İşlem              Bellek      Süre
─────────────────  ─────────   ──────
Veri Yükleme       +50 MB      1s
Model Yükleme      +400 MB     5s
Embedding (5000)   +15 MB      60s
İndeks Oluşturma   +15 MB      2s
Öneri Üretimi      +10 MB      0.5s
```

---

## 🎯 Performans Karakteristikleri

### Latency (Gecikme)
| İşlem | Süre | Not |
|-------|------|-----|
| Tek metin embedding | 20ms | GPU: 5ms |
| Batch (32) embedding | 200ms | GPU: 50ms |
| Faiss arama (k=100) | 5ms | 5000 vektör |
| Tam öneri pipeline | 500ms | CV parse + embed + search |

### Throughput (İşlem Hızı)
| Metrik | Değer |
|--------|-------|
| Embedding/sn | 50-100 metin |
| Arama/sn | 200 sorgu |
| Öneri/sn | 2-3 tam işlem |

---

## 🔐 Güvenlik Notları

### Veri Gizliliği
- ✅ Tüm veriler yerel
- ✅ Harici API yok
- ✅ CV'ler kaydedilmiyor
- ⚠️ Embedding'ler disk'te (şifreli değil)

### Güvenlik İyileştirmeleri
```python
# Örnek: CV encryption
from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher = Fernet(key)

encrypted_cv = cipher.encrypt(cv_text.encode())
# Store encrypted_cv instead of plain text
```

---

## 🧪 Test Kapsamı

### Unit Tests
- ✅ Data Generator
- ✅ Text Embedder
- ✅ Faiss Search
- ✅ Recommender
- ✅ Pipeline

### Integration Tests
- ✅ End-to-end pipeline
- ✅ Streamlit UI (manuel)
- ⏳ API endpoints (gelecek)

### Test Çalıştırma
```bash
python test.py
```

---

## 📈 Ölçeklenebilirlik

### Mevcut Limitler
- İş ilanları: 5,000
- Embedding boyutu: 768
- RAM: ~600 MB
- Disk: ~2 GB

### Ölçekleme Stratejileri

#### 1. Daha Fazla İlan (10K - 100K)
```python
# IVF indeks kullan
search_engine.build_index(embeddings, index_type="ivf")
```

#### 2. Büyük Ölçek (>1M)
```python
# Daha karmaşık Faiss indeksi
import faiss
index = faiss.IndexIVFPQ(
    quantizer,
    d=768,
    nlist=1000,
    m=64,
    nbits=8
)
```

#### 3. Distributed Setup
```python
# Redis cache
# Celery task queue
# Multiple Faiss shards
```

---

## 🔧 Bakım ve Güncelleme

### Model Güncelleme
```bash
# Yeni BERT modeli
pip install --upgrade sentence-transformers

# Verileri yeniden işle
python pipeline.py --force-regenerate
```

### Veri Güncelleme
```python
# Yeni ilanlar ekle
new_jobs = pd.read_csv("new_jobs.csv")
all_jobs = pd.concat([existing_jobs, new_jobs])

# Yeniden indeksle
pipeline.setup(force_regenerate=True)
```

---

## 📚 Referanslar

### Kullanılan Teknolojiler
- [Sentence Transformers](https://www.sbert.net/)
- [Faiss Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Streamlit Docs](https://docs.streamlit.io/)

### Akademik Referanslar
- BERT: Devlin et al., 2018
- Sentence-BERT: Reimers & Gurevych, 2019
- Faiss: Johnson et al., 2019

---

**Son Güncelleme**: Aralık 2024  
**Versiyon**: 1.0.0  
**Yazar**: AI Engineering Student