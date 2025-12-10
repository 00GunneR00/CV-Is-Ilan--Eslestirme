# ⚡ Hızlı Başlangıç

## 3 Adımda Çalıştırın!

### 1️⃣ Kurulum
```bash
pip install -r requirements.txt
```

### 2️⃣ Çalıştırma
```bash
streamlit run app.py
```

### 3️⃣ Kullanım
- Tarayıcınızda `http://localhost:8501` açılacak
- İlk çalıştırma 3-5 dakika sürebilir (veri üretimi + model indirme)
- Sonraki çalıştırmalar anında başlar! ⚡

---

## 🎯 İlk Test İçin

1. **Sidebar**'dan "📁 Örnek CV Seç" modunu seçin
2. Bir örnek CV seçin
3. "Bu CV için Öneri Al" butonuna tıklayın
4. Sonuçları inceleyin! 🎉

---

## 📋 Gereksinimler

- Python 3.8+
- 4GB RAM
- 2GB Disk Alanı
- İnternet (ilk kurulum için)

---

## ⚙️ Alternatif Başlatma

### Linux/Mac:
```bash
./run.sh
```

### Windows:
```bash
run.bat
```

### Manuel:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## 🆘 Sorun mu Yaşıyorsunuz?

### Port kullanımda hatası:
```bash
streamlit run app.py --server.port 8502
```

### Model indirme sorunu:
```bash
pip install --upgrade sentence-transformers
```

### Test yapın:
```bash
python test.py
```

---

## 📚 Daha Fazla Bilgi

- Detaylı kullanım: `USAGE_GUIDE.md`
- Proje yapısı: `PROJECT_STRUCTURE.md`
- Ana dokümantasyon: `README.md`

---

**🚀 İyi çalışmalar!**