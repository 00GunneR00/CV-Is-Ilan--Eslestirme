#!/bin/bash

echo "=========================================="
echo "CV Odaklı Akıllı İş Bulma Platformu"
echo "=========================================="
echo ""

# Python versiyonunu kontrol et
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 bulunamadı! Lütfen Python 3.8+ yükleyin."
    exit 1
fi

echo "✓ Python bulundu: $(python3 --version)"
echo ""

# Sanal ortam var mı kontrol et
if [ ! -d "venv" ]; then
    echo "📦 Sanal ortam oluşturuluyor..."
    python3 -m venv venv
    echo "✓ Sanal ortam oluşturuldu"
fi

# Sanal ortamı aktifleştir
echo "🔧 Sanal ortam aktifleştiriliyor..."
source venv/bin/activate

# Bağımlılıkları kontrol et ve yükle
if [ ! -f "venv/installed" ]; then
    echo "📥 Bağımlılıklar yükleniyor (bu birkaç dakika sürebilir)..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    touch venv/installed
    echo "✓ Bağımlılıklar yüklendi"
else
    echo "✓ Bağımlılıklar zaten yüklü"
fi

echo ""
echo "🚀 Streamlit uygulaması başlatılıyor..."
echo "📱 Uygulama http://localhost:8501 adresinde açılacak"
echo ""
echo "⚠️  İlk çalıştırmada veri üretimi ve model yükleme 3-5 dakika sürebilir."
echo "⚠️  Sonraki çalıştırmalarda cache kullanılacağı için hızlı başlayacaktır."
echo ""
echo "Çıkmak için: Ctrl+C"
echo ""

# Streamlit'i çalıştır
streamlit run app.py