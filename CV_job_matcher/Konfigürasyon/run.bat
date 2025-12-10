@echo off
chcp 65001 >nul
echo ==========================================
echo CV Odaklı Akıllı İş Bulma Platformu
echo ==========================================
echo.

REM Python versiyonunu kontrol et
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python bulunamadı! Lütfen Python 3.8+ yükleyin.
    pause
    exit /b 1
)

echo ✓ Python bulundu
echo.

REM Sanal ortam var mı kontrol et
if not exist "venv\" (
    echo 📦 Sanal ortam oluşturuluyor...
    python -m venv venv
    echo ✓ Sanal ortam oluşturuldu
)

REM Sanal ortamı aktifleştir
echo 🔧 Sanal ortam aktifleştiriliyor...
call venv\Scripts\activate.bat

REM Bağımlılıkları kontrol et ve yükle
if not exist "venv\installed" (
    echo 📥 Bağımlılıklar yükleniyor (bu birkaç dakika sürebilir)...
    python -m pip install -q --upgrade pip
    pip install -q -r requirements.txt
    type nul > venv\installed
    echo ✓ Bağımlılıklar yüklendi
) else (
    echo ✓ Bağımlılıklar zaten yüklü
)

echo.
echo 🚀 Streamlit uygulaması başlatılıyor...
echo 📱 Uygulama http://localhost:8501 adresinde açılacak
echo.
echo ⚠️  İlk çalıştırmada veri üretimi ve model yükleme 3-5 dakika sürebilir.
echo ⚠️  Sonraki çalıştırmalarda cache kullanılacağı için hızlı başlayacaktır.
echo.
echo Çıkmak için: Ctrl+C
echo.

REM Streamlit'i çalıştır
streamlit run app.py