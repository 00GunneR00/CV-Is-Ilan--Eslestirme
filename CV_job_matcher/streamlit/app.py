"""
CV Odaklı Akıllı İş Bulma Platformu
Streamlit Uygulaması
"""
import sys
from pathlib import Path

# ÖNEMLİ: Proje kök dizinini path'e ekle - EN BAŞTA OLMALI
# Bu satır sistem pipeline modülü yerine bizim pipeline.py'ı yüklemesini sağlar
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time


from pipeline import JobMatcherPipeline
from models.recommender import RecommendationFormatter


# Sayfa konfigürasyonu
st.set_page_config(
    page_title="CV Odaklı İş Bulma Platformu",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #000000;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f0f8f0;
        border-left: 5px solid #2ca02c;
    }
    .cross-section-header {
        font-size: 1.8rem;
        color: #000000;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #fff8f0;
        border-left: 5px solid #ff7f0e;
    }
    .info-box {
        padding: 1rem;
        background-color: #e8f4f8;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border: 2px solid #dee2e6;
        border-radius: 8px;
    }
    .highlight-primary {
        background-color: #d4edda !important;
    }
    .highlight-cross {
        background-color: #fff3cd !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Pipeline'ı yükler ve cache'ler"""
    with st.spinner("🔧 Sistem başlatılıyor..."):
        pipeline = JobMatcherPipeline(data_dir="data", models_dir="models")
        
        # İlk kurulum gerekiyorsa yap
        if not Path("models/faiss_index.bin").exists():
            st.info("İlk kurulum yapılıyor... Bu birkaç dakika sürebilir.")
            pipeline.setup(force_regenerate=True)
        else:
            pipeline.setup(force_regenerate=False)
    
    return pipeline


def display_recommendations(recommendations, title, color="primary"):
    """Önerileri güzel bir şekilde gösterir"""
    if not recommendations:
        st.warning("Öneri bulunamadı.")
        return
    
    # DataFrame'e dönüştür
    formatter = RecommendationFormatter()
    df = formatter.to_dataframe(recommendations)
    
    if not df.empty:
        # Metrikleri göster
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(recommendations)}</h3>
                <p>Toplam Öneri</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_score = np.mean([rec.similarity_score for rec in recommendations])
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_score:.3f}</h3>
                <p>Ortalama Benzerlik</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_sectors = df['Sektör'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{unique_sectors}</h3>
                <p>Farklı Sektör</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tablo göster
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
        
        # Detaylı görünüm için expanderlar
        st.markdown("### 📋 Detaylı İlan Bilgileri")
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec.title} - Benzerlik: {rec.similarity_score:.3f}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📄 Açıklama:**")
                    st.write(rec.description)
                    st.markdown(f"**🛠️ Gereken Beceriler:**")
                    st.write(rec.required_skills)
                
                with col2:
                    st.markdown(f"**🏢 Sektör:** {rec.sector}")
                    st.markdown(f"**📊 Deneyim:** {rec.experience_level}")
                    st.markdown(f"**📍 Lokasyon:** {rec.location}")
                    st.markdown(f"**🎯 Benzerlik:** {rec.similarity_score:.3f}")


def main():
    """Ana uygulama fonksiyonu"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 CV Odaklı Akıllı İş Bulma Platformu</h1>', 
                unsafe_allow_html=True)
    
    # Pipeline'ı yükle
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"❌ Sistem yüklenirken hata oluştu: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/resume.png", width=100)
        st.title("⚙️ Ayarlar")
        
        # Mod seçimi
        app_mode = st.radio(
            "Uygulama Modu:",
            ["📁 Örnek CV Seç", "📤 CV Yükle (Metin)"],
            help="Hazır örneklerden seçin veya kendi CV metninizi girin"
        )
        
        st.markdown("---")
        
        # Sektör seçimi
        available_sectors = pipeline.get_available_sectors()
        primary_sector = st.selectbox(
            "🎯 Hedef Sektör:",
            available_sectors,
            help="Öncelikli olarak görmek istediğiniz sektör"
        )
        
        st.markdown("---")
        
        # Öneri ayarları
        st.subheader("🎛️ Öneri Ayarları")
        k_primary = st.slider(
            "Birincil Sektör Önerisi",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Ana sektörden kaç öneri gösterilsin?"
        )
        
        k_cross = st.slider(
            "Çapraz Sektör Önerisi",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            help="Farklı sektörlerden kaç öneri gösterilsin?"
        )
        
        st.markdown("---")
        
        # İstatistikler
        st.subheader("📊 Sistem İstatistikleri")
        total_jobs = len(pipeline.job_df) if pipeline.job_df is not None else 0
        total_sectors = len(available_sectors)
        
        st.metric("Toplam İş İlanı", f"{total_jobs:,}")
        st.metric("Toplam Sektör", total_sectors)
        
        st.markdown("---")
        
        # Bilgi
        with st.expander("ℹ️ Hakkında"):
            st.markdown("""
            **CV Odaklı Akıllı İş Bulma Platformu**
            
            Bu platform, yapay zeka destekli bir öneri sistemidir:
            
            - 🤖 BERT tabanlı NLP modeli
            - 🔍 Faiss vektör arama motoru
            - 🎯 İki aşamalı öneri sistemi
            - 🌐 Çapraz sektör önerileri
            
            **Geliştirici:** AI Engineering Student
            **Teknolojiler:** Python, Transformers, Faiss, Streamlit
            """)
    
    # Ana içerik
    if app_mode == "📁 Örnek CV Seç":
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **📁 Örnek CV Modu**
        
        Sistemde hazır bulunan örnek CV'lerden birini seçerek hızlıca test edebilirsiniz.
        Her CV farklı sektör ve beceri kombinasyonlarını temsil eder.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Örnek CV'leri göster
        sample_cvs = pipeline.get_sample_cvs()
        
        if sample_cvs.empty:
            st.error("Örnek CV bulunamadı!")
            st.stop()
        
        # CV seçimi
        st.subheader("📄 Örnek CV'leri")
        
        # CV bilgilerini göster
        for idx, row in sample_cvs.iterrows():
            with st.expander(f"CV #{idx+1}: {row['cv_id']} - {row['primary_sector']} ({row['years_of_experience']} yıl)"):
                st.markdown(f"**Ana Sektör:** {row['primary_sector']}")
                st.markdown(f"**Deneyim:** {row['years_of_experience']} yıl")
                st.markdown(f"**Beceriler:** {row['skills']}")
                
                if st.button(f"Bu CV için Öneri Al", key=f"btn_{idx}"):
                    st.session_state.selected_cv_index = idx
        
        # Öneri butonuna basıldıysa
        if 'selected_cv_index' in st.session_state:
            cv_index = st.session_state.selected_cv_index
            
            st.markdown("---")
            st.markdown(f"### 🔍 Analiz Edilen CV: {sample_cvs.iloc[cv_index]['cv_id']}")
            
            with st.spinner("🤖 AI ile eşleştirmeler yapılıyor..."):
                # İlerleme çubuğu
                progress_bar = st.progress(0)
                
                progress_bar.progress(30)
                time.sleep(0.3)
                
                # Önerileri al
                recommendations = pipeline.get_recommendations_for_sample_cv(
                    cv_index=cv_index,
                    primary_sector=primary_sector,
                    k_primary=k_primary,
                    k_cross=k_cross
                )
                
                progress_bar.progress(70)
                time.sleep(0.2)
                
                # Sektör analizi
                cv_embedding = pipeline.cv_embeddings[cv_index]
                sector_dist = pipeline.recommender.get_sector_distribution(
                    cv_embedding, k=100
                )
                
                progress_bar.progress(100)
                time.sleep(0.2)
                progress_bar.empty()
            
            # Sonuçları göster
            st.success("✅ Analiz tamamlandı!")
            
            # Sektör dağılımı
            st.markdown("### 📊 Sektör Uyum Analizi")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("En uyumlu olduğunuz sektörler ve istatistikleri:")
                st.dataframe(
                    sector_dist.head(10),
                    use_container_width=True,
                    height=300
                )
            
            with col2:
                st.markdown("**Top 5 Sektör**")
                for idx, row in sector_dist.head(5).iterrows():
                    st.metric(
                        row['sector'],
                        f"{row['count']} ilan",
                        f"Ort: {row['avg_score']:.3f}"
                    )
            
            # Birincil sektör önerileri
            st.markdown(f'<h2 class="section-header">🎯 Birincil Sektör Önerileri: {primary_sector}</h2>', 
                       unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <strong>Birincil Öneriler:</strong> Seçtiğiniz hedef sektördeki size en uygun iş ilanları.
            Yüksek benzerlik skorları, CV'nizin bu pozisyonlar için güçlü bir eşleşme olduğunu gösterir.
            </div>
            """, unsafe_allow_html=True)
            
            display_recommendations(
                recommendations['primary'],
                "Birincil Sektör Önerileri",
                color="primary"
            )
            
            # Çapraz sektör önerileri
            st.markdown(f'<h2 class="cross-section-header">🌐 Çapraz Sektör Önerileri</h2>', 
                       unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box" style="background-color: #fff8f0; border-left-color: #ff7f0e;">
            <strong>🚀 Potansiyel Fırsatlar:</strong> Becerilerinizin başka sektörlerde de değerli olabileceği pozisyonlar.
            Bu öneriler, kariyerinizde yeni yönler keşfetmenize yardımcı olabilir!
            </div>
            """, unsafe_allow_html=True)
            
            display_recommendations(
                recommendations['cross_sector'],
                "Çapraz Sektör Önerileri",
                color="cross"
            )
    
    else:  # CV Yükle modu
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **📤 CV Metin Girişi**
        
        Kendi CV metninizi yazarak veya yapıştırarak kişiselleştirilmiş iş önerileri alabilirsiniz.
        CV'niz deneyimlerinizi, becerilerinizi ve eğitim bilgilerinizi içermelidir.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CV metni girişi
        cv_text = st.text_area(
            "CV Metninizi Buraya Yapıştırın:",
            height=300,
            placeholder="""Örnek CV formatı:

ÖZET:
5 yıllık Python ve Machine Learning deneyimim var. TensorFlow, PyTorch ve Scikit-learn ile projeler geliştirdim.

TEKNİK BECERİLER:
Python, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, SQL, Docker, Git

ÇALIŞMA DENEYİMİ:
- ABC Şirketi - Data Scientist (3 yıl)
- XYZ Teknoloji - ML Engineer (2 yıl)

EĞİTİM:
- Bilgisayar Mühendisliği - XYZ Üniversitesi
- Deep Learning Sertifikası - Coursera
"""
        )
        
        if st.button("🔍 İş Önerilerini Getir", type="primary", use_container_width=True):
            if not cv_text or len(cv_text.strip()) < 50:
                st.error("❌ Lütfen en az 50 karakter uzunluğunda bir CV metni girin!")
            else:
                st.markdown("---")
                
                with st.spinner("🤖 CV analiz ediliyor ve eşleştirmeler yapılıyor..."):
                    # İlerleme
                    progress_bar = st.progress(0)
                    
                    progress_bar.progress(40)
                    time.sleep(0.4)
                    
                    # Önerileri al
                    recommendations = pipeline.get_recommendations_for_cv(
                        cv_text=cv_text,
                        primary_sector=primary_sector,
                        k_primary=k_primary,
                        k_cross=k_cross
                    )
                    
                    progress_bar.progress(80)
                    time.sleep(0.2)
                    
                    # Sektör analizi
                    sector_dist = pipeline.get_sector_analysis(cv_text)
                    
                    progress_bar.progress(100)
                    time.sleep(0.2)
                    progress_bar.empty()
                
                st.success("✅ Analiz tamamlandı!")
                
                # Sektör dağılımı
                st.markdown("### 📊 Sektör Uyum Analizi")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("CV'nize göre en uyumlu sektörler:")
                    st.dataframe(
                        sector_dist.head(10),
                        use_container_width=True,
                        height=300
                    )
                
                with col2:
                    st.markdown("**Top 5 Sektör**")
                    for idx, row in sector_dist.head(5).iterrows():
                        st.metric(
                            row['sector'],
                            f"{row['count']} ilan",
                            f"Ort: {row['avg_score']:.3f}"
                        )
                
                # Birincil sektör önerileri
                st.markdown(f'<h2 class="section-header">🎯 Birincil Sektör Önerileri: {primary_sector}</h2>', 
                           unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                <strong>Birincil Öneriler:</strong> Seçtiğiniz hedef sektördeki size en uygun iş ilanları.
                </div>
                """, unsafe_allow_html=True)
                
                display_recommendations(
                    recommendations['primary'],
                    "Birincil Sektör Önerileri",
                    color="primary"
                )
                
                # Çapraz sektör önerileri
                st.markdown(f'<h2 class="cross-section-header">🌐 Çapraz Sektör Önerileri</h2>', 
                           unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box" style="background-color: #fff8f0; border-left-color: #ff7f0e;">
                <strong>🚀 Potansiyel Fırsatlar:</strong> Becerilerinizin farklı sektörlerde değerlendirilmesi!
                </div>
                """, unsafe_allow_html=True)
                
                display_recommendations(
                    recommendations['cross_sector'],
                    "Çapraz Sektör Önerileri",
                    color="cross"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>CV Odaklı Akıllı İş Bulma Platformu</strong></p>
        <p>🤖 BERT | 🔍 Faiss | 🎯 Two-Stage Recommender | 💻 Streamlit</p>
        <p style='font-size: 0.9rem;'>Yapay Zeka Mühendisliği Portfolio Projesi</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()