"""
CV Odaklı Akıllı İş Bulma Platformu
Clean Professional Design - Blue, Black & White
"""

import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st
import pandas as pd
import numpy as np
import time

from pipeline import JobMatcherPipeline
from models.recommender import RecommendationFormatter


# Sayfa konfigürasyonu
st.set_page_config(
    page_title="İş Eşleştirme Platformu",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mavi-Siyah-Beyaz Profesyonel Tasarım
st.markdown("""
<style>
    /* Profesyonel Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Ana sayfa - Açık gri arka plan */
    .main {
        background-color: #f0f2f5;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Sidebar - Koyu mavi gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a365d 0%, #2c5282 100%);
        padding-top: 2rem;
    }
    
    /* Sidebar yazıları - BEYAZ */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    /* Ana içerik yazıları - SİYAH */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    .main p, .main span, .main div, .main label {
        color: #000000 !important;
    }
    
    /* Ana başlık */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a365d !important;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        margin-bottom: 0.5rem;
        letter-spacing: -0.8px;
    }
    
    .sub-header {
        text-align: center;
        color: #4a5568 !important;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Bölüm başlıkları */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a365d !important;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #2b6cb0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Bilgi kutuları - Mavi kenarlı beyaz */
    .info-box {
        background-color: #ffffff;
        border-left: 4px solid #2b6cb0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #2d3748 !important;
        line-height: 1.8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .info-box strong {
        color: #1a365d !important;
        font-weight: 600;
    }
    
    /* Metrik kartları - Beyaz kartlar */
    .metric-card {
        background: #ffffff;
        padding: 2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #2b6cb0 0%, #2c5282 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(43, 108, 176, 0.2);
        border-color: #2b6cb0;
    }
    
    .metric-card h2 {
        font-size: 3rem;
        font-weight: 700;
        color: #1a365d !important;
        margin: 0.5rem 0;
        letter-spacing: -1.5px;
    }
    
    .metric-card p {
        font-size: 0.9rem;
        color: #4a5568 !important;
        margin: 0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* CV Kartları - Beyaz */
    .cv-card {
        background: #ffffff;
        padding: 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 2px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .cv-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 5px;
        background: linear-gradient(180deg, #2b6cb0 0%, #2c5282 100%);
        border-radius: 12px 0 0 12px;
    }
    
    .cv-card:hover {
        border-color: #2b6cb0;
        box-shadow: 0 4px 16px rgba(43, 108, 176, 0.15);
        transform: translateY(-2px);
    }
    
    .cv-card h4 {
        color: #1a365d !important;
        margin: 0 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .cv-card p {
        color: #2d3748 !important;
        margin: 0.6rem 0;
        font-size: 0.95rem;
    }
    
    .cv-card strong {
        color: #1a365d !important;
        font-weight: 600;
    }
    
    /* Button - Mavi */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0 0%, #2c5282 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(43, 108, 176, 0.3);
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e4e8c 0%, #234668 100%);
        box-shadow: 0 6px 20px rgba(43, 108, 176, 0.4);
        transform: translateY(-2px);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        font-weight: 600;
        color: #1a365d !important;
        padding: 1rem;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #edf2f7;
        border-color: #2b6cb0;
    }
    
    /* DataFrame */
    .stDataFrame {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        overflow: hidden;
    }
    
    /* Tablo başlıkları */
    thead tr th {
        background: linear-gradient(180deg, #2b6cb0 0%, #2c5282 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    tbody tr:hover {
        background-color: #f7fafc !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2b6cb0 0%, #2c5282 100%);
        border-radius: 10px;
        height: 8px;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        color: #2d3748 !important;
        font-size: 1rem;
        transition: all 0.2s ease;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #2b6cb0;
        box-shadow: 0 0 0 3px rgba(43, 108, 176, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
    }
    
    /* Radio buttons - Sidebar */
    .stRadio > label {
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Slider - Sidebar */
    .stSlider label {
        font-weight: 600;
    }
    
    .stSlider > div > div > div > div {
        background-color: #ffffff;
    }
    
    /* Messages */
    .stSuccess {
        background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%);
        color: #22543d !important;
        border-left: 5px solid #38a169;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-weight: 500;
    }
    
    .stError {
        background: linear-gradient(135deg, #fed7d7 0%, #fc8181 100%);
        color: #742a2a !important;
        border-left: 5px solid #e53e3e;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-weight: 500;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #feebc8 0%, #fbd38d 100%);
        color: #7c2d12 !important;
        border-left: 5px solid #ed8936;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-weight: 500;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #bee3f8 0%, #90cdf4 100%);
        color: #1a365d !important;
        border-left: 5px solid #2b6cb0;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-weight: 500;
    }
    
    /* Built-in metrics - Sidebar */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        background: #ffffff;
        border-top: 3px solid #2b6cb0;
        margin-top: 4rem;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.08);
        border-radius: 12px 12px 0 0;
    }
    
    .footer h3 {
        color: #1a365d !important;
        margin-bottom: 1rem;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .footer p {
        color: #4a5568 !important;
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    
    /* Skill badges */
    code {
        background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
        color: #1a365d !important;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 2px solid #cbd5e0;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #2b6cb0, transparent);
        margin: 3rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #edf2f7;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #2b6cb0 0%, #2c5282 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #1e4e8c 0%, #234668 100%);
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #2d3748 !important;
    }
    
    /* Caption */
    .caption {
        color: #718096 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Pipeline'ı yükler ve cache'ler"""
    with st.spinner("Sistem başlatılıyor..."):
        pipeline = JobMatcherPipeline(data_dir="data", models_dir="models")
        
        if not Path("models/faiss_index.bin").exists():
            st.info("İlk kurulum yapılıyor... Bu birkaç dakika sürebilir.")
            pipeline.setup(force_regenerate=True)
        else:
            pipeline.setup(force_regenerate=False)
    
    return pipeline


def display_metrics(recommendations):
    """Metrikleri kartlarda gösterir"""
    if not recommendations:
        st.warning("Öneri bulunamadı.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">💼</span>
            <h2>{len(recommendations)}</h2>
            <p>Toplam Öneri</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_score = np.mean([rec.similarity_score for rec in recommendations])
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">⭐</span>
            <h2>{avg_score:.2f}</h2>
            <p>Ortalama Uyum</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sectors = set([rec.sector for rec in recommendations])
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">🏢</span>
            <h2>{len(sectors)}</h2>
            <p>Farklı Sektör</p>
        </div>
        """, unsafe_allow_html=True)


def display_recommendations(recommendations):
    """Önerileri gösterir"""
    if not recommendations:
        st.warning("Öneri bulunamadı.")
        return
    
    # Metrikleri göster
    display_metrics(recommendations)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # DataFrame'e dönüştür
    formatter = RecommendationFormatter()
    df = formatter.to_dataframe(recommendations)
    
    if not df.empty:
        # Tablo
        st.dataframe(
            df[['Pozisyon', 'Sektör', 'Benzerlik', 'Deneyim', 'Lokasyon']],
            use_container_width=True,
            height=350
        )
        
        # Detaylı görünüm
        st.markdown("### 📋 Detaylı İlan Bilgileri")
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec.title} — Uyum: {rec.similarity_score:.3f}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**İş Tanımı**")
                    st.write(rec.description)
                    
                    st.markdown("**Gereken Beceriler**")
                    skills_list = rec.required_skills.split(', ')
                    skill_badges = ' '.join([f'`{skill}`' for skill in skills_list[:10]])
                    st.markdown(skill_badges)
                    if len(skills_list) > 10:
                        st.caption(f"... ve {len(skills_list)-10} beceri daha")
                
                with col2:
                    st.markdown("**İlan Detayları**")
                    st.info(f"""
**Sektör**  
{rec.sector}

**Deneyim Seviyesi**  
{rec.experience_level}

**Lokasyon**  
{rec.location}

**Uyum Skoru**  
{rec.similarity_score:.3f}
                    """)


def main():
    """Ana uygulama"""
    
    # Header
    st.markdown('<h1 class="main-header">İş Eşleştirme Platformu</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Yapay Zeka Destekli Kariyer Fırsatları</p>',
                unsafe_allow_html=True)
    
    # Pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Ayarlar")
        st.markdown("---")
        
        app_mode = st.radio(
            "Uygulama Modu",
            ["Örnek CV Seç", "Kendi CV'mi Yükle"]
        )
        
        st.markdown("---")
        
        available_sectors = pipeline.get_available_sectors()
        primary_sector = st.selectbox(
            "Hedef Sektör",
            available_sectors
        )
        
        st.markdown("---")
        st.markdown("### Öneri Parametreleri")
        
        k_primary = st.slider("Birincil Sektör", 5, 50, 20, 5)
        k_cross = st.slider("Çapraz Sektör", 5, 30, 15, 5)
        
        st.markdown("---")
        st.markdown("### 📊 Sistem İstatistikleri")
        
        total_jobs = len(pipeline.job_df) if pipeline.job_df is not None else 0
        total_sectors = len(available_sectors)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("İş İlanı", f"{total_jobs:,}")
        with col2:
            st.metric("Sektör", total_sectors)
    
    # Ana içerik
    if app_mode == "Örnek CV Seç":
        st.markdown("""
        <div class="info-box">
        <strong>Örnek CV Modu</strong><br><br>
        Sistemde hazır bulunan örnek CV'lerden birini seçerek platformu test edebilirsiniz.
        Her CV farklı sektör ve yetkinlik kombinasyonlarını içermektedir.
        </div>
        """, unsafe_allow_html=True)
        
        sample_cvs = pipeline.get_sample_cvs()
        
        if sample_cvs.empty:
            st.error("Örnek CV bulunamadı!")
            st.stop()
        
        st.markdown("### 📄 Örnek CV Seçimi")
        
        cols = st.columns(2)
        for idx, row in sample_cvs.iterrows():
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="cv-card">
                    <h4>{row['cv_id']}</h4>
                    <p><strong>Sektör:</strong> {row['primary_sector']}</p>
                    <p><strong>Deneyim:</strong> {row['years_of_experience']} yıl</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔍 Yetkinlikler"):
                    st.write(row['skills'])
                
                if st.button(f"Bu CV'yi Seç", key=f"btn_{idx}", use_container_width=True):
                    st.session_state.selected_cv_index = idx
                    st.rerun()
        
        if 'selected_cv_index' in st.session_state:
            cv_index = st.session_state.selected_cv_index
            
            st.markdown("---")
            
            with st.spinner("Analiz ediliyor..."):
                progress_bar = st.progress(0)
                
                progress_bar.progress(40)
                time.sleep(0.3)
                
                recommendations = pipeline.get_recommendations_for_sample_cv(
                    cv_index=cv_index,
                    primary_sector=primary_sector,
                    k_primary=k_primary,
                    k_cross=k_cross
                )
                progress_bar.progress(80)
                
                cv_embedding = pipeline.cv_embeddings[cv_index]
                sector_dist = pipeline.recommender.get_sector_distribution(cv_embedding, k=100)
                
                progress_bar.progress(100)
                time.sleep(0.2)
                progress_bar.empty()
            
            st.success("✅ Analiz başarıyla tamamlandı")
            
            # Sektör analizi
            st.markdown("---")
            st.markdown("### 📊 Sektör Uyum Analizi")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.dataframe(
                    sector_dist.head(10),
                    use_container_width=True,
                    height=350
                )
            
            with col2:
                st.markdown("**🏆 En Uyumlu Sektörler**")
                for idx, row in sector_dist.head(5).iterrows():
                    st.metric(
                        row['sector'],
                        f"{row['count']} ilan",
                        f"{row['avg_score']:.2f}"
                    )
            
            # Birincil
            st.markdown("---")
            st.markdown(f'<h2 class="section-header">🎯 Birincil Sektör: {primary_sector}</h2>', 
                       unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            Seçtiğiniz hedef sektördeki en uygun pozisyonlar.
            </div>
            """, unsafe_allow_html=True)
            
            display_recommendations(recommendations['primary'])
            
            # Çapraz
            st.markdown("---")
            st.markdown(f'<h2 class="section-header">🌐 Çapraz Sektör Önerileri</h2>', 
                       unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            Yetkinliklerinizin değer bulabileceği alternatif sektörlerdeki pozisyonlar.
            </div>
            """, unsafe_allow_html=True)
            
            display_recommendations(recommendations['cross_sector'])
    
    else:  # CV Yükle
        st.markdown("""
        <div class="info-box">
        <strong>CV Yükleme</strong><br><br>
        Kendi CV metninizi sisteme girerek kişiselleştirilmiş iş önerileri alabilirsiniz.
        </div>
        """, unsafe_allow_html=True)
        
        cv_text = st.text_area(
            "CV Metni",
            height=300,
            placeholder="CV metninizi buraya yapıştırın..."
        )
        
        if st.button("Analiz Et", type="primary", use_container_width=True):
            if not cv_text or len(cv_text.strip()) < 50:
                st.error("Lütfen en az 50 karakter girin!")
            else:
                with st.spinner("Analiz ediliyor..."):
                    progress_bar = st.progress(0)
                    
                    progress_bar.progress(50)
                    recommendations = pipeline.get_recommendations_for_cv(
                        cv_text=cv_text,
                        primary_sector=primary_sector,
                        k_primary=k_primary,
                        k_cross=k_cross
                    )
                    
                    progress_bar.progress(100)
                    time.sleep(0.2)
                    progress_bar.empty()
                
                st.success("✅ Analiz tamamlandı")
                
                sector_dist = pipeline.get_sector_analysis(cv_text)
                
                # Sektör analizi
                st.markdown("---")
                st.markdown("### 📊 Sektör Uyum Analizi")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.dataframe(sector_dist.head(10), use_container_width=True, height=350)
                
                with col2:
                    st.markdown("**🏆 En Uyumlu Sektörler**")
                    for idx, row in sector_dist.head(5).iterrows():
                        st.metric(row['sector'], f"{row['count']} ilan", f"{row['avg_score']:.2f}")
                
                # Birincil
                st.markdown("---")
                st.markdown(f'<h2 class="section-header">🎯 Birincil Sektör: {primary_sector}</h2>', 
                           unsafe_allow_html=True)
                display_recommendations(recommendations['primary'])
                
                # Çapraz
                st.markdown("---")
                st.markdown(f'<h2 class="section-header">🌐 Çapraz Sektör Önerileri</h2>', 
                           unsafe_allow_html=True)
                display_recommendations(recommendations['cross_sector'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>İş Eşleştirme Platformu</h3>
        <p><strong>BERT · Faiss · Machine Learning · Streamlit</strong></p>
        <p>Yapay Zeka Mühendisliği Projesi - Güner Bektaş</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()