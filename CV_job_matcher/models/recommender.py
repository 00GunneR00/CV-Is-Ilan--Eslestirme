"""
İki Aşamalı Öneri Sistemi Modülü
Birincil sektör eşleşmeleri ve çapraz sektör önerileri sunar.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class JobRecommendation:
    """Tek bir iş önerisi"""
    job_id: str
    title: str
    sector: str
    description: str
    required_skills: str
    experience_level: str
    location: str
    similarity_score: float
    is_cross_sector: bool


class TwoStageRecommender:
    """İki aşamalı öneri sistemi"""
    
    def __init__(self, job_df: pd.DataFrame, vector_search_engine):
        """
        Args:
            job_df: İş ilanları DataFrame'i
            vector_search_engine: VectorSearchEngine instance
        """
        self.job_df = job_df
        self.search_engine = vector_search_engine
        
        print(f"✓ Öneri sistemi başlatıldı")
        print(f"   - Toplam iş ilanı: {len(job_df)}")
        print(f"   - Sektör sayısı: {job_df['sector'].nunique()}")
    
    def recommend(self, 
                  cv_embedding: np.ndarray,
                  primary_sector: str,
                  k_total: int = 50,
                  k_primary: int = 20,
                  k_cross: int = 15) -> Dict[str, List[JobRecommendation]]:
        """
        İki aşamalı öneri üretir
        
        Args:
            cv_embedding: CV'nin embedding vektörü
            primary_sector: Kullanıcının tercih ettiği ana sektör
            k_total: Faiss'ten çekilecek toplam sonuç sayısı
            k_primary: Gösterilecek birincil sektör önerisi sayısı
            k_cross: Gösterilecek çapraz sektör önerisi sayısı
            
        Returns:
            {
                "primary": [JobRecommendation, ...],
                "cross_sector": [JobRecommendation, ...]
            }
        """
        print(f"\n🎯 Öneri üretiliyor...")
        print(f"   - Ana sektör: {primary_sector}")
        print(f"   - K_total: {k_total}, K_primary: {k_primary}, K_cross: {k_cross}")
        
        # 1. Faiss ile en yakın k_total ilanı bul
        indices, scores = self.search_engine.search_similar(cv_embedding, k=k_total)
        
        # 2. İndekslere karşılık gelen iş ilanlarını çek
        candidate_jobs = self.job_df.iloc[indices].copy()
        candidate_jobs['similarity_score'] = scores
        
        # 3. Birincil sektör önerileri
        primary_jobs = candidate_jobs[
            candidate_jobs['sector'] == primary_sector
        ].head(k_primary)
        
        # 4. Çapraz sektör önerileri (birincil sektör dışındaki yüksek skorlular)
        cross_sector_jobs = candidate_jobs[
            candidate_jobs['sector'] != primary_sector
        ].head(k_cross)
        
        print(f"   ✓ {len(primary_jobs)} birincil sektör önerisi")
        print(f"   ✓ {len(cross_sector_jobs)} çapraz sektör önerisi")
        
        # 5. JobRecommendation objelerine dönüştür
        primary_recommendations = self._to_recommendations(primary_jobs, is_cross_sector=False)
        cross_recommendations = self._to_recommendations(cross_sector_jobs, is_cross_sector=True)
        
        return {
            "primary": primary_recommendations,
            "cross_sector": cross_recommendations
        }
    
    def _to_recommendations(self, jobs_df: pd.DataFrame, 
                           is_cross_sector: bool) -> List[JobRecommendation]:
        """DataFrame'i JobRecommendation listesine dönüştürür"""
        recommendations = []
        
        for _, row in jobs_df.iterrows():
            rec = JobRecommendation(
                job_id=row['job_id'],
                title=row['title'],
                sector=row['sector'],
                description=row['description'],
                required_skills=row['required_skills'],
                experience_level=row['experience_level'],
                location=row['location'],
                similarity_score=row['similarity_score'],
                is_cross_sector=is_cross_sector
            )
            recommendations.append(rec)
        
        return recommendations
    
    def get_sector_distribution(self, cv_embedding: np.ndarray, 
                               k: int = 100) -> pd.DataFrame:
        """
        En yakın k ilandaki sektör dağılımını analiz eder
        
        Args:
            cv_embedding: CV embedding vektörü
            k: Analiz edilecek ilan sayısı
            
        Returns:
            Sektör dağılımı DataFrame'i
        """
        indices, scores = self.search_engine.search_similar(cv_embedding, k=k)
        candidate_jobs = self.job_df.iloc[indices].copy()
        candidate_jobs['similarity_score'] = scores
        
        sector_stats = candidate_jobs.groupby('sector').agg({
            'job_id': 'count',
            'similarity_score': ['mean', 'max']
        }).reset_index()
        
        sector_stats.columns = ['sector', 'count', 'avg_score', 'max_score']
        sector_stats = sector_stats.sort_values('count', ascending=False)
        
        return sector_stats
    
    def explain_match(self, cv_skills: str, job_skills: str) -> Dict[str, any]:
        """
        CV ve iş ilanı becerilerini karşılaştırarak eşleşmeyi açıklar
        
        Args:
            cv_skills: CV'deki beceriler (virgülle ayrılmış)
            job_skills: İş ilanındaki beceriler (virgülle ayrılmış)
            
        Returns:
            Eşleşme analizi
        """
        cv_skill_set = set([s.strip().lower() for s in cv_skills.split(',')])
        job_skill_set = set([s.strip().lower() for s in job_skills.split(',')])
        
        matching_skills = cv_skill_set & job_skill_set
        missing_skills = job_skill_set - cv_skill_set
        extra_skills = cv_skill_set - job_skill_set
        
        match_percentage = (len(matching_skills) / len(job_skill_set) * 100) if job_skill_set else 0
        
        return {
            "matching_skills": list(matching_skills),
            "missing_skills": list(missing_skills),
            "extra_skills": list(extra_skills),
            "match_percentage": round(match_percentage, 2),
            "total_cv_skills": len(cv_skill_set),
            "total_job_skills": len(job_skill_set)
        }


class RecommendationFormatter:
    """Öneri sonuçlarını formatlama yardımcı sınıfı"""
    
    @staticmethod
    def to_dataframe(recommendations: List[JobRecommendation]) -> pd.DataFrame:
        """Öneri listesini DataFrame'e dönüştürür"""
        if not recommendations:
            return pd.DataFrame()
        
        data = []
        for rec in recommendations:
            data.append({
                "İş ID": rec.job_id,
                "Pozisyon": rec.title,
                "Sektör": rec.sector,
                "Deneyim": rec.experience_level,
                "Lokasyon": rec.location,
                "Benzerlik": f"{rec.similarity_score:.3f}",
                "Açıklama (İlk 100 Karakter)": rec.description[:100] + "...",
                "Gereken Beceriler": rec.required_skills
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def to_html(recommendations: List[JobRecommendation]) -> str:
        """Öneri listesini HTML formatında döndürür"""
        if not recommendations:
            return "<p>Öneri bulunamadı.</p>"
        
        html = "<div class='recommendations'>"
        
        for i, rec in enumerate(recommendations, 1):
            html += f"""
            <div class='job-card'>
                <h3>{i}. {rec.title}</h3>
                <p><strong>Sektör:</strong> {rec.sector}</p>
                <p><strong>Deneyim:</strong> {rec.experience_level}</p>
                <p><strong>Lokasyon:</strong> {rec.location}</p>
                <p><strong>Benzerlik Skoru:</strong> {rec.similarity_score:.3f}</p>
                <p><strong>Açıklama:</strong> {rec.description[:200]}...</p>
                <p><strong>Gereken Beceriler:</strong> {rec.required_skills}</p>
            </div>
            """
        
        html += "</div>"
        return html


if __name__ == "__main__":
    # Test için
    print("=" * 60)
    print("Two-Stage Recommender Test")
    print("=" * 60)
    
    # Mock data
    job_data = {
        'job_id': [f'JOB_{i:05d}' for i in range(1, 101)],
        'title': ['Developer'] * 50 + ['Data Scientist'] * 50,
        'sector': ['Yazılım Geliştirme'] * 50 + ['Veri Bilimi'] * 50,
        'description': ['Test açıklama'] * 100,
        'required_skills': ['Python, JavaScript'] * 100,
        'experience_level': ['Mid-Level'] * 100,
        'location': ['İstanbul'] * 100
    }
    job_df = pd.DataFrame(job_data)
    
    print(f"\n✓ Mock job_df oluşturuldu: {len(job_df)} ilan")
    
    # Mock vector search engine
    class MockSearchEngine:
        def search_similar(self, query, k):
            indices = list(range(min(k, len(job_df))))
            scores = [0.9 - i*0.01 for i in range(len(indices))]
            return indices, scores
    
    mock_engine = MockSearchEngine()
    
    # Recommender oluştur
    recommender = TwoStageRecommender(job_df, mock_engine)
    
    # Mock CV embedding
    mock_cv_embedding = np.random.randn(384)
    
    # Öneri al
    recommendations = recommender.recommend(
        mock_cv_embedding,
        primary_sector="Yazılım Geliştirme",
        k_total=50,
        k_primary=10,
        k_cross=5
    )
    
    print(f"\n✓ Öneriler oluşturuldu:")
    print(f"   - Birincil: {len(recommendations['primary'])}")
    print(f"   - Çapraz: {len(recommendations['cross_sector'])}")
    
    # DataFrame'e dönüştür
    formatter = RecommendationFormatter()
    primary_df = formatter.to_dataframe(recommendations['primary'])
    
    print(f"\n✓ DataFrame oluşturuldu: {primary_df.shape}")
    print(primary_df.head(3))