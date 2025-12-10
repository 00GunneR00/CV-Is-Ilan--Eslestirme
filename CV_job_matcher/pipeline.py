"""
Ana Pipeline Modülü
Tüm bileşenleri koordine eder ve end-to-end iş akışını yönetir.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Optional, Dict, List

from utils.data_generator import SyntheticDataGenerator
from models.embedder import TextEmbedder, EmbeddingCache
from models.vector_search import VectorSearchEngine
from models.recommender import TwoStageRecommender, JobRecommendation


class JobMatcherPipeline:
    """Ana pipeline sınıfı - tüm sistemi yönetir"""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Args:
            data_dir: Veri dosyalarının saklanacağı dizin
            models_dir: Model ve indeks dosyalarının saklanacağı dizin
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        # Dizinleri oluştur
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Dosya yolları
        self.job_csv_path = self.data_dir / "job_postings.csv"
        self.cv_csv_path = self.data_dir / "sample_cvs.csv"
        self.job_embeddings_path = self.models_dir / "job_embeddings.npy"
        self.cv_embeddings_path = self.models_dir / "cv_embeddings.npy"
        self.faiss_index_path = self.models_dir / "faiss_index.bin"
        self.metadata_path = self.models_dir / "job_metadata.pkl"
        
        # Bileşenler
        self.job_df: Optional[pd.DataFrame] = None
        self.cv_df: Optional[pd.DataFrame] = None
        self.embedder: Optional[TextEmbedder] = None
        self.search_engine: Optional[VectorSearchEngine] = None
        self.recommender: Optional[TwoStageRecommender] = None
        
        print("✓ Pipeline başlatıldı")
        print(f"   - Veri dizini: {self.data_dir}")
        print(f"   - Model dizini: {self.models_dir}")
    
    def setup(self, force_regenerate: bool = False):
        """
        Pipeline'ı kurulum yapar (veri üretme, embedding, indeksleme)
        
        Args:
            force_regenerate: True ise mevcut verileri siler ve yeniden üretir
        """
        print("\n" + "="*60)
        print("PIPELINE KURULUM")
        print("="*60)
        
        # 1. Veri Üretimi
        self._setup_data(force_regenerate)
        
        # 2. Embedder Yükleme
        self._setup_embedder()
        
        # 3. Embedding Üretimi
        self._setup_embeddings(force_regenerate)
        
        # 4. Faiss İndeks Oluşturma
        self._setup_search_engine(force_regenerate)
        
        # 5. Recommender Kurulumu
        self._setup_recommender()
        
        print("\n" + "="*60)
        print("✓ KURULUM TAMAMLANDI")
        print("="*60)
    
    def _setup_data(self, force_regenerate: bool):
        """Veri setlerini yükler veya oluşturur"""
        print("\n[1/5] Veri Yükleme/Üretme")
        print("-" * 60)
        
        if force_regenerate or not self.job_csv_path.exists():
            print("📊 Sentetik veri üretiliyor...")
            generator = SyntheticDataGenerator()
            self.job_df = generator.generate_job_postings(n=5000)
            self.cv_df = generator.generate_sample_cvs(n=10)
            generator.save_data(self.job_df, self.cv_df, self.data_dir)
        else:
            print("📂 Mevcut veri yükleniyor...")
            self.job_df = pd.read_csv(self.job_csv_path)
            self.cv_df = pd.read_csv(self.cv_csv_path)
            print(f"✓ {len(self.job_df)} iş ilanı yüklendi")
            print(f"✓ {len(self.cv_df)} CV yüklendi")
    
    def _setup_embedder(self):
        """Text embedder'ı yükler"""
        print("\n[2/5] Embedder Yükleme")
        print("-" * 60)
        
        self.embedder = TextEmbedder()
    
    def _setup_embeddings(self, force_regenerate: bool):
        """Embedding'leri oluşturur veya yükler"""
        print("\n[3/5] Embedding Üretimi")
        print("-" * 60)
        
        # İş ilanı embeddings
        if force_regenerate or not self.job_embeddings_path.exists():
            print("🔄 İş ilanı embedding'leri oluşturuluyor...")
            job_embeddings = self.embedder.prepare_job_embeddings(
                self.job_df,
                text_columns=['title', 'description', 'required_skills']
            )
            EmbeddingCache.save_embeddings(job_embeddings, str(self.job_embeddings_path))
        else:
            print("📂 Mevcut iş ilanı embedding'leri yükleniyor...")
            job_embeddings = EmbeddingCache.load_embeddings(str(self.job_embeddings_path))
        
        self.job_embeddings = job_embeddings
        
        # CV embeddings (opsiyonel - sadece test için)
        if force_regenerate or not self.cv_embeddings_path.exists():
            print("\n🔄 CV embedding'leri oluşturuluyor...")
            cv_embeddings = self.embedder.prepare_cv_embeddings(
                self.cv_df,
                text_column='cv_text'
            )
            EmbeddingCache.save_embeddings(cv_embeddings, str(self.cv_embeddings_path))
        else:
            print("📂 Mevcut CV embedding'leri yükleniyor...")
            cv_embeddings = EmbeddingCache.load_embeddings(str(self.cv_embeddings_path))
        
        self.cv_embeddings = cv_embeddings
    
    def _setup_search_engine(self, force_regenerate: bool):
        """Faiss arama motorunu kurar"""
        print("\n[4/5] Faiss Arama Motoru")
        print("-" * 60)
        
        embedding_dim = self.embedder.get_embedding_dim()
        self.search_engine = VectorSearchEngine(embedding_dim)
        
        if force_regenerate or not self.faiss_index_path.exists():
            print("🏗️  Faiss indeksi oluşturuluyor...")
            
            # Metadata oluştur
            metadata = {
                'job_ids': self.job_df['job_id'].tolist(),
                'titles': self.job_df['title'].tolist(),
                'sectors': self.job_df['sector'].tolist()
            }
            
            self.search_engine.index_documents(
                self.job_embeddings,
                metadata=metadata,
                index_type="flatip"  # Normalize edilmiş vektörler için
            )
            
            self.search_engine.save(
                str(self.faiss_index_path),
                str(self.metadata_path)
            )
        else:
            print("📂 Mevcut Faiss indeksi yükleniyor...")
            self.search_engine.load(
                str(self.faiss_index_path),
                str(self.metadata_path)
            )
    
    def _setup_recommender(self):
        """Öneri sistemini kurar"""
        print("\n[5/5] Öneri Sistemi")
        print("-" * 60)
        
        self.recommender = TwoStageRecommender(self.job_df, self.search_engine)
    
    def get_recommendations_for_cv(self, 
                                   cv_text: str,
                                   primary_sector: str,
                                   k_primary: int = 20,
                                   k_cross: int = 15) -> Dict[str, List[JobRecommendation]]:
        """
        CV metni için iş önerileri üretir
        
        Args:
            cv_text: CV metni
            primary_sector: Tercih edilen ana sektör
            k_primary: Birincil sektör önerisi sayısı
            k_cross: Çapraz sektör önerisi sayısı
            
        Returns:
            Öneriler dictionary'si
        """
        if self.recommender is None:
            raise RuntimeError("Pipeline henüz kurulmadı! Önce setup() çağırın.")
        
        # CV'yi vektörleştir
        print(f"\n🔍 CV analiz ediliyor...")
        cv_embedding = self.embedder.encode_single(cv_text)
        
        # Önerileri al
        recommendations = self.recommender.recommend(
            cv_embedding,
            primary_sector=primary_sector,
            k_total=100,
            k_primary=k_primary,
            k_cross=k_cross
        )
        
        return recommendations
    
    def get_recommendations_for_sample_cv(self,
                                         cv_index: int,
                                         primary_sector: str,
                                         k_primary: int = 20,
                                         k_cross: int = 15) -> Dict[str, List[JobRecommendation]]:
        """
        Hazır sample CV için öneriler üretir
        
        Args:
            cv_index: Örnek CV indeksi (0-9)
            primary_sector: Tercih edilen ana sektör
            k_primary: Birincil sektör önerisi sayısı
            k_cross: Çapraz sektör önerisi sayısı
            
        Returns:
            Öneriler dictionary'si
        """
        if cv_index < 0 or cv_index >= len(self.cv_df):
            raise ValueError(f"Geçersiz CV indeksi: {cv_index}")
        
        cv_embedding = self.cv_embeddings[cv_index]
        
        print(f"\n🔍 Örnek CV analiz ediliyor (ID: {self.cv_df.iloc[cv_index]['cv_id']})...")
        
        recommendations = self.recommender.recommend(
            cv_embedding,
            primary_sector=primary_sector,
            k_total=100,
            k_primary=k_primary,
            k_cross=k_cross
        )
        
        return recommendations
    
    def get_sector_analysis(self, cv_text: str) -> pd.DataFrame:
        """CV için sektör dağılım analizi yapar"""
        if self.recommender is None:
            raise RuntimeError("Pipeline henüz kurulmadı!")
        
        cv_embedding = self.embedder.encode_single(cv_text)
        return self.recommender.get_sector_distribution(cv_embedding, k=100)
    
    def get_available_sectors(self) -> List[str]:
        """Mevcut sektörleri döndürür"""
        if self.job_df is None:
            return []
        return sorted(self.job_df['sector'].unique().tolist())
    
    def get_sample_cvs(self) -> pd.DataFrame:
        """Örnek CV'leri döndürür"""
        if self.cv_df is None:
            return pd.DataFrame()
        return self.cv_df[['cv_id', 'primary_sector', 'years_of_experience', 'skills']]


if __name__ == "__main__":
    # Test için
    print("\n" + "="*60)
    print("JOB MATCHER PIPELINE TEST")
    print("="*60)
    
    # Pipeline oluştur ve kur
    pipeline = JobMatcherPipeline(data_dir="data", models_dir="models")
    
    # İlk kurulum (veya force_regenerate=True ile yeniden üret)
    pipeline.setup(force_regenerate=False)
    
    # Mevcut sektörleri göster
    print("\n📋 Mevcut Sektörler:")
    sectors = pipeline.get_available_sectors()
    for i, sector in enumerate(sectors, 1):
        print(f"   {i}. {sector}")
    
    # Örnek CV ile test
    print("\n" + "="*60)
    print("TEST: Örnek CV ile Öneri")
    print("="*60)
    
    sample_cvs = pipeline.get_sample_cvs()
    print(f"\n📄 Örnek CV'ler:")
    print(sample_cvs)
    
    # İlk CV için öneri al
    print(f"\n🎯 İlk CV için öneri üretiliyor...")
    recommendations = pipeline.get_recommendations_for_sample_cv(
        cv_index=0,
        primary_sector=sectors[0],
        k_primary=5,
        k_cross=3
    )
    
    print(f"\n✓ Öneriler:")
    print(f"   - Birincil sektör: {len(recommendations['primary'])} öneri")
    print(f"   - Çapraz sektör: {len(recommendations['cross_sector'])} öneri")
    
    # Birkaç örnek göster
    if recommendations['primary']:
        print(f"\n📌 Birincil Sektör Önerileri (İlk 3):")
        for i, rec in enumerate(recommendations['primary'][:3], 1):
            print(f"   {i}. {rec.title} - {rec.sector} (Skor: {rec.similarity_score:.3f})")
    
    if recommendations['cross_sector']:
        print(f"\n🌐 Çapraz Sektör Önerileri (İlk 3):")
        for i, rec in enumerate(recommendations['cross_sector'][:3], 1):
            print(f"   {i}. {rec.title} - {rec.sector} (Skor: {rec.similarity_score:.3f})")
    
    print("\n✓ Pipeline test başarılı!")