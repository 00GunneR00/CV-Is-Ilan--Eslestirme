"""
Test Script - Tüm modülleri test eder
"""

import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

def test_data_generator():
    """Veri üretimi testi"""
    print("\n" + "="*60)
    print("TEST 1/5: Veri Üretimi")
    print("="*60)
    
    from utils.data_generator import SyntheticDataGenerator
    
    generator = SyntheticDataGenerator()
    jobs_df = generator.generate_job_postings(n=100)
    cvs_df = generator.generate_sample_cvs(n=5)
    
    assert len(jobs_df) == 100, "İş ilanı sayısı yanlış!"
    assert len(cvs_df) == 5, "CV sayısı yanlış!"
    assert 'job_id' in jobs_df.columns, "job_id sütunu eksik!"
    assert 'cv_text' in cvs_df.columns, "cv_text sütunu eksik!"
    
    print(f"✓ {len(jobs_df)} iş ilanı oluşturuldu")
    print(f"✓ {len(cvs_df)} CV oluşturuldu")
    print("✓ Test başarılı!")
    
    return jobs_df, cvs_df


def test_embedder(jobs_df, cvs_df):
    """Embedder testi"""
    print("\n" + "="*60)
    print("TEST 2/5: Text Embedder")
    print("="*60)
    
    from models.embedder import TextEmbedder
    
    embedder = TextEmbedder()
    
    # İş ilanı embedding'leri
    job_embeddings = embedder.prepare_job_embeddings(jobs_df.head(10))
    assert job_embeddings.shape == (10, embedder.get_embedding_dim()), "Job embedding shape yanlış!"
    
    # CV embedding'leri
    cv_embeddings = embedder.prepare_cv_embeddings(cvs_df)
    assert cv_embeddings.shape == (5, embedder.get_embedding_dim()), "CV embedding shape yanlış!"
    
    print(f"✓ Embedding boyutu: {embedder.get_embedding_dim()}")
    print(f"✓ Job embeddings: {job_embeddings.shape}")
    print(f"✓ CV embeddings: {cv_embeddings.shape}")
    print("✓ Test başarılı!")
    
    return embedder, job_embeddings, cv_embeddings


def test_vector_search(embedder, job_embeddings):
    """Faiss arama testi"""
    print("\n" + "="*60)
    print("TEST 3/5: Faiss Vector Search")
    print("="*60)
    
    from models.vector_search import VectorSearchEngine
    import numpy as np
    
    search_engine = VectorSearchEngine(embedder.get_embedding_dim())
    search_engine.index_documents(job_embeddings, index_type="flatip")
    
    # Test query
    query_vec = job_embeddings[0]  # İlk job'u query olarak kullan
    indices, scores = search_engine.search_similar(query_vec, k=5)
    
    assert len(indices) == 5, "Dönen sonuç sayısı yanlış!"
    assert indices[0] == 0, "En yakın sonuç kendisi olmalı!"
    assert scores[0] >= scores[1], "Skorlar azalan sırada olmalı!"
    
    print(f"✓ İndeks oluşturuldu: {search_engine.faiss_search.get_index_size()} vektör")
    print(f"✓ Arama yapıldı: {len(indices)} sonuç")
    print(f"✓ Top skor: {scores[0]:.4f}")
    print("✓ Test başarılı!")
    
    return search_engine


def test_recommender(jobs_df, search_engine, cv_embeddings):
    """Öneri sistemi testi"""
    print("\n" + "="*60)
    print("TEST 4/5: Two-Stage Recommender")
    print("="*60)
    
    from models.recommender import TwoStageRecommender
    
    recommender = TwoStageRecommender(jobs_df, search_engine)
    
    # Test CV
    cv_embedding = cv_embeddings[0]
    primary_sector = jobs_df['sector'].iloc[0]
    
    recommendations = recommender.recommend(
        cv_embedding,
        primary_sector=primary_sector,
        k_total=20,
        k_primary=5,
        k_cross=3
    )
    
    assert 'primary' in recommendations, "Primary öneriler eksik!"
    assert 'cross_sector' in recommendations, "Cross-sector öneriler eksik!"
    assert len(recommendations['primary']) <= 5, "Primary öneri sayısı fazla!"
    
    print(f"✓ Primary öneriler: {len(recommendations['primary'])}")
    print(f"✓ Cross-sector öneriler: {len(recommendations['cross_sector'])}")
    print("✓ Test başarılı!")
    
    return recommender, recommendations


def test_pipeline():
    """Pipeline testi"""
    print("\n" + "="*60)
    print("TEST 5/5: Full Pipeline")
    print("="*60)
    
    from pipeline import JobMatcherPipeline
    
    pipeline = JobMatcherPipeline(data_dir="test_data", models_dir="test_models")
    
    print("⚠️  Pipeline kurulumu başlıyor (birkaç dakika sürebilir)...")
    pipeline.setup(force_regenerate=True)
    
    # Test
    sectors = pipeline.get_available_sectors()
    sample_cvs = pipeline.get_sample_cvs()
    
    assert len(sectors) > 0, "Sektör bulunamadı!"
    assert len(sample_cvs) > 0, "Örnek CV bulunamadı!"
    
    # Öneri al
    recommendations = pipeline.get_recommendations_for_sample_cv(
        cv_index=0,
        primary_sector=sectors[0],
        k_primary=5,
        k_cross=3
    )
    
    assert 'primary' in recommendations, "Öneriler alınamadı!"
    
    print(f"✓ Sektör sayısı: {len(sectors)}")
    print(f"✓ CV sayısı: {len(sample_cvs)}")
    print(f"✓ Öneriler alındı")
    print("✓ Test başarılı!")
    
    # Temizlik
    import shutil
    shutil.rmtree("test_data", ignore_errors=True)
    shutil.rmtree("test_models", ignore_errors=True)
    print("✓ Test dosyaları temizlendi")


def main():
    """Ana test fonksiyonu"""
    print("\n" + "="*70)
    print(" " * 15 + "CV İŞ EŞLEŞTIRME SİSTEMİ - TEST PAKETİ")
    print("="*70)
    
    try:
        # Test 1: Veri üretimi
        jobs_df, cvs_df = test_data_generator()
        
        # Test 2: Embedder
        embedder, job_embeddings, cv_embeddings = test_embedder(jobs_df, cvs_df)
        
        # Test 3: Vector search
        search_engine = test_vector_search(embedder, job_embeddings)
        
        # Test 4: Recommender
        recommender, recommendations = test_recommender(jobs_df, search_engine, cv_embeddings)
        
        # Test 5: Full pipeline (opsiyonel - uzun sürer)
        run_pipeline_test = input("\n🔍 Full pipeline testi çalıştırılsın mı? (y/n): ").lower()
        if run_pipeline_test == 'y':
            test_pipeline()
        
        # Özet
        print("\n" + "="*70)
        print(" " * 25 + "🎉 TÜM TESTLER BAŞARILI! 🎉")
        print("="*70)
        print("\n✅ Veri Üretimi")
        print("✅ Text Embedder")
        print("✅ Faiss Vector Search")
        print("✅ Two-Stage Recommender")
        if run_pipeline_test == 'y':
            print("✅ Full Pipeline")
        print("\n" + "="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print(" " * 30 + "❌ TEST BAŞARISIZ!")
        print("="*70)
        print(f"\nHata: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()