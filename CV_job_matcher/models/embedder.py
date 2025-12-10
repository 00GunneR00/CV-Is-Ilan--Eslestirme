"""
NLP Pipeline Modülü
Transformers (BERT) kullanarak metin embedding'leri oluşturur.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import pandas as pd
from tqdm import tqdm


class TextEmbedder:
    """BERT tabanlı metin embedding sınıfı"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Args:
            model_name: Kullanılacak sentence transformer modeli
                       Türkçe desteği için multilingual model kullanıyoruz
        """
        print(f"📦 Model yükleniyor: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model yüklendi. Embedding boyutu: {self.embedding_dim}")
        
        # GPU varsa kullan
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"✓ Cihaz: {self.device}")
    
    def encode_texts(self, texts: Union[List[str], pd.Series], 
                     batch_size: int = 32,
                     show_progress: bool = True) -> np.ndarray:
        """
        Metin listesini embedding vektörlerine dönüştürür
        
        Args:
            texts: Embedding'i alınacak metinler
            batch_size: Batch boyutu (büyük veriler için)
            show_progress: İlerleme çubuğu göster
            
        Returns:
            (n_samples, embedding_dim) boyutunda numpy array
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Boş veya None değerleri temizle
        texts = [str(t) if t is not None else "" for t in texts]
        
        print(f"🔄 {len(texts)} metin vektörleştiriliyor...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Kosinüs benzerliği için normalize et
        )
        
        print(f"✓ Vektörleştirme tamamlandı. Shape: {embeddings.shape}")
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Tek bir metni vektörleştirir
        
        Args:
            text: Vektörleştirilerek metin
            
        Returns:
            (embedding_dim,) boyutunda numpy array
        """
        embedding = self.model.encode(
            [str(text)],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]
    
    def prepare_job_embeddings(self, job_df: pd.DataFrame,
                               text_columns: List[str] = None) -> np.ndarray:
        """
        İş ilanı DataFrame'inden embedding'ler oluşturur
        
        Args:
            job_df: İş ilanları DataFrame'i
            text_columns: Birleştirilecek metin sütunları
            
        Returns:
            İş ilanı embedding'leri
        """
        if text_columns is None:
            text_columns = ['title', 'description', 'required_skills']
        
        print(f"📄 İş ilanları için metin hazırlanıyor...")
        
        # Tüm text sütunlarını birleştir
        combined_texts = []
        for _, row in job_df.iterrows():
            text_parts = []
            for col in text_columns:
                if col in job_df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            combined_text = " | ".join(text_parts)
            combined_texts.append(combined_text)
        
        return self.encode_texts(combined_texts)
    
    def prepare_cv_embeddings(self, cv_df: pd.DataFrame,
                             text_column: str = 'cv_text') -> np.ndarray:
        """
        CV DataFrame'inden embedding'ler oluşturur
        
        Args:
            cv_df: CV DataFrame'i
            text_column: CV metni sütunu
            
        Returns:
            CV embedding'leri
        """
        print(f"📄 CV'ler için metin hazırlanıyor...")
        
        if text_column not in cv_df.columns:
            raise ValueError(f"'{text_column}' sütunu bulunamadı!")
        
        cv_texts = cv_df[text_column].tolist()
        return self.encode_texts(cv_texts)
    
    def get_embedding_dim(self) -> int:
        """Embedding boyutunu döndürür"""
        return self.embedding_dim


class EmbeddingCache:
    """Embedding'leri önbelleğe alma ve yükleme sınıfı"""
    
    @staticmethod
    def save_embeddings(embeddings: np.ndarray, filepath: str):
        """Embedding'leri kaydet"""
        np.save(filepath, embeddings)
        print(f"✓ Embedding'ler kaydedildi: {filepath}")
    
    @staticmethod
    def load_embeddings(filepath: str) -> np.ndarray:
        """Embedding'leri yükle"""
        embeddings = np.load(filepath)
        print(f"✓ Embedding'ler yüklendi: {filepath} - Shape: {embeddings.shape}")
        return embeddings
    
    @staticmethod
    def embeddings_exist(filepath: str) -> bool:
        """Embedding dosyası var mı kontrol et"""
        import os
        return os.path.exists(filepath)


if __name__ == "__main__":
    # Test için
    print("=" * 60)
    print("NLP Pipeline Test")
    print("=" * 60)
    
    # Embedder oluştur
    embedder = TextEmbedder()
    
    # Test metinleri
    test_texts = [
        "Python ve Machine Learning deneyimi olan Senior Data Scientist arıyoruz",
        "5 yıllık Python, TensorFlow ve NLP deneyimim var",
        "JavaScript ve React ile frontend geliştirme yapabilecek developer aranıyor"
    ]
    
    print("\n📝 Test metinleri vektörleştiriliyor...")
    embeddings = embedder.encode_texts(test_texts, show_progress=False)
    
    print(f"\n✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Embedding boyutu: {embedder.get_embedding_dim()}")
    
    # Kosinüs benzerliği hesapla
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    
    print("\n📊 Kosinüs Benzerlik Matrisi:")
    print(similarities)
    
    print(f"\n✓ Text 1 ve Text 2 benzerliği: {similarities[0, 1]:.4f}")
    print(f"✓ Text 1 ve Text 3 benzerliği: {similarities[0, 2]:.4f}")