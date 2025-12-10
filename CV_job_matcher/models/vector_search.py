"""
Faiss Vektör Arama Modülü
Yüksek performanslı benzerlik araması için Faiss kullanır.
"""

import numpy as np
import faiss
import pickle
from typing import Tuple, List, Dict
import os


class FaissVectorSearch:
    """Faiss kullanarak vektör benzerlik araması yapan sınıf"""
    
    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: Vektör boyutu
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.is_trained = False
        print(f"🔧 Faiss arama motoru başlatıldı (dim={embedding_dim})")
    
    def build_index(self, embeddings: np.ndarray, index_type: str = "flatl2"):
        """
        Faiss indeksini oluşturur
        
        Args:
            embeddings: (n_samples, embedding_dim) boyutunda embedding matrisi
            index_type: İndeks tipi
                - "flatl2": Exact search, L2 distance (küçük-orta veri setleri için)
                - "flatip": Exact search, Inner Product (normalize edilmiş vektörler için)
                - "ivf": Approximate search (büyük veri setleri için)
        """
        n_samples, dim = embeddings.shape
        
        if dim != self.embedding_dim:
            raise ValueError(
                f"Embedding boyutu uyuşmuyor! Beklenen: {self.embedding_dim}, "
                f"Gelen: {dim}"
            )
        
        print(f"🏗️  Faiss indeksi oluşturuluyor...")
        print(f"   - Veri sayısı: {n_samples}")
        print(f"   - Vektör boyutu: {dim}")
        print(f"   - İndeks tipi: {index_type}")
        
        # Vektörleri float32'ye dönüştür (Faiss gereksinimi)
        embeddings = embeddings.astype('float32')
        
        if index_type == "flatl2":
            # L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(dim)
            
        elif index_type == "flatip":
            # Inner Product (normalize edilmiş vektörler için kosinüs benzerliği)
            self.index = faiss.IndexFlatIP(dim)
            
        elif index_type == "ivf":
            # IVF (Inverted File) - approximate search
            # Büyük veri setleri için daha hızlı ama approximate
            nlist = min(100, n_samples // 10)  # Cluster sayısı
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            
            # IVF indeksi training gerektirir
            print(f"   - Training IVF indeksi ({nlist} cluster)...")
            self.index.train(embeddings)
            self.is_trained = True
        
        else:
            raise ValueError(f"Desteklenmeyen indeks tipi: {index_type}")
        
        # Vektörleri indekse ekle
        self.index.add(embeddings)
        
        print(f"✓ İndeks oluşturuldu. Toplam vektör sayısı: {self.index.ntotal}")
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query vektörüne en yakın k vektörü bulur
        
        Args:
            query_vector: (embedding_dim,) veya (1, embedding_dim) boyutunda query
            k: Döndürülecek sonuç sayısı
            
        Returns:
            distances: (k,) boyutunda uzaklık/skor dizisi
            indices: (k,) boyutunda indeks dizisi
        """
        if self.index is None:
            raise RuntimeError("İndeks henüz oluşturulmadı! Önce build_index() çağırın.")
        
        # Vektör şeklini düzenle
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # float32'ye dönüştür
        query_vector = query_vector.astype('float32')
        
        # Arama yap
        k = min(k, self.index.ntotal)  # k, toplam vektör sayısından fazla olamaz
        distances, indices = self.index.search(query_vector, k)
        
        return distances[0], indices[0]
    
    def search_batch(self, query_vectors: np.ndarray, 
                    k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Birden fazla query için batch arama
        
        Args:
            query_vectors: (n_queries, embedding_dim) boyutunda query matrisi
            k: Her query için döndürülecek sonuç sayısı
            
        Returns:
            distances: (n_queries, k) boyutunda uzaklık matrisi
            indices: (n_queries, k) boyutunda indeks matrisi
        """
        if self.index is None:
            raise RuntimeError("İndeks henüz oluşturulmadı!")
        
        query_vectors = query_vectors.astype('float32')
        k = min(k, self.index.ntotal)
        
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices
    
    def save_index(self, filepath: str):
        """İndeksi diske kaydet"""
        if self.index is None:
            raise RuntimeError("Kaydedilecek indeks yok!")
        
        faiss.write_index(self.index, filepath)
        print(f"✓ Faiss indeksi kaydedildi: {filepath}")
    
    def load_index(self, filepath: str):
        """İndeksi diskten yükle"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"İndeks dosyası bulunamadı: {filepath}")
        
        self.index = faiss.read_index(filepath)
        print(f"✓ Faiss indeksi yüklendi: {filepath}")
        print(f"   - Toplam vektör: {self.index.ntotal}")
    
    def get_index_size(self) -> int:
        """İndeksteki toplam vektör sayısını döndür"""
        return self.index.ntotal if self.index else 0


class VectorSearchEngine:
    """Üst seviye vektör arama motoru - Faiss'i wrap eder"""
    
    def __init__(self, embedder_dim: int):
        """
        Args:
            embedder_dim: Embedding boyutu
        """
        self.faiss_search = FaissVectorSearch(embedder_dim)
        self.metadata = None  # İlgili metadata (job_id, title, vb.)
    
    def index_documents(self, embeddings: np.ndarray, 
                       metadata: Dict = None,
                       index_type: str = "flatip"):
        """
        Dökümanları indeksle
        
        Args:
            embeddings: Döküman embedding'leri
            metadata: Döküman metadata'sı (opsiyonel)
            index_type: Faiss indeks tipi
        """
        self.faiss_search.build_index(embeddings, index_type=index_type)
        self.metadata = metadata
        
        if metadata:
            print(f"✓ {len(metadata)} döküman metadata'sı kaydedildi")
    
    def search_similar(self, query_embedding: np.ndarray, 
                      k: int = 20) -> Tuple[List[int], List[float]]:
        """
        Benzer dökümanları ara
        
        Args:
            query_embedding: Query vektörü
            k: Döndürülecek sonuç sayısı
            
        Returns:
            indices: Bulunan döküman indeksleri
            scores: Benzerlik skorları
        """
        distances, indices = self.faiss_search.search(query_embedding, k)
        
        # Inner Product (IP) indeksi kullanıyorsak, skorlar zaten benzerlik
        # L2 kullanıyorsak, uzaklığı benzerliğe çevir
        scores = distances.tolist()
        
        return indices.tolist(), scores
    
    def save(self, index_path: str, metadata_path: str = None):
        """Arama motorunu kaydet"""
        self.faiss_search.save_index(index_path)
        
        if self.metadata and metadata_path:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"✓ Metadata kaydedildi: {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str = None):
        """Arama motorunu yükle"""
        self.faiss_search.load_index(index_path)
        
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✓ Metadata yüklendi: {metadata_path}")


if __name__ == "__main__":
    # Test için
    print("=" * 60)
    print("Faiss Vector Search Test")
    print("=" * 60)
    
    # Test verileri oluştur
    np.random.seed(42)
    embedding_dim = 384
    n_samples = 1000
    
    print(f"\n📊 Test verileri oluşturuluyor...")
    print(f"   - Vektör boyutu: {embedding_dim}")
    print(f"   - Vektör sayısı: {n_samples}")
    
    # Random embedding'ler oluştur ve normalize et
    embeddings = np.random.randn(n_samples, embedding_dim).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Arama motoru oluştur
    search_engine = VectorSearchEngine(embedding_dim)
    search_engine.index_documents(embeddings, index_type="flatip")
    
    # Test query
    query = np.random.randn(embedding_dim).astype('float32')
    query = query / np.linalg.norm(query)
    
    print(f"\n🔍 Arama yapılıyor (k=10)...")
    indices, scores = search_engine.search_similar(query, k=10)
    
    print(f"\n✓ Top 10 sonuç:")
    for i, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"   {i}. İndeks: {idx:4d} | Skor: {score:.4f}")
    
    print(f"\n✓ Test başarılı!")