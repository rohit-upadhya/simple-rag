import faiss # type: ignore
import numpy as np # type: ignore
class FaissDB:
    """Class to do the vector db operations
    """
    def __init__(self):
        self.ids = {}
    
    def build_index(
        self,
        vectors,
    ):
        vectors = vectors.astype('float32')
        vector_dimension = vectors.shape[1]
        faiss.normalize_L2(vectors)
        cpu_index = faiss.IndexFlatIP(vector_dimension)
        self.index = faiss.IndexIDMap(cpu_index)
        ids = np.arange(0, len(vectors)).astype('int64')
        self.index.add_with_ids(vectors, ids)
        
    
    def perform_search(self, query):
        
        query = query.astype('float32')
        faiss.normalize_L2(query)
        distances, ann = self.index.search(query, k=10)
        return ann[0]
    