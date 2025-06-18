import torch
import numpy as np

class FusionModule:
    def __init__(self, image_dim=768, text_dim=384, k=3):
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.k = k

    def __call__(self, image_vec, retrieved_text_vecs):
        assert image_vec.shape[0] == self.image_dim
        assert retrieved_text_vecs.shape == (self.k, self.text_dim)
        
        if isinstance(image_vec, np.ndarray):
            image_vec = torch.from_numpy(image_vec)
        
        if isinstance(retrieved_text_vecs, np.ndarray):
            retrieved_text_vecs = torch.from_numpy(retrieved_text_vecs)

        retrieved_text_vecs = retrieved_text_vecs.reshape(self.k*self.text_dim)
        out = torch.cat((image_vec, retrieved_text_vecs), dim=0)
        return out
    
    def test(self, image_vec, retrieved_text_vecs):
        return self.__call__(image_vec, retrieved_text_vecs)