import torch
import numpy as np

class FusionModule:
    def __init__(self, image_dim=768, text_dim=384, k=3):
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.k = k

    def __call__(self, image_vec: torch.Tensor, retrieved_vecs: torch.Tensor) -> torch.Tensor:
        assert image_vec.shape[0] == self.image_dim
        assert retrieved_vecs.shape == (self.k, self.text_dim)

        retrieved_text_vecs = retrieved_text_vecs.reshape(self.k*self.text_dim)
        return torch.cat((image_vec, retrieved_text_vecs), dim=0)