import torch

class FusionModule:
    def __init__(self, query_dim=1536, ret_text_dim=384, k=3):
        self.query_dim = query_dim
        self.ret_text_dim = ret_text_dim
        self.k = k

    def __call__(self, query_vec: torch.Tensor, retrieved_vecs: torch.Tensor) -> torch.Tensor:
        assert query_vec.shape[0] == self.query_dim
        assert retrieved_vecs.shape == (self.k, self.ret_text_dim)

        retrieved_vecs = retrieved_vecs.reshape(self.k*self.ret_text_dim)
        return torch.cat((query_vec, retrieved_vecs), dim=0)