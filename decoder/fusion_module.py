import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, query_dim=1536, ret_text_dim=384, k=3):
        super().__init__()
        self.query_dim = query_dim
        self.ret_text_dim = ret_text_dim
        self.k = k

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def __call__(self, query_vec: torch.Tensor, retrieved_vecs: torch.Tensor, projected_vec: torch.Tensor) -> torch.Tensor:
        assert query_vec.shape[1] == self.query_dim
        assert retrieved_vecs.shape[1:] == (self.k, self.ret_text_dim)
        assert projected_vec.shape[1] == self.ret_text_dim

        retrieved_vecs = retrieved_vecs.reshape(-1, self.k*self.ret_text_dim)

        scaled_query_vec = self.alpha*query_vec
        scaled_retrieved_vecs = self.beta*retrieved_vecs
        scaled_projected_vec = self.gamma*projected_vec

        fused_vec = torch.cat((scaled_query_vec, scaled_retrieved_vecs, scaled_projected_vec), dim=-1)
        return fused_vec
    
if __name__ == '__main__':
    model = FusionModule()
    query_vec = torch.ones((8, 1536)).to('cuda')
    retrieved_vecs = torch.ones((8, 3, 384)).to('cuda')
    projected_vec = torch.ones((8, 384)).to('cuda')
    print(model(query_vec, retrieved_vecs, projected_vec).shape)