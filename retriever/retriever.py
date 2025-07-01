import os
import json
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever(nn.Module):
    def __init__(
            self,
            faiss_index_path='retriever/faiss_index.bin',
            k=3,
            kb_path='data/knowledge_base/wiki_entries.jsonl',
            projection_ckpt_path='retriever/projection.pt',
            device='cuda' if torch.cuda.is_available() else 'cpu'
            ):
        super().__init__()
        self.index = faiss.read_index(faiss_index_path)
        self.k = k
        self.project = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 384),
        ).to(device)
            
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.ckpt_path = projection_ckpt_path
        self.device = device

        if os.path.exists(self.ckpt_path):
            self.project.load_state_dict(torch.load(self.ckpt_path))

        with open(kb_path, 'r', encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]

    def save(self):
        torch.save(self.project.state_dict(), self.ckpt_path)
    
    def get_topk_contents(self, joint_emb: torch.Tensor) -> list[str]:
        projected = self.project(joint_emb)
        projected_normed = torch.nn.functional.normalize(projected, p=2, dim=1)
        projected_np = projected_normed.detach().cpu().numpy().astype('float32')

        _, I = self.index.search(projected_np, self.k)
        return [[self.entries[i]['content'] for i in row] for row in I], projected

    def retrieve(self, text_emb, image_emb, gt_retrievals_emb = None) -> torch.Tensor:
        joint_emb = torch.cat([text_emb, image_emb], dim=-1)
        topk_contents, projected = self.get_topk_contents(joint_emb)

        topk_contents = np.array(topk_contents)
        flat_contents = topk_contents.flatten().tolist()

        flat_vecs = self.embedder.encode(flat_contents, convert_to_tensor=True)
        flat_vecs.requires_grad_()
        retrieved_vecs = flat_vecs.view(joint_emb.size(0), self.k, -1).to(self.device)

        if gt_retrievals_emb is not None:
            retr_loss = (1 - F.cosine_similarity(projected, gt_retrievals_emb)).mean()
            return retrieved_vecs.to(self.device), projected, retr_loss
        
        else:
            return retrieved_vecs.to(self.device), projected
    
if __name__ == '__main__':
    model = Retriever()
    text_emb = torch.ones((8, 768)).to('cuda')
    image_emb = torch.ones((8, 768)).to('cuda')
    x, y = model.retrieve(text_emb, image_emb)
    print(f"retrieved_vecs shape: {x.shape}")
    print(f"projected shape: {y.shape}")