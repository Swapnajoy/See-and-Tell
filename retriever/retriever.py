import os
import json
import faiss
import torch
import torch.nn as nn
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
        #print(f'projected_norm: {projected.norm(dim=1)}')
        projected = torch.nn.functional.normalize(projected, p=2, dim=1)
        projected = projected.detach().cpu().numpy().astype('float32')

        D, I = self.index.search(projected, self.k)
        #print(f'Distances: {D}')
        #print("Embed retrievals:")
        for idx in I[0]:
            print("-", self.entries[idx]['title'])

        return [[self.entries[i]['content'] for i in row] for row in I], D

    def retrieve(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        joint_emb = torch.cat([text_emb, image_emb], dim=-1)
        topk_contents, distances = self.get_topk_contents(joint_emb)
        topk_contents = np.array(topk_contents)

        flat_contents = topk_contents.flatten().tolist()
        flat_vecs = self.embedder.encode(flat_contents, convert_to_tensor=True)
        retrieved_vecs = flat_vecs.view(joint_emb.size(0), self.k, -1).to(self.device) 
        return retrieved_vecs.to(self.device), distances
    
if __name__ == '__main__':
    model = Retriever()
    text_emb = torch.ones((8, 768)).to('cuda')
    image_emb = torch.ones((8, 768)).to('cuda')
    print(model.retrieve(text_emb, image_emb).shape)