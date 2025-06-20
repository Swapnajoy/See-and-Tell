import os
import json
import faiss
import torch
import torch.nn as nn
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
            nn.Linear(1536, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 384)
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
        projected = self.project(joint_emb).detach().cpu().numpy().astype('float32').reshape(1, -1)
        _, I = self.index.search(projected, self.k)
        return [self.entries[i]['content'] for i in I[0]]

    def retrieve(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        joint_emb = torch.cat([text_emb, image_emb], dim=-1)
        topk_contents = self.get_topk_contents(joint_emb)
        
        retrieved_vecs = self.embedder.encode(topk_contents, convert_to_tensor=True)
        return retrieved_vecs.to(self.device)
    
