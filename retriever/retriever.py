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
            ):
        super().__init__()
        self.index = faiss.read_index(faiss_index_path)
        self.k = k
        self.project = nn.Linear(768, 384)
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.ckpt_path = projection_ckpt_path

        if os.path.exists(self.ckpt_path):
            self.project.load_state_dict(torch.load(self.ckpt_path))

        with open(kb_path, 'r', encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]

    def save(self):
        torch.save(self.project.state_dict, self.ckpt_path)

    def forward(self, image_emb: torch.Tensor) -> torch.Tensor:
        projected = self.project(image_emb).detach().cpu().numpy().astype('float32').reshape(1, -1)
        _, I = self.index.search(projected, self.k)
        topk_contents = [self.entries[i]['content'] for i in I[0]]
        
        retrieved_vecs = self.embedder.encode(topk_contents, convert_to_tensor=True)
        return retrieved_vecs.to(image_emb.device)