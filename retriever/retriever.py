import os
import json
import faiss
import torch
import torch.nn as nn

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
        self.ckpt_path = projection_ckpt_path

        if os.path.exists(self.ckpt_path):
            self.project.load_state_dict(torch.load(self.ckpt_path))

        with open(kb_path, 'r', encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]

    def save(self):
        torch.save(self.project.state_dict, self.ckpt_path)

    def __call__(self, image_emb):
        with torch.no_grad():
            projected = self.project(image_emb)
            projected = projected.cpu().numpy().astype('float32')
            projected = projected.reshape(1, -1)

        _, I = self.index.search(image_emb, k=self.k)
        results = [self.entries[i] for i in I[0]]
        return results