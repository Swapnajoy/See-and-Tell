import json
import faiss

class Retriever():
    def __init__(
            self,
            faiss_index_path='retriever/faiss_index.bin',
            k=3,
            kb_path='data/knowledge_base/wiki_entries.jsonl',
            ):
        self.index = faiss.read_index(faiss_index_path)
        self.k = k

        with open(kb_path, 'r', encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]

    def __call__(self, image_emb):
        _, I = self.index.search(image_emb, k=self.k)
        results = [self.entries[i] for i in I[0]]
        return results