import json
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index = faiss.read_index("retriever/faiss_index.bin")

with open('data/knowledge_base/wiki_entries.jsonl', 'r', encoding='utf-8') as f:
    entries = [json.loads(line) for line in f]

query = input('Enter Query: ')
query_emb = model.encode([query])

D, I = index.search(query_emb, k=3)
results = [entries[i] for i in I[0]]

for result in results:
    print(result)