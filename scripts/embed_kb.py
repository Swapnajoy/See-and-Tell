import json
from sentence_transformers import SentenceTransformer
import faiss

with open('data/knowledge_base/wiki_entries.jsonl', 'r', encoding='utf-8') as f:
    entries = [json.loads(line) for line in f]

content=[]
for entry in entries:
    content.append(entry['content'])

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(content, batch_size=32, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print(index.ntotal)