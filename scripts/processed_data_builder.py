import os
import sys
import json
import spacy
import re
import random
from tqdm import tqdm
import torch.nn.functional as F

import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

faiss_index_path='retriever/faiss_index.bin'
index = faiss.read_index(faiss_index_path)

kb_path='data/knowledge_base/wiki_entries.jsonl'
with open(kb_path, 'r', encoding='utf-8') as f:
    kb = [json.loads(line) for line in f]

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

with open('data/captions/mini_coco.json', 'r', encoding='utf-8') as f:
    mini_coco = json.load(f)

wiki_entries = {}
with open('data/knowledge_base/wiki_entries.jsonl', 'r', encoding='utf-8') as g:
    for line in g:
        entry = json.loads(line)
        wiki_entries[entry['title']] = entry['content']

nlp = spacy.load("en_core_web_sm")

queries = [
    "What is happening in this image?",
    "Can you describe the scene?",
    "What does this picture show?",
    "Explain the contents of the image.",
    "Tell me what's going on in the photo.",
    "Describe everything visible in the image.",
    "What can be seen here?",
    "What is the subject of this image?",
    "Summarize the visual elements in this picture.",
    "Provide a description of this image."
]

entries = []
for item in tqdm(mini_coco, desc='Preparing processed_data'):
    image_id = item['image_id'] + '.jpg'
    caption = item['caption'].strip()
    doc = nlp(caption)

    augmented_caption = caption

    nouns = set()
    for noun in doc.noun_chunks:
        nouns.add(noun.text.split()[-1].lower().strip())
    
    for noun in nouns:
        if noun in wiki_entries.keys():
            content = wiki_entries[noun]
            sentences = re.split(r'(?<=[.!?]) +', content)
            limited_content = ' '.join(sentences[:2])
            augmented_caption += ' ' + limited_content

    image_path = os.path.join('data/images', image_id)

    caption_embed = embedder.encode([caption])
    caption_embed = caption_embed.astype("float32")
    _, I = index.search(caption_embed, 3)

    gt = [kb[idx]['content'] for idx in I[0]]
    gt_emb = embedder.encode(gt, convert_to_tensor=True)
    gt_emb = F.normalize(gt_emb, p=2, dim=1)
    gt_emb = gt_emb.mean(dim=0, keepdim=True).cpu().tolist()

    entry = {
        'image_path': image_path,
        'query': random.choice(queries),
        'caption': augmented_caption,
        'faiss_indices': I[0].tolist(),
        'gt_emb': gt_emb,
    }

    entries.append(entry)

processed_data_path = 'data/processed/processed_data.jsonl'
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

with open(processed_data_path, 'w', encoding='utf-8') as h:
    for entry in entries:
        h.write(json.dumps(entry) + '\n')

print(f"Processed_data generated at: {processed_data_path}")

with open('data/processed/processed_data_preview.json', 'w', encoding='utf-8') as preview:
    json.dump(entries[:10], preview, indent=2)

print(f"Processed_data_preview generated at: 'data/processed/processed_data_preview.json'")