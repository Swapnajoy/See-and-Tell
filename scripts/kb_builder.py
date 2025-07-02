
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import spacy
import wikipediaapi
from tqdm import tqdm

with open('data/captions/mini_coco.json', 'r', encoding='utf-8') as f:
    mini_coco = json.load(f)

nlp = spacy.load("en_core_web_sm")
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='SeeAndTellBot/1.0 (sahaswapnajoy56@gmail.com)'
)

nouns = set()
for item in mini_coco:
    caption = item['caption']

    doc = nlp(caption)
    for np in doc.noun_chunks:
        nouns.add(np.text.split()[-1].lower().strip())

print(f'Number of nouns found: {len(nouns)}')

print("Nouns extracted. Generating Knowledge-Base")


entries = []
for noun in tqdm(nouns, desc='Wiki Search'):
    page = wiki.page(noun)
    summary = page.summary.strip().lower()

    if (
    page.exists()
    and not summary.startswith("may refer to")
    and " may refer to:" not in summary[:70]
    and len(summary) > 70
    ):
        entry = {
            "title": noun,
            "content": page.summary[:1000],
            }
        entries.append(entry)

kb_path = 'data/knowledge_base/wiki_entries.jsonl'
os.makedirs(os.path.dirname(kb_path), exist_ok=True)

with open(kb_path, 'w', encoding='utf-8') as g:
    for entry in entries:
        g.write(json.dumps(entry) + '\n')

print(f"Knowledge-base generated at: {kb_path}")