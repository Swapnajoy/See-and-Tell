import os

import json
import spacy
import wikipediaapi
from tqdm import tqdm

with open('data/captions/mini_coco.json', 'r', encoding='utf-8') as f:
    mini_coco = json.load(f)

nlp = spacy.load("en_core_web_trf")
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
        if len(nouns) == 200:
            break

    if len(nouns) == 200:
            break
    
print("Nouns extracted. Generating Knowledge-Base")


entries = []
for noun in tqdm(nouns, desc='Wiki Search'):
    page = wiki.page(noun)

    if page.exists():
        summary = page.summary.strip().lower()
        if not summary.startswith('may refer to'):
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