import os
import sys
import json
import spacy
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open('data/captions/mini_coco.json', 'r', encoding='utf-8') as f:
    mini_coco = json.load(f)

wiki_entries = {}
with open('data/knowledge_base/wiki_entries.jsonl', 'r', encoding='utf-8') as g:
    for line in g:
        entry = json.loads(line)
        wiki_entries[entry['title']] = entry['content']

nlp = spacy.load("en_core_web_trf")

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

triplets = []
for item in tqdm(mini_coco, desc='Preparing triplets'):
    image_id = item['image_id'] + '.jpg'
    caption = item['caption'].strip()
    doc = nlp(caption)

    augmented_caption = caption

    nouns = set()
    for noun in doc.noun_chunks:
        nouns.add(noun.text.split()[-1].lower().strip())
    
    for noun in nouns:
        if noun in wiki_entries.keys():
            augmented_caption += ' ' + wiki_entries[noun]

    image_path = os.path.join('data/images', image_id)

    triplet = {
        'image_path': image_path,
        'query': random.choice(queries),
        'caption': augmented_caption,
    }

    triplets.append(triplet)

triplets_path = 'data/triplets/triplets.jsonl'
os.makedirs(os.path.dirname(triplets_path), exist_ok=True)

with open(triplets_path, 'w', encoding='utf-8') as h:
    for triplet in triplets:
        h.write(json.dumps(triplet) + '\n')

print(f"Triplets generated at: {triplets_path}")

with open('data/triplets/triplets_preview.json', 'w', encoding='utf-8') as preview:
    json.dump(triplets[:10], preview, indent=2)

print(f"Triplets_preview generated at: 'data/triplets/triplets_preview.json'")