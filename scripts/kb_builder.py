import os
import sys
import json
import spacy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open('data\captions\mini_coco.json', 'r', encoding='utf-8') as f:
    mini_coco = json.load(f)

nlp = spacy.load("en_core_web_trf")

nouns = set()
for item in mini_coco:
    caption = item['caption']

    doc = nlp(caption)
    for np in doc.noun_chunks:
        nouns.add(np.text.split()[-1])
        if len(nouns) == 200:
            break

print("Nouns extracted")