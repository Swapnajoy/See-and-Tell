import os
import sys
import json
import spacy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open('data\captions\mini_coco.json', 'r', encoding='utf-8') as f:
    mini_coco = json.load(f)

count = 0
for item in mini_coco:
    count += 1
    caption = item['caption']
    print(caption)
    if count == 5:
        break