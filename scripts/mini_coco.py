import os
import sys
import json
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with open('data/captions_val2017.json', 'r', encoding='utf-8') as f:
    coco = json.load(f)

annotations = coco['annotations']
random.shuffle(annotations)

images_seen = set()
entries = []

annotation_savepath = 'data/captions/mini_coco.json'
images_savepath = 'data/images/'

for item in annotations:

    img_id = item['image_id']
    caption = item['caption']
    if img_id not in images_seen:
        entry = {
            'image_id': f"{img_id:012d}",
            'caption': f"{caption}",
            'objects': [],
        }

        entries.append(entry)
        images_seen.add(img_id)
    
    if len(images_seen)>=300:
        break

os.makedirs(os.path.dirname(annotation_savepath), exist_ok=True)

with open(annotation_savepath, 'w', encoding='utf-8') as g:
    json.dump(entries, g, indent=2)

print(f"{annotation_savepath} created with 300 captions")

import shutil

images_loadpath = 'data/val2017'
images_savepath = 'data/images'
os.makedirs(images_savepath, exist_ok=True)

for item in images_seen:
    filename = f"{item:012d}.jpg"
    src = os.path.join(images_loadpath, filename)
    dst = os.path.join(images_savepath, filename)
    shutil.copy(src, dst)

print(f"Images saved in {images_savepath}")

