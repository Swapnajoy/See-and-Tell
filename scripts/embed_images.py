import os
import sys

sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

import json
import torch
import numpy as np
from tqdm import tqdm

from encoder.image_encoder import ImageEncoder

image_folder = 'data/images'

model = ImageEncoder()
device = model.device

files = os.listdir(image_folder)

os.makedirs('data/image_embeddings', exist_ok=True)

with open('data/image_embeddings/image_embeddings.jsonl', 'w', encoding='utf-8') as f:
    for file_name in tqdm(files, desc='Processing Images'):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(image_folder, file_name)
            entry = {
                'image_id': file_name,
                'embedding': model.encode_from_path(file_path).tolist()
            }
            f.write(json.dumps(entry) + '\n')


