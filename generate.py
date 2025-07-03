
import torch
import torch.nn as nn

from vlmrag.vlmrag_model import VLMRAG
from utils.config import load_config

cfg = load_config()

image_path = ['data/original_val2017_img_and_captions/val2017/000000314709.jpg']
text_query = ['Describe the image.']

device = cfg['device']
ckpt_path = cfg['ckpt_path']

model = VLMRAG(mode='inference')
state_dict = torch.load(cfg['ckpt_path'], map_location=device)
model.retriever.project.load_state_dict(state_dict['retriever_proj'])
model.decoder.load_state_dict(state_dict['decoder'])

model.to(device)

output = model(image_path, text_query)

print(f"Description: {output}")
