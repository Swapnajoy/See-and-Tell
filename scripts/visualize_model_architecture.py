import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchviz import make_dot
import torch
from vlmrag.vlmrag_model import VLMRAG

model = VLMRAG(mode='train')

sample_input_image = ['data/images/000000127394.jpg']
sample_input_query = ["Describe this image."]

output, _ = model(sample_input_image, sample_input_query, gt_retrievals_emb=None)

make_dot(output, params=dict(model.named_parameters())).render("model_architecture", format="png")
