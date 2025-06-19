import os
import torch
import torch.nn as nn

from encoder.image_encoder import ImageEncoder
from retriever.retriever import Retriever
from generator.fusion_module import FusionModule
from generator.decoder import Decoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_path = 'data/images/000000052017.jpg'
describe_this = input('User Query: ')

class Generator:
    def __init__(self):
        self.image_encoder = ImageEncoder()
        self.retriever = Retriever()
        self.fusion_module = FusionModule()
        self.decoder = Decoder()

    def __call__(self, image_path, describe_this):
        if os.path.exists(image_path):
            image_vec = self.image_encoder.encode_from_path(image_path)
            retrieved_text_vecs = self.retriever(image_vec)
            encoder_outputs = self.fusion_module(image_vec, retrieved_text_vecs)
            output = self.decoder(describe_this, encoder_outputs)
            return output
        
model = Generator()

with torch.no_grad():
    output = model(image_path, describe_this)
    print(output)