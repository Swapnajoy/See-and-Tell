import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from encoder.image_encoder import ImageEncoder
from encoder.text_encoder import TextEncoder
from retriever.retriever import Retriever
from decoder.fusion_module import FusionModule
from decoder.decoder import Decoder

class VLMRAG(nn.Module):
    def __init__(self, mode='inference'):
        super().__init__()
        self.mode = mode
        assert self.mode in {'train', 'inference'}, f"Invalid mode: {self.mode}"
        self.img_enc = ImageEncoder()
        self.text_enc = TextEncoder()
        self.retriever = Retriever()
        self.fusion = FusionModule()
        self.decoder = Decoder()

        for param in self.parameters():
            param.requires_grad = False

        if self.mode == 'train':
            self.unfreeze_projection_params()

    def forward(self, image_path, query, target_ids = None, target_mask = None):
        img_embed = self.img_enc(image_path)
        text_embed = self.text_enc(query)
        retrieved_vecs, distances = self.retriever.retrieve(text_embed, img_embed)
        query_vec = torch.cat([text_embed, img_embed], dim=-1)
        fused_vec = self.fusion(query_vec, retrieved_vecs)
        if target_ids is not None:
            target_ids = target_ids.to(fused_vec.device)
            target_mask = target_mask.to(fused_vec.device)
        output = self.decoder(fused_vec, self.mode, target_ids, target_mask)

        return output, distances
    
    def unfreeze_projection_params(self):

        for param in self.retriever.project.parameters():
            param.requires_grad = True

        for param in self.decoder.projection.parameters():
            param.requires_grad = True
