import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from encoder.image_encoder import ImageEncoder
from encoder.text_encoder import TextEncoder
from retriever.retriever import Retriever
from decoder.fusion_module import FusionModule
from decoder.decoder import Decoder

class VLMRAG:
    def __init__(self, mode='inference'):

        self.mode = mode
        assert self.mode in {'train', 'inference'}, f"Invalid mode: {self.mode}"

        self.img_enc = ImageEncoder()
        self.text_enc = TextEncoder()
        self.retriever = Retriever()
        self.fusion = FusionModule()
        self.decoder = Decoder()

    def __call__(self, image_path, query, target_ids: torch.Tensor = None, target_mask: torch.Tensor = None):
        img_embed = self.img_enc(image_path)
        text_embed = self.text_enc(query)
        retrieved_vecs = self.retriever.retrieve(text_embed, img_embed)
        query_vec = torch.cat([text_embed, img_embed], dim=-1)
        fused_vec = self.fusion(query_vec, retrieved_vecs)
        if target_ids is not None:
            target_ids = target_ids.to(fused_vec.device)
            target_mask = target_mask.to(fused_vec.device)
        output = self.decoder(fused_vec, self.mode, target_ids, target_mask)

        return output
