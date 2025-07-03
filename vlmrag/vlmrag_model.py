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
            self.unfreeze_upper_decoder_layers()

    def forward(self, image_path, query, gt_retrievals_emb = None, target_ids = None, target_mask = None):
        img_embed = self.img_enc(image_path)
        text_embed = self.text_enc(query)

        retr_loss = None
        if gt_retrievals_emb is not None:
            retrieved_vecs, projected_vec, retr_loss = self.retriever.retrieve(text_embed, img_embed, gt_retrievals_emb)

        else:
            retrieved_vecs, projected_vec = self.retriever.retrieve(text_embed, img_embed)

        query_vec = torch.cat([text_embed, img_embed], dim=-1)
        fused_vec = self.fusion(query_vec, retrieved_vecs, projected_vec)

        if target_ids is not None:
            target_ids = target_ids.to(fused_vec.device)
            target_mask = target_mask.to(fused_vec.device)

        output = self.decoder(fused_vec, self.mode, target_ids, target_mask)

        if retr_loss is not None:
            return output, retr_loss
        else:
            return output
    
    def unfreeze_projection_params(self):

        for param in self.retriever.project.parameters():
            param.requires_grad = True

        for param in self.decoder.projection.parameters():
            param.requires_grad = True

    def unfreeze_upper_decoder_layers(self):
        for name, param in self.decoder.named_parameters():
            if "block.0" in name or "block.1" in name or "block.2" in name:
                param.requires_grad = False

            else:
                param.requires_grad = True
