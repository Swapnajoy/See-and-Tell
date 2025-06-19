import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from transformers import ViTModel

class ImageEncoder:
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.transformer = ViTModel.from_pretrained(model_name).to(device).eval()
        self.tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __call__(self, image) -> np.ndarray:
        out = self.tf(image)
        out = out.unsqueeze(0).to(self.device)
        out = self.transformer(out)
        cls_token_emb = out.last_hidden_state[:, 0, :]
        cls_token_emb = cls_token_emb.detach().cpu().numpy().squeeze(0)
        return cls_token_emb
    
    def encode_from_path(self, path) -> np.ndarray:
        image = Image.open(path).convert('RGB')
        cls_token_emb = self.__call__(image)
        return cls_token_emb