import torch
import torchvision.transforms as transforms
from PIL import Image

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

    def __call__(self, image_path) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        out = self.tf(image).unsqueeze(0).to(self.device)
        cls_token_emb = self.transformer(out).last_hidden_state[:, 0, :]
        return cls_token_emb.squeeze(0)