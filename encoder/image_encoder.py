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

    def __call__(self, image_path_list) -> torch.Tensor:
        images = []
        for image_path in image_path_list:
            image = Image.open(image_path).convert('RGB')
            images.append(self.tf(image))
        batch = torch.stack(images).to(self.device)
        cls_token_emb = self.transformer(batch).last_hidden_state[:, 0, :]
        return cls_token_emb
    
if __name__ == '__main__':
    model = ImageEncoder()
    image_path_list = ['data/images/000000000139.jpg', 'data/images/000000005529.jpg', 'data/images/000000010764.jpg']
    print(model(image_path_list).shape)
