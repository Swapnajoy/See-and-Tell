from torch.utils.data import Dataset
import json
from transformers import T5Tokenizer

class TripletDataset(Dataset):
    def __init__(
            self,
            triplet_path='data/triplets/triplets.jsonl',
            tokenizer_config_file='t5-base',
            max_len=128,
    ):
        with open(triplet_path, 'r', encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_config_file)
        self.max_len=max_len

    def __getitem__(self, idx):
        image_path = self.entries[idx]['image_path']
        query = self.entries[idx]['query']
        caption = self.entries[idx]['caption']

        encoding = self.tokenizer(
            caption,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        caption_emb = encoding.input_ids.squeeze(0)
        caption_mask = encoding.attention_mask.squeeze(0)

        return {
            'image_path': image_path,
            'query': query,
            'target_ids': caption_emb,
            'target_mask': caption_mask
        }
    
    def __len__(self):
        return len(self.entries)