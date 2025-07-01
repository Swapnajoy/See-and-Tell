import torch
from transformers import T5EncoderModel, T5Tokenizer

class TextEncoder:
    def __init__(self, config_file='t5-base', max_len=16, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(config_file)
        self.encoder = T5EncoderModel.from_pretrained(config_file).to(device).eval()
        self.max_len = max_len

    def __call__(self, text_list) -> torch.Tensor:
        encoding = self.tokenizer(
            text_list,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            output = self.encoder(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask
            )

        mask = encoding.attention_mask.unsqueeze(-1).expand(output.last_hidden_state.size())
        masked_output = output.last_hidden_state * mask
        pooled = masked_output.sum(dim=1) / mask.sum(dim=1)

        return pooled
    
if __name__ == '__main__':
    model = TextEncoder()
    text_list = ['Describe this image.', 'What can be seen in this image?', 'Explain what is happening here.']
    print(model(text_list).shape)