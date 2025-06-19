import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class Decoder(nn.Module):
    def __init__(self, config_file='t5-base', device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.projection = nn.Linear(1920, 768).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(config_file)
        self.decoder = T5ForConditionalGeneration.from_pretrained(config_file).to(device)

    def forward(self, describe_this: str, encoder_outputs: torch.Tensor) -> str:

        input_ids = self.tokenizer(describe_this, return_tensors='pt').input_ids.to(self.device)
        
        seq_len = 10
        encoder_outputs = self.projection(encoder_outputs).unsqueeze(0).repeat(seq_len, 1)
        encoder_outputs = encoder_outputs.unsqueeze(0)

        output_ids = self.decoder.generate(input_ids=input_ids, encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs))
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)