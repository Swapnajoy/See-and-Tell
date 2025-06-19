import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class Decoder(nn.Module):
    def __init__(self, config_file='t5-base'):
        super().__init__()
        self.projection = nn.Linear(1920, 768)
        self.tokenizer = T5Tokenizer.from_pretrained(config_file)
        self.decoder = T5ForConditionalGeneration.from_pretrained(config_file)

    def forward(self, describe_this: str, encoder_outputs: torch.Tensor) -> str:
        input_ids = self.tokenizer(describe_this, return_tensors='pt').input_ids.to(self.decoder.device)
        encoder_outputs = self.projection(encoder_outputs).unsqueeze(1).to(self.decoder.device)

        output_ids = self.decoder.generate(input_ids=input_ids, encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs))
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)