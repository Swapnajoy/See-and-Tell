import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class Decoder(nn.Module):
    def __init__(self, config_file='t5-base', device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.projection = nn.Sequential(
            nn.Linear(2688, 1536),
            nn.ReLU(inplace=True),
            nn.Linear(1536, 768),
        ).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(config_file)
        self.decoder = T5ForConditionalGeneration.from_pretrained(config_file).to(device)

    def forward(
            self,
            encoder_outputs,
            mode="inference",
            target_ids=None,
            target_mask=None,
        ):

        if mode == 'inference':
            input_ids = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.device)
        
            encoder_outputs = self.projection(encoder_outputs).unsqueeze(1)

            output_ids = self.decoder.generate(
                input_ids=input_ids,
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs),
                max_new_tokens=128,
                temperature=1.0,
                top_k=50,
                do_sample=True
            )

            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        else:
            assert target_ids is not None
            
            encoder_outputs = self.projection(encoder_outputs).unsqueeze(1)
            output = self.decoder(
                input_ids=target_ids,
                attention_mask=target_mask,
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs),
                labels=target_ids
            )
            return output.loss
