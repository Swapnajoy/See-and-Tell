import os
import yaml
import torch
import torch.nn as nn
import math
from tqdm import tqdm

from datasets.processed_dataset import ProcessedDataset
from torch.utils.data import DataLoader, random_split
from vlmrag.vlmrag_model import VLMRAG
from utils.config import load_config
from datetime import datetime

cfg = load_config()

def create_exp_folder(config, base_dir="train_exp"):
    dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"run_epoch{config['epochs']}_lr{config['lr']}_bs{config['batch_size']}_{dt_string}"
    exp_path = os.path.join(base_dir, folder_name)
    os.makedirs(exp_path, exist_ok=True)

    with open(os.path.join(exp_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    train_log_path = os.path.join(exp_path, 'train_log.txt')
    val_log_path = os.path.join(exp_path, 'val_log.txt')
    return exp_path, train_log_path, val_log_path

learning_rate = cfg['lr']
batch_size = cfg['batch_size']
epochs = cfg['epochs']
warmup_epochs = cfg['warmup_epochs']
max_retr_lambda = cfg['max_retr_lambda']
min_retr_lambda = cfg['min_retr_lambda']
weight_decay = cfg['weight_decay']
log_freq = cfg['log_frequency']
eval_freq = cfg['eval_frequency']
train_test_split = cfg['train_test_split']

dataset = ProcessedDataset()

split_idx = int(train_test_split*len(dataset))

train_size = split_idx
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = cfg['device']

model = VLMRAG(mode='train').to(device)

num_train_batches = len(train_loader)
num_val_batches = len(test_loader)
max_steps = epochs*num_train_batches

trainable_params = [
    {"params": model.retriever.project.parameters()},
    {"params": model.decoder.projection.parameters()},
    {"params": [p for n, p in model.decoder.named_parameters() if p.requires_grad]}
]

trainable_params = [p for p in model.parameters() if p.requires_grad]
print("Sanity check trainable parameters:", sum(p.numel() for p in trainable_params))

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=learning_rate,
    weight_decay=1e-4
)

num_params = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
print("Total trainable parameters in optimizer:", num_params)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_steps)

exp_path, train_log_path, val_log_path = create_exp_folder(config=cfg)

for epoch in range(epochs):
    model.train()
    running_loss = 0
    decoder_loss = 0
    retriever_loss = 0

    if epoch <= warmup_epochs:
        retr_lambda = max_retr_lambda * epoch/warmup_epochs
        
    else:
        decay_epochs = epochs - warmup_epochs
        decay_progress = (epoch - warmup_epochs) / decay_epochs
        cosine_decay = math.cos(0.5 * math.pi * decay_progress)
        retr_lambda = min_retr_lambda + (max_retr_lambda - min_retr_lambda) * cosine_decay

    for item in tqdm(train_loader, desc=f'{epoch+1}'):
        image_path = item['image_path']
        query = item['query']
        target_ids = item['target_ids'].to(device)
        target_mask = item['target_mask'].to(device)
        gt_emb = item['gt_emb'].to(device)

        optimizer.zero_grad()

        dec_loss, retr_loss = model(image_path, query, gt_emb, target_ids, target_mask)
        loss = dec_loss + retr_lambda*retr_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        decoder_loss += dec_loss.item()
        retriever_loss += retr_loss.item()
    
    training_loss = running_loss/num_train_batches
    decoder_loss = decoder_loss/num_train_batches
    retriever_loss = retriever_loss/num_train_batches

    print(f"Epoch {epoch+1}: Training Loss: {training_loss:.4f}, Decoder_loss: {decoder_loss:.4f}, Retriever_loss: {retriever_loss:.4f}")
    log_line = f"Epoch {epoch+1} | Training Loss: {training_loss:.4f} | Decoder_loss: {decoder_loss:.4f} | Retriever_loss: {retriever_loss:.4f}"

    with open(train_log_path, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')

    if (epoch+1)%eval_freq == 0:
        model.eval()
        validation_loss = 0
        decoder_loss = 0
        retriever_loss = 0
        with torch.no_grad():
            running_loss = 0
            for item in test_loader:
                image_path = item['image_path']
                query = item['query']
                target_ids = item['target_ids'].to(device)
                target_mask = item['target_mask'].to(device)
                gt_emb = item['gt_emb'].to(device)

                dec_loss, retr_loss = model(image_path, query, gt_emb, target_ids, target_mask)
                loss = dec_loss + retr_lambda*retr_loss
                running_loss += loss.item()
                decoder_loss += dec_loss.item()
                retriever_loss += retr_loss.item()

            validation_loss = running_loss/num_val_batches
            decoder_loss = decoder_loss/num_val_batches
            retriever_loss = retriever_loss/num_val_batches
            print(f"Epoch {epoch+1}: Validation Loss: {validation_loss:.4f}, Decoder_loss: {decoder_loss:.4f}, Retriever_loss: {retriever_loss:.4f}")
        
    if (epoch+1)%log_freq == 0:
        log_line = f"Epoch {epoch+1} | Validation Loss: {validation_loss:.4f} | Decoder_loss: {decoder_loss:.4f} | Retriever_loss: {retriever_loss:.4f}"
        ckpt_path = os.path.join(exp_path, f'epoch_{epoch+1}.pt')


        with open(val_log_path, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')
       
        torch.save({
            'retriever_proj': model.retriever.project.state_dict(),
            'decoder': model.decoder.state_dict(),
        }, ckpt_path)

print("Training Completed.")
