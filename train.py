import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets.triplet_dataset import TripletDataset
from torch.utils.data import DataLoader, random_split
from vlmrag.vlmrag_model import VLMRAG
from utils.config import load_config

cfg = load_config()

from datetime import datetime
import os
import yaml

def create_exp_folder(config, base_dir="train_exp"):
    dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"run_epoch{config['epochs']}_lr{config['lr']}_bs{config['batch_size']}_time{dt_string}"
    exp_path = os.path.join(base_dir, folder_name)
    os.makedirs(exp_path, exist_ok=True)

    with open(os.path.join(exp_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    log_path = os.path.join(exp_path, 'log.txt')
    return exp_path, log_path

learning_rate = cfg['lr']
batch_size = cfg['batch_size']
epochs = cfg['epochs']
weight_decay = cfg['weight_decay']
log_freq = cfg['log_frequency']
eval_freq = cfg['eval_frequency']
train_test_split = cfg['train_test_split']

dataset = TripletDataset()

split_idx = int(train_test_split*len(dataset))

train_size = split_idx
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = cfg['device']

model = VLMRAG(mode='train').to(device)

steps_per_batch = len(train_loader)
max_steps = epochs*steps_per_batch

optimizer = torch.optim.AdamW(
    list(model.retriever.project.parameters()) + list(model.decoder.projection.parameters()),
    lr=learning_rate,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_steps)

exp_path, log_path = create_exp_folder(config=cfg)

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for item in tqdm(train_loader, desc=f'{epoch+1}'):
        image_path = item['image_path']
        query = item['query']
        target_ids = item['target_ids'].to(device)
        target_mask = item['target_mask'].to(device)

        optimizer.zero_grad()

        loss = model(image_path, query, target_ids, target_mask)
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
    
    training_loss = running_loss/steps_per_batch
    print(f"Epoch {epoch+1}: Training Loss: {training_loss:.4f}")

    if (epoch+1)%eval_freq == 0:
        model.eval()
        validation_loss = None
        with torch.no_grad():
            running_loss = 0
            for item in tqdm(test_loader, desc=f"Eval {epoch+1}"):
                image_path = item['image_path']
                query = item['query']
                target_ids = item['target_ids'].to(device)
                target_mask = item['target_mask'].to(device)

                loss = model(image_path, query, target_ids, target_mask)
                running_loss += loss.item()

            validation_loss = running_loss/steps_per_batch
            print(f"Epoch {epoch+1}: Validation Loss: {validation_loss:.4f}")
        
    if (epoch+1)%log_freq == 0:
        log_line = f"Epoch {epoch+1} | Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f}"
        ckpt_path = os.path.join(exp_path, f'epoch_{epoch+1}.pt')

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)
        
        torch.save(model.state_dict(), ckpt_path)
print("Training Completed.")
