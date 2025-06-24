import torch
import torch.nn as nn
from tqdm import tqdm

from datasets.triplet_dataset import TripletDataset
from torch.utils.data import DataLoader, random_split
from vlmrag.vlmrag_model import VLMRAG

learning_rate = 0.0003
batch_size = 32
epochs = 50
log_and_eval_freq = epochs//epochs

dataset = TripletDataset()

split_idx = int(0.9*len(dataset))

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VLMRAG(mode='train').to(device)

steps_per_batch = len(train_loader)
max_steps = epochs*steps_per_batch

optimizer = torch.optim.AdamW(
    list(model.retriever.project.parameters()) + list(model.decoder.projection.parameters()),
    lr=learning_rate,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_steps)

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

    if (epoch+1)%log_and_eval_freq == 0:
        model.eval()
        with torch.no_grad():
            running_loss = 0
            for item in test_loader:
                image_path = item['image_path']
                query = item['query']
                target_ids = item['target_ids'].to(device)
                target_mask = item['target_mask'].to(device)

                loss = model(image_path, query, target_ids, target_mask)
                running_loss += loss.item()

            validation_loss = running_loss/steps_per_batch
        
        print(f"Epoch {epoch+1}: Training Loss: {training_loss:.4f} Validation Loss: {validation_loss:.4f}")

print("Training Completed.")
