# Import the required libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from rastogi_functions import *
from bert_transformer import *
import wandb 
import argparse
from tqdm import tqdm
from loss_functions import FocalLoss
import random
from collections import Counter
from torch.optim import swa_utils
import torch.optim as optim


# Set random seed for Python's random module
random_seed = 423
random.seed(random_seed)

# Set random seed for NumPy 
np.random.seed(random_seed)

# Set random seed for PyTorch
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


parser = argparse.ArgumentParser(description='Command-line argument parser')
parser.add_argument('-d', '--data', type=str, help='DPA database profile file')
parser.add_argument('-n', '--name', type=str)
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('-f', '--file_model', type=str)
parser.add_argument('-q', '--quantity', type=int)
    
args = parser.parse_args()

dpa_database = args.data
run_name = args.name
num_epochs = args.epochs
file_model = args.file_model
data_size = args.quantity

print("Loading DPA data...")
(X_train, Y_train) = load_dpa(dpa_database, nrows=data_size)
print("Data is loaded")

class_counts = Counter(Y_train)
class_counts = sorted(class_counts.items(), key=lambda x: x[0])
for class_label, count in class_counts:
    print(f"Class {class_label}: {count} samples")

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_seed)

# Tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

print("Creating DataLoader for train dataset...")
# Create DataLoader for training and validation sets
train_dataset = CustomDataset(X_train, Y_train, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=45, shuffle=True)
print("DataLoader is created")

print("Creating DataLoader for validation dataset...")
val_dataset = CustomDataset(X_val, Y_val, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=45)
print("DataLoader is created")

# Initialize the BERT model
num_classes = 256  # Number of classes (AES key bytes)
bert_model = BERTModel(num_classes)

# remove later
learning_rate = 2e-8

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="bert-thesis-sca-train-ranked",
    name = run_name,
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "Encoder-only Transformer",
    "dataset": "DPAv2",
    "epochs": num_epochs,
    }
)

print(bert_model.state_dict().keys())
print(bert_model.state_dict()['fc.weight'][0])
wandb.log({'bert.encoder.layer.11.attention.output.dense.weight': wandb.Histogram(bert_model.state_dict()['bert.encoder.layer.11.attention.output.dense.weight'][0])})
wandb.log({'fc.weight': wandb.Histogram(bert_model.state_dict()['fc.weight'][0])})

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

print("DEVICE:")
print(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(bert_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# Training loop
warmup_proportion = 0.2  # Warmup rate

# Create a learning rate scheduler with linear warmup
total_steps = len(train_loader) * num_epochs
warmup_steps = int(total_steps * warmup_proportion)
print("WARMUP STEPS:", str(warmup_steps))
# warmup_steps = 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


table = wandb.Table(data=class_counts, columns = ["class", "samples"])
wandb.log({"my_lineplot_id" : wandb.plot.line(table, "class", 
            "samples", stroke=None, title="Sample distribution")})


print("Starting training process...")
for epoch in range(num_epochs):
    bert_model.train()
    train_loss = 0.0
    print(f'Training epoch number: {epoch}')
    batch_number = 1

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = bert_model(input_ids, attention_mask)
        print(outputs)
        print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate with linear warmup

        train_loss += loss.item()
        
        batch_number += 1
    
    
    bert_layer_w = bert_model.state_dict()['bert.encoder.layer.11.attention.output.dense.weight'][0]
    fc_layer_w = bert_model.state_dict()['fc.weight'][0]
    wandb.log({'epoch: ' + str(epoch) + ', bert.encoder.layer.11.attention.output.dense.weight': wandb.Histogram(torch.Tensor.cpu(bert_layer_w))})
    wandb.log({'epoch: ' + str(epoch) + ', fc.weight': wandb.Histogram(torch.Tensor.cpu(fc_layer_w))})

    wandb.log({"train_loss": train_loss/len(train_loader)})

    # Validation
    bert_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = bert_model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    wandb.log({"val_loss": val_loss/len(val_loader)})

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")


# Save the trained BERT model
torch.save(bert_model.state_dict(), file_model)
wandb.save(file_model)
