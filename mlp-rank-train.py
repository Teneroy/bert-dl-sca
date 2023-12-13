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
from mlp_power_trace import *
import wandb 
import argparse
from tqdm import tqdm
from loss_functions import FocalLoss
import random
from collections import Counter
from torch.optim import swa_utils
import torch.optim as optim

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
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

print("Creating DataLoader for train dataset...")
# Create DataLoader for training and validation sets
train_dataset = CustomDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=45, shuffle=True)
print("DataLoader is created")

print("Creating DataLoader for validation dataset...")
val_dataset = CustomDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=45)
print("DataLoader is created")

# Initialize the BERT model
mlp_model = PowerTraceMLP()

# remove later
learning_rate = 2e-8
# num_epochs = 3
# 

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="mlp-thesis-sca-train-ranked",
    name = run_name,
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "MLP",
    "dataset": "DPAv2",
    "epochs": num_epochs,
    }
)

# for key in bert_model.state_dict().keys():
#     if('weight' in key):
#         print(key)
#         wandb.log({key: wandb.Histogram(bert_model.state_dict()[key][0])})

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model.to(device)

print("DEVICE:")
print(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

table = wandb.Table(data=class_counts, columns = ["class", "samples"])
wandb.log({"my_lineplot_id" : wandb.plot.line(table, "class", 
            "samples", stroke=None, title="Sample distribution")})

print("Starting training process...")
for epoch in range(num_epochs):
    mlp_model.train()
    train_loss = 0.0
    print(f'Training epoch number: {epoch}')
    batch_number = 1

    # Update SWA model if epoch 0
    # if epoch == 0:
    #     print("SWA")
    #     swa_model.update_parameters(bert_model)
    #     swa_scheduler.step()  

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        traces = batch['traces'].to(device)
        labels = batch['labels'].to(device)
        # bert_model.zero_grad()   
        optimizer.zero_grad()

        outputs = mlp_model(traces)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()  # Update the learning rate with linear warmup

        train_loss += loss.item()
        
        batch_number += 1
    wandb.log({"train_loss": train_loss/len(train_loader)})

    # Validation
    mlp_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            traces = batch['traces'].to(device)
            labels = batch['labels'].to(device)

            outputs = mlp_model(traces)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    wandb.log({"val_loss": val_loss/len(val_loader)})

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

# Rank and full_ranks functions
# (Modify the rank and full_ranks functions to work with BERT model's predictions and outputs)

# Save the trained BERT model
torch.save(mlp_model.state_dict(), file_model)
wandb.save(file_model)
# trained_bert_1k_seed1000_base_e200_lr28_bias09_0999
# trained_bert_1k_seed1000_base_e60_lr28_bias09_0999_warmup01_SWAe28
# Now you can use the trained BERT model for side-channel analysis and key recovery.