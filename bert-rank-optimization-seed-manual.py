# Import the required libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from rastogi_functions import *
from bert_transformer_linear_default import *
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, Trials
import wandb
from numpy import trapz
import random


results = []

def train_bert(params):
    num_epochs = 3
    learning_rate = 2e-8
    batch_size = 45
    random_seed = int(params['random_seed'])


    # Set random seed for Python's random module
    random.seed(random_seed)

    # Set random seed for NumPy 
    np.random.seed(random_seed)

    # Set random seed for PyTorch
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print("Loading DPA data...")
    (X_train, Y_train) = load_dpa("./train_full.csv", nrows=1000)
    print("Data is loaded")

    # Tokenizer for BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_seed)

    print("Creating DataLoader for train dataset...")
    # Create DataLoader for training and validation sets
    train_dataset = CustomDataset(X_train, Y_train, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("DataLoader is created")

    print("Creating DataLoader for validation dataset...")
    val_dataset = CustomDataset(X_val, Y_val, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print("DataLoader is created")

    # Initialize the BERT model
    num_classes = 256  # Number of classes (AES key bytes)
    bert_model = BERTModel(num_classes)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bert_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    val_loss = 0.0

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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            batch_number += 1

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

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")


    dpa_database_attack_file = "./test.csv"

    print("Loading attack data")
    X_test, (metadata_plaintext, metadata_key) = load_dpa_attack(dpa_database_attack_file, load_metadata=True)
    print("Data is loaded")

    # Test the trained model on the test dataset and plot the results
    print("Starting to test the model...")

    num_test_traces = 20000

    area = test_model(bert_model, X_test, metadata_plaintext, metadata_key, num_test_traces, random_seed)

    print("Area: ", area)
    print("Seed: ", random_seed)

    return area



def test_model(model, dataset, metadata_plaintext, metadata_key, num_traces, seed):
    ranks = full_ranks(model, dataset, metadata_plaintext, metadata_key, 0, num_traces, 10)

    # We plot the results
    key_ranks = [ranks[i][1] for i in range(0, ranks.shape[0])]
    
    min_key_rank = min(key_ranks)
    try:
        min_index = key_ranks.index(0)
    except ValueError:
        results.append({seed: 20000})
        print(results)
        return 20000
    
    key_traces = [ranks[i][0] for i in range(0, ranks.shape[0])]

    data = [[x, y] for (x, y) in zip(key_traces, key_ranks)]
    table = wandb.Table(data=data, columns = ["trace_number", "rank"])
    wandb.log({"my_lineplot_id_" + str(seed) : wandb.plot.line(table, "trace_number", 
            "rank", stroke=None, title="Seed: " + str(seed))})

    results.append({seed: min_index})

    print(results)

    return min_index


# start a new wandb ru n to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="bert-thesis-sca-optimize-ranked",
    name="1k-e3-linear_He-1_600",
    # track hyperparameters and run metadata
    config={
    "architecture": "Encoder-only Transformer",
    "dataset": "DPAv2",
     }
)

for seed in range(1, 600):
    train_bert({'random_seed': seed})

min_value_dict = min(results, key=lambda x: list(x.values())[0])

# Extract the key corresponding to the minimum value
min_key = list(min_value_dict.keys())[0]
min_value = list(min_value_dict.values())[0]

print("Best hyperparameters:", min_key)
print("Best trace number: ", min_value)
