from mlp_power_trace import *
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from rastogi_functions_mlp import *
import wandb
import argparse



# Test the trained model on the test dataset (DPAContestV2Testing.csv)
def test_model(model, dataset, metadata_plaintext, metadata_key, num_traces):
    ranks = full_ranks(model, dataset, metadata_plaintext, metadata_key, 0, num_traces, 10)

    # We plot the results
    key_traces = [ranks[i][0] for i in range(0, ranks.shape[0])]
    key_ranks = [ranks[i][1] for i in range(0, ranks.shape[0])]

    data = [[x, y] for (x, y) in zip(key_traces, key_ranks)]
    table = wandb.Table(data=data, columns = ["trace_number", "rank"])
    wandb.log({"my_lineplot_id" : wandb.plot.line(table, "trace_number", 
            "rank", stroke=None, title="Performance of trained BERT model against test dataset")})



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE: ", device)
num_classes = 256  # Number of classes (AES key bytes)

# Load the test dataset (DPAContestV2Testing.csv)
# dpa_database_attack_file = dpa_data_folder + "DPAContestV2Testing.csv"

parser = argparse.ArgumentParser(description='Command-line argument parser')
parser.add_argument('-d', '--data', type=str, help='DPA database attack file')
parser.add_argument('-n', '--name', type=str)
parser.add_argument('-f', '--file_model', type=str)

args = parser.parse_args()

dpa_database_attack_file = args.data
run_name = args.name
file_model = args.file_model

# Load the trained BERT model
print("Loading the model...")
trained_model_path = file_model
mlp_model = PowerTraceMLP()  # Assuming num_classes is defined as 256
mlp_model.load_state_dict(torch.load(trained_model_path))
mlp_model.to(device)
mlp_model.eval()
print("Model is loaded")

print("Loading attack data")
X_test, (metadata_plaintext, metadata_key) = load_dpa_attack(dpa_database_attack_file, load_metadata=True)
print("Data is loaded")

# Test the trained model on the test dataset and plot the results
print("Starting to test the model...")

num_test_traces = 20000

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="mlp-thesis-sca-test-ranked",
    name=run_name,
    # track hyperparameters and run metadata
    config={
    "architecture": "MLP",
    "dataset": "DPAv2",
    "num_test_traces": num_test_traces,
    }
)

test_model(mlp_model, X_test, metadata_plaintext, metadata_key, num_test_traces)