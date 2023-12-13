import os
import numpy as np
import pandas as pd
import sys
from sca_constants import AES_Sbox
from torch.utils.data import DataLoader
from bert_transformer import CustomDataset
from transformers import BertTokenizer
from tqdm import tqdm
import torch


def check_file_exists(file_path):
	file_path = os.path.normpath(file_path)
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)


def load_dpa(dpa_database_file, nrows=100000):    
    check_file_exists(dpa_database_file)

    df=pd.read_csv(dpa_database_file,nrows=nrows) 
    df.Trace = df.Trace.replace('\s+', ',', regex=True) 
    df.Trace=df.Trace.apply(lambda x: x.replace('[,',''))
    df.Trace=df.Trace.apply(lambda x: x.replace('[',''))
    df.Trace=df.Trace.apply(lambda x: x.replace(']',''))
    df["Trace"]=df["Trace"].apply(lambda x: x.split(','))

    df["Trace"]=df["Trace"].apply(lambda x: np.array(x).astype(np.int16)) 

    X_profiling=np.stack(df["Trace"].to_numpy())[:,0:400]

    Y_profiling=dfL=df["Label"]
    
    return (X_profiling,Y_profiling)


def load_dpa_attack(dpa_database_file, load_metadata=False):
    # To check whether the file exists or not if not exit with error
    file_path = os.path.normpath(dpa_database_file)
    if os.path.exists(dpa_database_file) == False:
        print("Error: provided file path '%s' does not exist!" % dpa_database_file)
        sys.exit(-1)
    #If the file exists read it 
    df=pd.read_csv(dpa_database_file,nrows=20000)
    # df = df.sample(frac=1)
    df.Trace = df.Trace.replace('\s+', ',', regex=True)
    df.Trace=df.Trace.apply(lambda x: x.replace('[,',''))
    df.Trace=df.Trace.apply(lambda x: x.replace('[',''))
    df.Trace=df.Trace.apply(lambda x: x.replace(']',''))
    df["Trace"]=df["Trace"].apply(lambda x: x.split(','))
    df["Trace"]=df["Trace"].apply(lambda x: np.array(x).astype(np.float64)) 
    X_profiling=np.stack(df["Trace"].to_numpy())[:,0:400]
    if load_metadata == False:
        return (X_profiling)
    else:
        return (X_profiling), (df['PlainText'], df['Key'])
    

def rank(predictions, metadata_plaintext,metadata_key, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba):
    # Compute the rank
    print("Calling rank function!")

    predictions = predictions - torch.min(predictions)

    if len(last_key_bytes_proba) == 0:
        key_bytes_proba = np.zeros(256)
    else:
        # This is not the first rank we compute: we optimize things by using the previous computations to save time!
        key_bytes_proba = last_key_bytes_proba

    for p in range(0, max_trace_idx-min_trace_idx):
        # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
        plaintext = metadata_plaintext[min_trace_idx + p][4:6]
        for i in range(0, 256):
            # Our candidate key byte probability is the sum of the predictions logs
            proba = predictions[p][AES_Sbox[int(plaintext,16) ^ i]].item()

            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                # We do not want an -inf here, put a very small epsilon that correspondis to a power of our min non zero proba
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions).item()
                key_bytes_proba[i] += np.log(min_proba**2)
    # Now we find where our real key candidate lies in the estimation.We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return (real_key_rank, key_bytes_proba)


def full_ranks(model, dataset, metadata_plaintext,metadata_key, min_trace_idx, max_trace_idx, rank_step):
    # Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
    real_key = int(metadata_key[0][4:6],16)
    # Check for overflow
    if max_trace_idx > dataset.shape[0]:
        print("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))
        sys.exit(-1)
    input_data = dataset[min_trace_idx:max_trace_idx, :]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Convert test data to BERT format
    test_dataset = CustomDataset(dataset, pd.DataFrame(np.zeros(len(dataset))), tokenizer)  # We only need traces for prediction, labels can be dummy

    # Create DataLoader for the test set
    test_loader = DataLoader(test_dataset, batch_size=50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)

    batch_number = 0
    pred = None

    with torch.no_grad():  # No need to track gradients during inference
        for batch in tqdm(test_loader):
            print("Batch number: ", batch_number)
            batch_number += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            predictions = model(input_ids, attention_mask)

            if pred is None:
                pred = predictions.cpu()  # Move predictions to CPU and keep as the initial tensor
            else:
                pred = torch.cat((pred, predictions.cpu()), dim=0)  # Concatenate along the first dimension
            
            del input_ids, attention_mask, batch, predictions

            torch.cuda.empty_cache()  # Clear cache periodically to manage memory

    # Now 'pred' contains all the predictions
    print("Predictions are",pred,"Size",pred.shape)
    index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t, i in zip(index, range(0, len(index))):
        print("Iteration number: ", i)
        real_key_rank, key_bytes_proba = rank(pred[t-rank_step:t], metadata_plaintext,metadata_key, real_key, t-rank_step, t, key_bytes_proba)
        print("real_key_rank: ", real_key_rank)
        print("t - min_trace_idx: ", t - min_trace_idx)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks