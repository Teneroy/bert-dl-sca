import os.path
import sys
import numpy as np
import random
from tqdm import tqdm
import os
import scandir
import traces
import datetime
from glob import glob
from glob import iglob
import pandas as pd
import csv
import argparse


AES_Sbox = np.array([0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
                     0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
                     0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
                     0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
                     0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
                     0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
                     0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
                     0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
                     0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
                     0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
                     0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
                     0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
                     0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
                     0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
                     0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
                     0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16])
                     
# Our labelization function:
# It is as simple as the computation of the result of Sbox(p[2] + k[2]) (see the White Paper)
# Note: you can of course adapt the labelization here (say if you want to attack the first byte Sbox(p[0] + k[0])
# or if you want to attack another round of the algorithm).
# first round of AES and third SBox substituion and corresponding 3rd byte of plain text and key.
def labelize(plaintexts, keys):
    # print("plaintext third byte",plaintexts)
    # print("Key third byte",keys)
    # print(int(plaintexts,16)^int(keys,16))
    l=AES_Sbox[int(plaintexts,16) ^ int(keys,16)]
    # print(l)
    return l


def extract_traces(traces_file, labeled_traces_file, target_points,  number=100000, profiling_desync=0):
    print("Begin extraction")
    print("Number of samples: ", number)
    #"DPA_contest2_public_base_diff_vcc_a128_2009_12_23"
    #Extract the metadata in detail here cipher text will not be taken for extraction purposed as it is useless
    obj=os.scandir(traces_file)
    #raw_filename=read_all(traces_file) 
    sample_number=0
    # Extract a larger set of points to handle desynchronization
    min_target_point = min(target_points)
    max_target_point = max(target_points)+2
      
    with open(labeled_traces_file, 'w',newline='') as file1:
      writer = csv.writer(file1,lineterminator='\n')
      writer.writerow(['Trace', 'Label', 'SampleNumber','PlainText','Key','CipherText'])
      i = 1
      for entry in obj:
        if entry.is_dir() or entry.is_file() and i < number:
            print(i)
            i+=1
            raw_filename=entry.name
            # print(raw_filename)
            full_path = os.path.join(traces_file, entry)
            # print(entry.is_file())

            raw_traces = pd.read_csv(entry.path, encoding="ISO-8859-1",skiprows=24,header=None,names=["Amplitude(Volts)"])
            
            # print(raw_traces)
            raw_sample0=entry.name.split('_n=')[1]
            raw_sample=raw_sample0.split('_k=')
            sample_number=raw_sample[0]
            print("n=",sample_number)
            raw_key0=raw_sample[1].split('_m=')
            raw_keys=raw_key0[0]
            # print("k=",raw_keys)
            pt=raw_key0[1].split('_c=')
            raw_plaintexts=pt[0]
            # print("m=",raw_plaintexts)
            raw_cipher=pt[1].split('.csv')[0]
            # print("c=",raw_cipher)    
            # Compute our labels using plain text and key only 
            labels = labelize(raw_plaintexts[4:6], raw_keys[4:6])
            last_round_samples0=raw_traces[min_target_point:max_target_point]
            last_round_samples=last_round_samples0["Amplitude(Volts)"].to_numpy()
            #print(last_round_samples)
            row=[last_round_samples,int(labels),int(sample_number), raw_plaintexts,raw_keys,raw_cipher]
            # print(row)
            writer.writerow(row)
        else:
            sys.exit(-1)


def extract_attack_traces(traces_file, labeled_traces_file, target_points, number=100000, profiling_desync=0):
    print("Begin extraction")
    #"DPA_contest2_public_base_diff_vcc_a128_2009_12_23"
    #Extract the metadata in detail here cipher text will not be taken for extraction purposed as it is useless
    obj=os.scandir(traces_file)
    #raw_filename=read_all(traces_file) 

    # Extract a larger set of points to handle desynchronization
    min_target_point = min(target_points)
    max_target_point = max(target_points)+2
      
    with open(labeled_traces_file, 'w') as file1:
        writer = csv.writer(file1,lineterminator='\n')
        writer.writerow(['Trace','SampleNumber','PlainText','Key','CipherText'])
        i = 0
        for entry in obj:
            if entry.is_dir() or entry.is_file() and i < number:
                print(i)
                i += 1
                raw_filename=entry.name
                # print(raw_filename)
                full_path = os.path.join(traces_file, entry)
                #print(entry.is_file())
                raw_traces = pd.read_csv(entry.path, encoding="ISO-8859-1",skiprows=24,header=None,names=["Amplitude(Volts)"])
                #raw_traces.plot()
                #plt.show()
                # print(raw_traces)
                raw_sample0=entry.name.split('_n=')[1]
                raw_sample=raw_sample0.split('_k=')
                sample_number=raw_sample[0]
                # print("n=",sample_number)
                raw_key0=raw_sample[1].split('_m=')
                raw_keys=raw_key0[0]
                # print("k=",raw_keys)
                pt=raw_key0[1].split('_c=')
                raw_plaintexts=pt[0]
                # print("m=",raw_plaintexts)
                raw_cipher=pt[1].split('.csv')[0]
                # print("c=",raw_cipher)    
                # Compute our labels using plain text and key only 
                #labels = labelize(raw_plaintexts[4:6], raw_keys[4:6])
                last_round_samples0=raw_traces[min_target_point:max_target_point]
                last_round_samples=last_round_samples0["Amplitude(Volts)"].to_numpy()
                row=[last_round_samples,sample_number, raw_plaintexts,raw_keys,raw_cipher]
                # print(row)
                writer.writerow(row)
            else:
                sys.exit(-1)


parser = argparse.ArgumentParser(description='Command-line argument parser')
parser.add_argument('-r', '--raw', type=str, help='Original raw traces file')
parser.add_argument('-t', '--target', type=str, help='Target file')
parser.add_argument('-a', '--attack', type=str, help='Is attack dataset or profiling dataset(true, false)')
parser.add_argument('-n', '--number', type=int, help='Number of traces')

args = parser.parse_args()

original_raw_traces_file = args.raw
dpa_target_file = args.target
is_attack = args.attack
number = args.number

# dpa_databases_folder = "/data/acp19tsr/DPAContestV2"
# original_raw_traces_file =dpa_databases_folder+"/DPA_contest2_template_base_diff_vcc_a128_2009_12_23"
# dpa_target="/data/acp19tsr/DPAContestV2/"
target_points=[n for n in range(2300, 2700)] 
profiling_desync=0
# dpa_target+"DPAContestV2Training1.csv"

if is_attack.startswith("true"):
   print("Processing attack traces")
   extract_attack_traces(original_raw_traces_file, dpa_target_file, target_points, number, profiling_desync=0)
   sys.exit(0)

print("Processing test traces")
extract_traces(original_raw_traces_file, dpa_target_file,target_points, number, profiling_desync=0)