import argparse
import copy
import json
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import classification_report, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import transformers
from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

def get_timestamp(x):
    timestamp = []
    for t in x:
        timestamp.append(datetime.timestamp(t))

    np.array(timestamp) - timestamp[-1]
    return timestamp

class SuicidalDataset(Dataset):
    def __init__(self, label, temporal, timestamp, tf_idf_vector):

        super().__init__()
        self.label = label
        self.temporal = temporal
        self.timestamp = timestamp
        self.tf_idf_vector = tf_idf_vector

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):

        labels = torch.tensor(self.label[item])

        result = self.temporal[item]
        temporal_features = torch.tensor(result)

        tf_idf_vector = self.tf_idf_vector[item]

        timestamp = torch.tensor(get_timestamp(self.timestamp[item]))

        return [labels, temporal_features, timestamp, tf_idf_vector]

def pad_ts_collate(batch):

    target = [item[0] for item in batch]
    data_post = [item[1] for item in batch]
    timestamp_post = [item[2] for item in batch]
    tf_idf_vector = [item[3] for item in batch]

    lens_post = [len(x) for x in data_post]

    data_post = nn.utils.rnn.pad_sequence(data_post, batch_first=True, padding_value=0)

    timestamp_post = nn.utils.rnn.pad_sequence(timestamp_post, batch_first=True, padding_value=0)

    target = torch.tensor(target)
    lens_post = torch.tensor(lens_post)

    tf_idf_vector = torch.tensor(tf_idf_vector).float()

    return [target, data_post, timestamp_post, lens_post, tf_idf_vector]