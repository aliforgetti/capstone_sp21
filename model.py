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

class HistoricCurrent(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, subreddit_embedding = 512, dropout = 0.2):
        super().__init__()
        self.historic_model = TimeLSTM(embedding_dim, hidden_dim[0])
        self.dropout = nn.Dropout(dropout) 

        self.fc_tfidf = nn.Linear(799, subreddit_embedding)

        current_dim = hidden_dim[0] + subreddit_embedding #concatenated

        self.layers = nn.ModuleList()
        for hdim in hidden_dim[1:]:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim

        self.layers.append(nn.Linear(current_dim, 2))

    @staticmethod
    def combine_features(features_1, features_2):
        return torch.cat((features_1, features_2), 1)

    def forward(self, historic_features, timestamp, tf_idf_vector):
        outputs = self.historic_model(historic_features, timestamp)
        outputs = torch.mean(outputs, 1)

        subreddit_features = F.relu(self.fc_tfidf(tf_idf_vector))

        combined_features = self.combine_features(subreddit_features, outputs)
        x = self.dropout(combined_features)

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.dropout(x)

        return self.layers[-1](x) # final layer

class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        # assumes that batch_first is always true
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)

        h = h.cuda()
        c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        return outputs