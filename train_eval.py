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

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights, dtype=torch.float32).cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)

    return cb_loss

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

def loss_fn(output, targets, samples_per_cls, loss_type = "focal"):
    beta = 0.9999
    gamma = 2.0
    no_of_classes = 2

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)


def train_loop(model, dataloader, optimizer, device, dataset_len):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for bi, inputs in enumerate(tqdm_notebook(dataloader, total=len(dataloader), leave=False)):
        
        labels, data, timestamp, lens, tf_idf_vector = inputs

        labels = labels.to(device)
        data = data.to(device)
        timestamp = timestamp.to(device)
        tf_idf_vector = tf_idf_vector.to(device)

        optimizer.zero_grad()
        output = model(data, timestamp,tf_idf_vector)
        _, preds = torch.max(output, 1)

        loss = loss_fn(output, labels, labels.unique(return_counts=True)[1].tolist())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / dataset_len

    return epoch_loss, epoch_acc

def eval_loop(model, dataloader, device, dataset_len):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    fin_targets = []
    fin_outputs = []

    for bi, inputs in enumerate(tqdm_notebook(dataloader, total=len(dataloader), leave=False)):

        labels, data, timestamp, lens, tf_idf_vector = inputs

        labels = labels.to(device)
        data = data.to(device)
        timestamp = timestamp.to(device)
        tf_idf_vector = tf_idf_vector.to(device)

        with torch.no_grad():
            output = model(data, timestamp, tf_idf_vector)
        _, preds = torch.max(output, 1)
        
        loss = loss_fn(output, labels, labels.unique(return_counts=True)[1].tolist())
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        fin_targets.append(labels.cpu().detach().numpy())
        fin_outputs.append(preds.cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_corrects.double() / dataset_len

    return epoch_loss, epoch_accuracy, np.hstack(fin_outputs), np.hstack(fin_targets)


### EXAMPLE OF HOW TO RUN THIS
# HIDDEN_DIM = [128,64]
# EMBEDDING_DIM = 768
# SUBREDDIT_EMBEDDING = 128
# DROPOUT = 0.7
# LEARNING_RATE = 0.001

# model = HistoricCurrent(EMBEDDING_DIM, HIDDEN_DIM, SUBREDDIT_EMBEDDING, DROPOUT)
# print(model)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

# model.to(device)

# best_metric = 0.0
# best_model_wts = copy.deepcopy(model.state_dict())

# optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
# print(optimizer)

# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=EPOCHS)
# print(scheduler)