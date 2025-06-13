"""Evaluation module for the multiple attention model.

This module provides functions for evaluating model performance on test datasets,
including accuracy, AUC, and other metrics at both frame and video levels.
"""

from datasets.data import CustomDeepfakeDataset  # Import your custom dataset class
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score as AUC
import numpy as np
import torch
import json
import os
import pickle
from copy import deepcopy
from torchvision import transforms
from models.MAT import MAT

# Define transformations (resize and normalization if needed)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image channels
])

# Correctly specify the root path for your dataset
dataset_root = 'D:/Thesis/datasets/final/01_ffc23_final_unaltered'

def load_model(name):
    """Load model and configuration from saved checkpoint.
    
    Args:
        name (str): Name of the experiment to load.
        
    Returns:
        tuple: (config, net) containing the loaded configuration and model.
    """
    with open(f'runs/{name}/config.pkl', 'rb') as f:
        config = pickle.load(f)
    net = MAT(**config.net_config)
    return config, net

def acc_eval(labels, preds):
    """Calculate accuracy with a fixed threshold.
    
    Args:
        labels (numpy.ndarray): Ground truth labels.
        preds (numpy.ndarray): Model predictions.
        
    Returns:
        tuple: (threshold, accuracy) where threshold is fixed at 0.5.
    """
    labels = np.array(labels)
    preds = np.array(preds)
    thres = 0.5
    acc = np.mean((preds >= thres) == labels)
    return thres, acc

def test_eval(net, setting, test_loader):
    """Evaluate model on test dataset.
    
    Args:
        net (nn.Module): The model to evaluate.
        setting (dict): Dataset settings.
        test_loader (DataLoader): DataLoader for test dataset.
        
    Returns:
        list: List of predictions for each batch.
    """
    net.eval()
    testset = []
    for i, (X, y) in enumerate(test_loader):
        X = X.to('cuda', non_blocking=True)
        with torch.no_grad():
            logits = net(X)
            pred = torch.nn.functional.softmax(logits, dim=1)[:, 1]  # Assume binary classification (real or fake)
            testset.append(pred.cpu().numpy())  # Append predictions to testset

    # You can now calculate the metrics, e.g., accuracy, AUC, etc., based on `testset`
    return testset

def test_metric(testset):
    """Calculate evaluation metrics for both frame and video levels.
    
    Args:
        testset (list): List of test results containing frame and video predictions.
        
    Returns:
        dict: Dictionary containing accuracy, threshold, and AUC metrics for both
              frame and video levels.
    """
    frame_labels = []
    frame_preds = []
    video_labels = []
    video_preds = []
    for i in testset:
        frame_preds += i[2]
        frame_labels += [i[1]] * len(i[2])
        video_preds.append(i[3])
        video_labels.append(i[1])
    video_thres, video_acc = acc_eval(video_labels, video_preds)
    frame_thres, frame_acc = acc_eval(frame_labels, frame_preds)
    video_auc = AUC(video_labels, video_preds)
    frame_auc = AUC(frame_labels, frame_preds)
    rs = {'video_acc': video_acc, 'video_threshold': video_thres, 'video_auc': video_auc, 
          'frame_acc': frame_acc, 'frame_threshold': frame_thres, 'frame_auc': frame_auc}
    return rs

def all_eval(name, ckpt=None, test_sets=['ff-all', 'celeb', 'deeper']):
    """Run evaluation on multiple test sets.
    
    Args:
        name (str): Name of the experiment to evaluate.
        ckpt (int, optional): Checkpoint number to evaluate. Defaults to None.
        test_sets (list, optional): List of test sets to evaluate on. 
            Defaults to ['ff-all', 'celeb', 'deeper'].
    """
    config, net = load_model(name)
    setting = config.val_dataset
    codec = setting['datalabel'].split('-')[2]
    
    # Use CustomDeepfakeDataset instead of FF_dataset
    if 'ff-all' in test_sets:
        testset = CustomDeepfakeDataset(root_dir=dataset_root, phase='test', transform=train_transforms)
        test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)
        test_eval(net, setting, test_loader)
