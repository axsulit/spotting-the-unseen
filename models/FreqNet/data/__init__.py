import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import ConcatDataset
from .datasets import dataset_folder


import os


def get_dataset(opt):
    """
    Load dataset from the specified root directory containing 'real' and 'fake' subfolders.

    Args:
        opt: An object containing dataset options, including 'dataroot'.

    Returns:
        A dataset object created from the specified root.

    Raises:
        FileNotFoundError: If the expected dataset folders do not exist.
    """
    
    dataset_root = opt.dataroot  # Use exactly what is passed

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")

    real_path = os.path.join(dataset_root, 'real')
    fake_path = os.path.join(dataset_root, 'fake')

    print(f"Final dataset root: {dataset_root}")
    # print(f"init:Checking if '{real_path}' exists: {os.path.exists(real_path)}")
    # print(f"init:Checking if '{fake_path}' exists: {os.path.exists(fake_path)}")

    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(f"Expected 'real/' and 'fake/' inside {dataset_root}, but they are missing!")

    return dataset_folder(opt, dataset_root)  # Pass only the dataset root, not subfolders

def get_bal_sampler(dataset):
    """
    Create a weighted sampler to balance class distribution during training.

    Args:
        dataset: A ConcatDataset object with multiple datasets having 'targets'.

    Returns:
        WeightedRandomSampler to balance classes during sampling.
    """
    
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    """
    Create a PyTorch DataLoader with optional class balancing and multi-threading.

    Args:
        opt: An object containing options such as batch size, number of threads, etc.

    Returns:
        DataLoader object for training or evaluation.
    """
    
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
