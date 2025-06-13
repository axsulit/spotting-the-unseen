"""Training module for the multiple attention model.

This module provides functions for training, validation, and testing the multiple attention model.
It supports both single-GPU and multi-GPU training scenarios, with distributed training capabilities.
"""

import os
import time
import logging
import warnings
import numpy 
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from models.MAT import MAT
from datasets.dataset import DeepfakeDataset
from AGDA import AGDA
import cv2
from utils import dist_average, ACC
#from torch.utils.tensorboard import SummaryWriter
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# GPU settings
assert torch.cuda.is_available()
#torch.autograd.set_detect_anomaly(True)
from datasets.data import CustomDeepfakeDataset  # Import the CustomDeepfakeDataset class from datasets/data.py
from torchvision import transforms

train_transforms = transforms.Compose([
    # transforms.Resize((224, 224)),  # Resize to match your model input size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize image channels
])


def load_state(net, ckpt):
    """Load model state from checkpoint.
    
    Args:
        net (nn.Module): The model to load state into.
        ckpt (dict): Checkpoint dictionary containing model state.
        
    Returns:
        bool: True if all layers were loaded successfully, False otherwise.
    """
    sd = net.state_dict()
    nd = {}
    goodmatch = True
    for i in ckpt:
        if i in sd and sd[i].shape == ckpt[i].shape:
            nd[i] = ckpt[i]
            #print(i)
        else:
            print('fail to load %s' % i)
            goodmatch = False
    net.load_state_dict(nd, strict=False)
    return goodmatch

def main_worker(local_rank, world_size, rank_offset, config):
    """Main worker function for training.
    
    Args:
        local_rank (int): Local rank of the current GPU.
        world_size (int): Total number of GPUs in distributed training.
        rank_offset (int): Offset for the global rank.
        config (train_config): Training configuration object.
    """
    rank = local_rank + rank_offset
    if rank == 0:
        logging.basicConfig(
            filename=os.path.join('runs', config.name, 'train.log'),
            filemode='a',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO
        )
    warnings.filterwarnings("ignore")

    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model here before moving to device
    net = MAT(**config.net_config)
    net.to(device)  # Move model to device (GPU or CPU)

    # AGDA setup (if using it)
    AG = AGDA(**config.AGDA_config).to(device)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, betas=config.adam_betas, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)

    # If resuming from a checkpoint, load the model and optimizer states
    if config.ckpt:
        checkpoint = torch.load(config.ckpt, map_location=device)
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch']) + 1
        # Load model weights from checkpoint
        if load_state(net, checkpoint['state_dict']) and config.resume_optim:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            except KeyError:
                pass
        else:
            net.auxiliary_loss.alpha = torch.tensor(config.alpha)  # Ensure correct value if not loading checkpoint
        del checkpoint
    else:
        start_epoch = 0  # Start from epoch 0 if no checkpoint is provided

    # Clear CUDA cache (useful to avoid memory leaks)
    torch.cuda.empty_cache()

    # Load the custom dataset
    train_dataset = CustomDeepfakeDataset(root_dir=r"D:\.THESIS\datasets\final\01_ffc23_final_unaltered", phase='train', transform=train_transforms)
    val_dataset = CustomDeepfakeDataset(root_dir=r"D:\.THESIS\datasets\final\01_ffc23_final_unaltered", phase='validate', transform=train_transforms)
    test_dataset = CustomDeepfakeDataset(root_dir=r"D:\.THESIS\datasets\final\01_ffc23_final_unaltered", phase='test', transform=train_transforms)

    # Use DataLoader (no distributed sampler for single-GPU)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    validate_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)

    logs = {}
    # Freezing layers (based on config)
    for i in config.freeze:
        if 'backbone' in i:
            net.net.requires_grad_(False)
        elif 'attention' in i:
            net.attentions.requires_grad_(False)
        elif 'feature_center' in i:
            net.auxiliary_loss.alpha = 0
        elif 'texture_enhance' in i:
            net.texture_enhance.requires_grad_(False)
        elif 'fcs' in i:
            net.projection_local.requires_grad_(False)
            net.project_final.requires_grad_(False)
            net.ensemble_classifier_fc.requires_grad_(False)
        else:
            if 'xception' in str(type(net.net)):
                for j in net.net.seq:
                    if j[0] == i:
                        for t in j[1]:
                            t.requires_grad_(False)
            if 'EfficientNet' in str(type(net.net)):
                if i == 'b0':
                    net.net._conv_stem.requires_grad_(False)
                stage_map = net.net.stage_map
                for c in range(len(stage_map)-2, -1, -1):
                    if not stage_map[c]:
                        stage_map[c] = stage_map[c+1]
                for c1, c2 in zip(stage_map, net.net._blocks):
                    if c1 == i:
                        c2.requires_grad_(False)

    # If you're using multiple GPUs, wrap the model in DataParallel
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)  # Use DataParallel for multi-GPU

    # Training loop
    for epoch in range(start_epoch, config.epochs):
        logs['epoch'] = epoch
        run(logs=logs, data_loader=train_loader, net=net, optimizer=optimizer, local_rank=local_rank, config=config, AG=AG, phase='train')
        run(logs=logs, data_loader=validate_loader, net=net, optimizer=optimizer, local_rank=local_rank, config=config, phase='valid')
        
        # Test phase (you may want to call this during the evaluation)
        run(logs=logs, data_loader=test_loader, net=net, optimizer=optimizer, local_rank=local_rank, config=config, phase='test')

        # Update learning rate scheduler
        scheduler.step()

        # Save checkpoints every epoch
        if local_rank == 0:
            torch.save({
                'logs': logs,
                'state_dict': net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict()},
                'checkpoints/' + config.name + '/ckpt_' + str(epoch) + '.pth')


def train_loss(loss_pack, config):
    """Calculate the total training loss.
    
    Args:
        loss_pack (dict): Dictionary containing different loss components.
        config (train_config): Training configuration object.
        
    Returns:
        torch.Tensor: Total loss value.
    """
    if 'loss' in loss_pack:
        return loss_pack['loss']
    loss = config.ensemble_loss_weight * loss_pack['ensemble_loss'] + config.aux_loss_weight * loss_pack['aux_loss']
    if config.AGDA_loss_weight != 0:
        loss += config.AGDA_loss_weight * loss_pack['AGDA_ensemble_loss'] + config.match_loss_weight * loss_pack['match_loss']
    return loss
    
def run(logs, data_loader, net, optimizer, local_rank, config, AG=None, phase='train'):
    """Run a single epoch of training, validation, or testing.
    
    Args:
        logs (dict): Dictionary to store training logs.
        data_loader (DataLoader): DataLoader for the current phase.
        net (nn.Module): The model to train/evaluate.
        optimizer (Optimizer): Optimizer for training.
        local_rank (int): Local rank of the current GPU.
        config (train_config): Training configuration object.
        AG (AGDA, optional): AGDA module if using adversarial training. Defaults to None.
        phase (str, optional): Current phase ('train', 'valid', or 'test'). Defaults to 'train'.
    """
    if local_rank == 0:
        print('start ', phase)
    if config.AGDA_loss_weight == 0:
        AG = None
    recorder = {}
    if config.feature_layer == 'logits':
        record_list = ['loss', 'acc']
    else:
        record_list = ['ensemble_loss', 'aux_loss', 'ensemble_acc']
        if AG is not None:
            record_list += ['AGDA_ensemble_loss', 'match_loss']
    for i in record_list:
        recorder[i] = dist_average(local_rank)
    # begin training
    start_time = time.time()
    if phase == 'train':
        net.train()
    else: 
        net.eval()
    for i, (X, y) in enumerate(data_loader):
        X = X.to(local_rank, non_blocking=True)
        y = y.to(local_rank, non_blocking=True)
        with torch.set_grad_enabled(phase == 'train'):
            loss_pack = net(X, y, train_batch=True, AG=AG)
        if phase == 'train':
            batch_loss = train_loss(loss_pack, config)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            if config.feature_layer == 'logits':
                loss_pack['acc'] = ACC(loss_pack['logits'], y)
            else:
                loss_pack['ensemble_acc'] = ACC(loss_pack['ensemble_logit'], y)
        for i in record_list:
            recorder[i].step(loss_pack[i])

    # end of this epoch
    batch_info = []
    for i in record_list:
        mesg = recorder[i].get()
        logs[i] = mesg
        batch_info.append('{}:{:.4f}'.format(i, mesg))
    end_time = time.time()

    # write log for this epoch
    if local_rank == 0:
        logging.info('{}: {}, Time {:3.2f}'.format(phase, '  '.join(batch_info), end_time - start_time))


def distributed_train(config, world_size=0, num_gpus=0, rank_offset=0):
    """Initialize distributed training.
    
    Args:
        config (train_config): Training configuration object.
        world_size (int, optional): Total number of GPUs in distributed training. Defaults to 0.
        num_gpus (int, optional): Number of GPUs to use. Defaults to 0.
        rank_offset (int, optional): Offset for the global rank. Defaults to 0.
    """
    if not num_gpus:
        num_gpus = torch.cuda.device_count()
    if not world_size:
        world_size = num_gpus
    # mp.spawn(main_worker, nprocs=num_gpus, args=(world_size, rank_offset, config))
    main_worker(0, 1, 0, config)
    torch.cuda.empty_cache()
