"""Main entry point for the multiple attention model training and evaluation.

This module provides functions for pretraining, resuming training, and testing
the multiple attention model. It supports both distributed and single-GPU training.
"""

from config import train_config
from train import distributed_train, main_worker
from evaluation import all_eval
import argparse
import fire
import torch
import subprocess
#torch.autograd.set_detect_anomaly(True)

# def pretrain():
#     name='Efb4'
#     url='tcp://127.0.0.1:27015'
#     Config=train_config(name,['ff-all-c23','efficientnet-b4'],url=url,attention_layer='b5',feature_layer='logits',epochs=20,batch_size=16,AGDA_loss_weight=0)
#     Config.mkdirs()
#     distributed_train(Config) 
#     procs=[subprocess.Popen(['/bin/bash','-c','CUDA_VISIBLE_DEVICES={} python main.py test {} {}'.format(i,name,j)]) for i,j in enumerate(range(-3,0))]
#     for i in procs:
#         i.wait()

def pretrain():
    """Perform pretraining of the model on a single GPU or CPU.
    
    This function initializes the model with default configuration and runs
    the training process using a single worker.
    """
    name = 'experiment_name'
    config = train_config(name, ['ff-all-c23', 'efficientnet-b4'], epochs=20, batch_size=16)
    config.mkdirs()

    # For single GPU or CPU
    main_worker(0, 1, 0, config)  # Use main_worker directly without mp.spawn()


## do pretrain first!
def aexp():
    """Run an attention experiment with distributed training.
    
    This function sets up a distributed training configuration with specific
    attention and feature layers, then launches multiple evaluation processes
    across different GPUs.
    """
    name='a1_b5_b2'  
    url='tcp://127.0.0.1:27016'
    Config=train_config(name,['ff-all-c23','efficientnet-b4'],url=url,attention_layer='b5',feature_layer='b2',epochs=50,batch_size=15,\
        ckpt='checkpoints/Efb4/ckpt_19.pth',inner_margin=[0.2,-0.8],margin=0.8)
    Config.mkdirs()
    distributed_train(Config) 
    procs=[subprocess.Popen(['/bin/bash','-c','CUDA_VISIBLE_DEVICES={} python main.py test {} {}'.format(i,name,j)]) for i,j in enumerate(range(-3,0))]
    for i in procs:
        i.wait()


def resume(name, epochs=0):
    """Resume training from a saved checkpoint.
    
    Args:
        name (str): Name of the experiment to resume.
        epochs (int, optional): Number of additional epochs to train. Defaults to 0.
    """
    Config=train_config.load(name)
    Config.epochs+=epochs
    Config.reload()
    Config.resume_optim=True
    distributed_train(Config) 
    for i in range(-3,0):
        all_eval(name,i)

def test(name, ckpt=None):
    """Run evaluation on the model.
    
    Args:
        name (str): Name of the experiment to evaluate.
        ckpt (int, optional): Checkpoint number to evaluate. Defaults to None.
    """
    all_eval(name,ckpt)
        
if __name__=="__main__":
    fire.Fire()
