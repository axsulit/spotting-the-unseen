import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger

def seed_torch(seed=100):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    
    opt = TrainOptions().parse()
    device = torch.device('cuda' if (len(opt.gpu_ids) > 0 and torch.cuda.is_available()) else 'cpu')
    print(f"[Info] Using device: {device}")
    seed_torch(100)  # Ensure reproducibility

    # Ensure dataset paths are correctly assigned
    train_dataroot = os.path.join(opt.dataroot, 'train')  # Training dataset
    val_dataroot = os.path.join(opt.dataroot, 'val')  # Validation dataset
    # train_dataroot = opt.dataroot
    # val_dataroot = opt.dataroot
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))

    print(f"Checking dataset paths:")
    print(f"Training Path: {train_dataroot}")
    print(f"Validation Path: {val_dataroot}")

    # Debugging: Check if 'real' and 'fake' exist
    real_path = os.path.join(train_dataroot, 'real')
    fake_path = os.path.join(train_dataroot, 'fake')
    print(f"Checking if '{real_path}' exists:", os.path.exists(real_path))
    print(f"Checking if '{fake_path}' exists:", os.path.exists(fake_path))

    print('  '.join(list(sys.argv)))

    # Training dataset options
    train_opt = TrainOptions().parse(print_options=False)
    train_opt.dataroot = train_dataroot
    train_opt.classes = ['real', 'fake']

    # Validation dataset options
    val_opt = TestOptions().parse(print_options=False)
    val_opt.dataroot = val_dataroot
    val_opt.classes = ['real', 'fake']

    # Create data loaders
    train_loader = create_dataloader(train_opt)
    val_loader = create_dataloader(val_opt)

    # TensorBoard Logging
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    # Initialize Model
    model = Trainer(opt)

    model.model.to(device)

    # Start Training
    model.train()
    print(f'Current Working Directory: {os.getcwd()}')

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            # Log Training Loss
            if model.total_steps % opt.loss_freq == 0:
                print(f"{time.strftime('%Y_%m_%d_%H_%M_%S')} - Train Loss: {model.loss:.4f} at step {model.total_steps}, lr {model.lr:.6f}")
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        # Adjust Learning Rate
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(f"{time.strftime('%Y_%m_%d_%H_%M_%S')} - Adjusting LR at epoch {epoch}, step {model.total_steps}")
            model.adjust_learning_rate()

        # Run validation on val dataset
        model.eval()
        # acc, ap = validate(model.model, val_opt)[:2]
        acc, ap, r_acc, f_acc, recall, f1, tp, tn, fp, fn, y_true, y_pred = validate(model.model, val_opt)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        val_writer.add_scalar('recall', recall, model.total_steps)
        val_writer.add_scalar('f1_score', f1, model.total_steps)
        val_writer.add_scalar('true_positives', tp, model.total_steps)
        val_writer.add_scalar('true_negatives', tn, model.total_steps)
        val_writer.add_scalar('false_positives', fp, model.total_steps)
        val_writer.add_scalar('false_negatives', fn, model.total_steps)

        print(f"(Validation @ epoch {epoch}) Accuracy: {acc:.4f}; AP: {ap:.4f}")

        print(f"Real Accuracy: {r_acc:.4f}" if r_acc is not None else "No real images in val set")
        print(f"Fake Accuracy: {f_acc:.4f}" if f_acc is not None else "No fake images in val set")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        model.train()

    # Save Final Model
    model.save_networks('last')
