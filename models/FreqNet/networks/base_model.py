# from pix2pix
import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class BaseModel(nn.Module):
    """
    Base class for all models. Handles training state, saving/loading, and device management.

    Attributes:
        opt: Configuration options.
        total_steps: Number of total training steps.
        isTrain: Boolean flag for training mode.
        lr: Learning rate.
        save_dir: Path to save checkpoints.
        device: PyTorch device (CPU or CUDA).
    """
    
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    def save_networks(self, epoch):
        """
        Save model parameters to disk.

        Args:
            epoch: Current epoch number used for naming the checkpoint file.
        """
        
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(self.model.state_dict(), save_path)
        print(f'Saving model {save_path}')

    def load_networks(self, epoch):
        """
        Load model and optimizer parameters from disk.

        Args:
            epoch: Epoch number of the checkpoint to load.
        """
        load_filename = 'model_epoch_%s.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)

        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        self.model.load_state_dict(state_dict['model'])
        self.total_steps = state_dict['total_steps']

        if self.isTrain and not self.opt.new_optim:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            ### move optimizer state to GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.optimizer.param_groups:
                g['lr'] = self.opt.lr

    def eval(self):
        """Set model to evaluation mode."""
        
        self.model.eval()

    def train(self):
        """Set model to training mode."""
        
        self.model.train()

    def test(self):
        """Perform forward pass without computing gradients."""
        
        with torch.no_grad():
            self.forward()


def init_weights(net, init_type='normal', gain=0.02):
    """
    Initialize network weights.

    Args:
        net: The neural network to initialize.
        init_type: Initialization type (normal, xavier, kaiming, orthogonal).
        gain: Scaling factor for some initialization types.

    Raises:
        NotImplementedError: If an unsupported initialization type is given.
    """
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
