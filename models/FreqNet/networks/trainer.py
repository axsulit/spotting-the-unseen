import functools
import torch
import torch.nn as nn
from networks.freqnet import freqnet
from networks.base_model import BaseModel, init_weights


class Trainer(BaseModel):
    """
    Trainer class responsible for initializing model, setting inputs, computing loss,
    optimizing parameters, and adjusting learning rate. Inherits from BaseModel.
    """
    
    def name(self):
        """Return the name of the trainer."""
        return 'Trainer'

    def __init__(self, opt):
        """
        Initialize Trainer with model, optimizer, and loss function based on options.

        Args:
            opt: Configuration object containing training parameters.
        """
        super(Trainer, self).__init__(opt)

        # self.model = freqnet()
        self.opt = opt
        self.isTrain = opt.isTrain

        # Select device (CPU if no GPU or --gpu_ids -1)
        self.device = torch.device(
            f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')

        self.model = freqnet().to(self.device)
        
        for name,pa in self.model.named_parameters():
            if pa.requires_grad: print('='*20, 'requires_grad True',name)
        for name,pa in self.model.named_parameters():
            if not pa.requires_grad: print('='*20, 'requires_grad False',name)
        print()
        
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        # self.model.to(opt.gpu_ids[0])
 

    def adjust_learning_rate(self, min_lr=1e-6):
        """
        Decay learning rate by multiplying with 0.8. Stops when below min_lr.

        Args:
            min_lr: Minimum learning rate allowed.

        Returns:
            bool: False if learning rate is below threshold, else True.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.8
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.8} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        """
        Set input data and labels to device.

        Args:
            input: Tuple of (input tensor, label tensor).
        """
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        """Run forward pass of the model."""
        self.output = self.model(self.input)

    def get_loss(self):
        """Compute loss between prediction and label."""
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        """Perform optimization step: forward pass, compute loss, backprop, update weights."""
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

