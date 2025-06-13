"""Configuration management module for the multiple attention model training.

This module provides the train_config class which handles all configuration parameters
for model training, including dataset settings, network architecture, and training hyperparameters.
"""

import os
import pickle
import shutil
import glob
import datetime

class train_config:
    """Configuration class for managing training parameters and settings.
    
    This class handles all aspects of the training configuration including dataset settings,
    network architecture, training hyperparameters, and experiment management.
    
    Args:
        name (str): Name of the experiment.
        recipes (list, optional): List of recipe names to apply. Defaults to [].
        **params: Additional parameters to override defaults.
    """
    
    def __init__(self, name, recipes=[], **params):
        """Initialize the training configuration with default values and apply recipes.
        
        Args:
            name (str): Name of the experiment.
            recipes (list, optional): List of recipe names to apply. Defaults to [].
            **params: Additional parameters to override defaults.
        """
        ###basic configs
        self.name = name
        self.comment = ''
        self.workers = 10                
        self.epochs = 100                
        self.learning_rate = 1e-4    
        self.adam_betas = (0.9, 0.999)
        self.weight_decay = 1e-6
        self.scheduler_step = 5
        self.scheduler_gamma = 0.5
        self.ckpt = ''
        self.resume_optim = False
        self.freeze = []
        self.url = 'tcp://127.0.0.1:27015'
        dataset_dir = r'D:\.THESIS\datasets\final\01_ffc23_final_unaltered'
        ###net config
        self.num_classes = 2
        self.num_attentions = 4
        self.attention_layer = 'b5'
        self.feature_layer = 'b1'
        self.mid_dims = 256
        self.dropout_rate = 0.25
        self.drop_final_rate = 0.5
        self.pretrained = ''
        self.alpha = 0.05
        self.alpha_decay = 0.9
        self.margin = 0.5
        self.inner_margin = [0.1, -2]
        ###AGDA configs
        self.AGDA_kernel_size = 11
        self.AGDA_dilation = 2
        self.AGDA_sigma = 7
        self.AGDA_scale_factor = 0.5
        self.AGDA_threshold = (0.4, 0.6)
        self.AGDA_zoom = (3, 5)
        self.AGDA_noise_rate = 0.1
        self.AGDA_mode = 'soft'

        ###loss configs
        self.ensemble_loss_weight = 1
        self.aux_loss_weight = 0.5
        self.AGDA_loss_weight = 1
        self.match_loss_weight = 0.1
        ###cook
        for i in recipes:
            self.recipe(i)
        for i in params:
            self.__setattr__(i, params[i])
        self.train_dataset = dict(datalabel=self.datalabel, resize=self.resize, imgs_per_video=self.imgs_per_video, normalize=self.normalize,\
            frame_interval=self.frame_interval, max_frames=self.max_frames, augment=self.augment)
        self.val_dataset = self.train_dataset
        self.net_config = dict(net=self.net, feature_layer=self.feature_layer, attention_layer=self.attention_layer, num_classes=self.num_classes, M=self.num_attentions,\
            mid_dims=self.mid_dims, dropout_rate=self.dropout_rate, drop_final_rate=self.drop_final_rate, pretrained=self.pretrained, alpha=self.alpha, size=self.resize, margin=self.margin, inner_margin=self.inner_margin)
        self.AGDA_config = dict(kernel_size=self.AGDA_kernel_size, dilation=self.AGDA_dilation, sigma=self.AGDA_sigma, scale_factor=self.AGDA_scale_factor, threshold=self.AGDA_threshold, zoom=self.AGDA_zoom, noise_rate=self.AGDA_noise_rate, mode=self.AGDA_mode)

    def recipe(self, name):
        """Apply a predefined recipe to the configuration.
        
        Args:
            name (str): Name of the recipe to apply. Can be 'ff-', 'dfdc', 'xception', or 'efficient'.
        """
        if 'ff-' in name:
            if 'ff-5' in name:
                self.num_classes = 5
            self.datalabel = name
            self.imgs_per_video = 50
            self.frame_interval = 10
            self.max_frames = 500
            self.augment = 'augment0'
        if 'dfdc' in name:
            self.datalabel = 'dfdc'
            self.max_frames = 300
            self.imgs_per_video = 30
            self.frame_interval = 10
            self.augment = 'augment2'
        if 'xception' in name:
            self.net = 'xception'
            self.batch_size = 16
            self.resize = (299, 299)
            self.normalize = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if 'efficient' in name:
            self.net = name
            self.batch_size = 16
            self.normalize = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            scale = int(name.split('b')[-1])
            sizes = [224, 240, 260, 300, 380, 456, 528, 600, 672]
            self.resize = (sizes[scale], sizes[scale])

    def mkdirs(self):
        """Create necessary directories for the experiment and save configuration.
        
        Creates directories for checkpoints, runs, and evaluations, and saves the
        current configuration. Also creates a git commit and tag for the experiment.
        """
        os.makedirs('checkpoints/' + self.name, exist_ok=True)
        os.makedirs('runs/' + self.name, exist_ok=True)
        os.makedirs('evaluations/' + self.name, exist_ok=True)
        with open('runs/%s/config.pkl' % self.name, 'wb') as f:
            pickle.dump(self, f)
        if not self.comment:
            self.comment = self.name + '_' + datetime.datetime.now().isoformat()
        os.system('git add . && git commit -m "{}" && git tag {} -f'.format(self.comment, self.name))
    

    @staticmethod
    def load(name):
        """Load a saved configuration.
        
        Args:
            name (str): Name of the experiment to load.
            
        Returns:
            train_config: A new configuration instance with loaded parameters.
        """
        with open('runs/%s/config.pkl'%name,'rb') as f:
            p=pickle.load(f)
        v=train_config('',['ff-','xception'])
        p=vars(p)
        for i in p:
            setattr(v,i,p[i])
        return v

        
    def reload(self,only_backnone=False):
        """Reload the latest checkpoint or pretrained model.
        
        Args:
            only_backnone (bool, optional): If True, only reload the backbone model.
                Defaults to False.
        """
        list_of_files = glob.glob('checkpoints/%s/*'%self.name)
        num=len(list_of_files)
        latest_file = max(list_of_files, key=os.path.getctime)
        if num>=0:
            if not only_backnone:
                self.ckpt=latest_file
            else:
                self.pretrained=latest_file