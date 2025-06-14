import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets.augmentations import augmentations
from albumentations import CenterCrop,Compose,Resize,RandomCrop
from albumentations.pytorch.transforms import ToTensor3D as ToTensor
import json
import random
import cv2
from datasets.data import *

class DeepfakeDataset(Dataset):
    """A PyTorch Dataset class for handling deepfake detection datasets.

    This dataset class supports multiple deepfake datasets including FaceForensics++,
    Celeb-DF, DeeperForensics, and DFDC. It handles video frames and applies
    various augmentations during training.

    Attributes:
        phase (str): Dataset phase - 'train', 'val', or 'test'
        datalabel (str): Identifier for the specific dataset to load
        imgs_per_video (int): Number of frames to sample per video
        frame_interval (int): Interval between sampled frames
        num_classes (int): Number of classes (2 for binary classification)
        epoch (int): Current epoch number
        max_frames (int): Maximum number of frames to consider per video
        min_frames (int): Minimum number of frames required per video
        dataset (list): List of video paths and their labels
        aug (Compose): Albumentations augmentation pipeline
        resize (tuple): Target size for resizing frames
        trans (Compose): Transformation pipeline for preprocessing
    """

    def __init__(self,phase='train',datalabel='', resize=(320,320),imgs_per_video=30,min_frames=0,\
    normalize=dict(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),frame_interval=10,max_frames=300,augment='augment0'):
        """Initialize the DeepfakeDataset.

        Args:
            phase (str, optional): Dataset phase. Defaults to 'train'.
            datalabel (str, optional): Dataset identifier. Defaults to ''.
            resize (tuple, optional): Target size for frames. Defaults to (320,320).
            imgs_per_video (int, optional): Frames per video. Defaults to 30.
            min_frames (int, optional): Minimum frames required. Defaults to 0.
            normalize (dict, optional): Normalization parameters. Defaults to dict(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]).
            frame_interval (int, optional): Frame sampling interval. Defaults to 10.
            max_frames (int, optional): Maximum frames per video. Defaults to 300.
            augment (str, optional): Augmentation strategy. Defaults to 'augment0'.
        """
        assert phase in ['train', 'val', 'test']
        self.datalabel=datalabel
        self.phase = phase
        self.imgs_per_video=imgs_per_video
        self.frame_interval=frame_interval
        self.num_classes = 2
        self.epoch=0
        self.max_frames=max_frames
        if min_frames:
            self.min_frames=min_frames
        else:
            self.min_frames=max_frames*0.3
        self.dataset=[]
        self.aug=augmentations[augment]
        resize_=(int(resize[0]/0.8),int(resize[1]/0.8))
        self.resize=resize
        #Resize(*resize_,interpolation=cv2.INTER_CUBIC),
        self.trans=Compose([CenterCrop(*resize),ToTensor(normalize=normalize)])
        ###############
        # doing resize and center crop in trans
        if type(datalabel)!=str:
            self.dataset=datalabel
            return
        if 'ff-5' in self.datalabel:
            for i,j in enumerate(['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face']):
                temp=FF_dataset(j,self.datalabel.split('-')[2],phase)
                temp=[[k[0],i]  for k in temp ]
                self.dataset+=temp
        elif 'ff-all' in self.datalabel:
            for i in ['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face']:
                self.dataset+=FF_dataset(i,self.datalabel.split('-')[2],phase)
            if phase!='test':
                self.dataset=make_balance(self.dataset)
        elif 'ff' in self.datalabel:
            self.dataset+=FF_dataset(self.datalabel.split('-')[1],self.datalabel.split('-')[2],phase)+FF_dataset("Origin",self.datalabel.split('-')[2],phase)
        elif 'celeb' in self.datalabel:
            self.dataset=Celeb_test
        elif 'deeper' in self.datalabel:
            self.dataset=deeperforensics_dataset(phase)+FF_dataset('Origin',self.datalabel.split('-')[1],phase)
        elif 'dfdc' in self.datalabel:
            self.dataset=dfdc_dataset(phase)
        else: raise(Exception('no such datset'))

    def next_epoch(self):
        """Increment the epoch counter.
        
        This method is called at the start of each epoch to update the frame sampling.
        """
        self.epoch+=1

    def __getitem__(self, item):
        """Get a single sample from the dataset.

        Args:
            item (int): Index of the sample to retrieve.

        Returns:
            tuple: (image tensor, label) where image is preprocessed and label is the class (0 for real, 1 for fake).
        """
        try:
            vid=self.dataset[item//self.imgs_per_video]
            vd=sorted(os.listdir(vid[0]))
            if len(vd)<self.min_frames:
                raise(Exception(str(vid)))
                #return self.__getitem__((item+self.imgs_per_video)%(self.__len__()))
            ind=(item%self.imgs_per_video*self.frame_interval+self.epoch)%min(len(vd),self.max_frames)
            ind=vd[ind]
            image =cv2.imread(os.path.join(vid[0],ind))
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.phase=='train':
                image=self.aug(image=image)['image']
            return self.trans(image=image)['image'], vid[1]
        except Exception as e:
            #print(e)
            #raise(e)
            if self.phase!='test':
                return self.__getitem__((item+self.imgs_per_video)%(self.__len__()))
            else:
                return torch.zeros(3,self.resize[0],self.resize[1]),-1

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples (videos * frames_per_video).
        """
        return len(self.dataset)*self.imgs_per_video


