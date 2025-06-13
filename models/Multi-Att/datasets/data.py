import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomDeepfakeDataset(Dataset):
    """A PyTorch Dataset class for handling custom deepfake detection datasets.

    This dataset class is designed for a simple directory structure where images are
    organized into 'real' and 'fake' subdirectories within train/validate/test folders.

    Attributes:
        root_dir (str): Root directory containing the dataset
        phase (str): Current phase ('train', 'validate', or 'test')
        transform (callable): Optional transform to be applied to images
        data_dir (str): Path to the current phase directory
        real_dir (str): Path to real images directory
        fake_dir (str): Path to fake images directory
        image_paths (list): List of all image paths
        labels (list): List of corresponding labels (0 for real, 1 for fake)
    """

    def __init__(self, root_dir, phase='train', transform=None):
        """Initialize the CustomDeepfakeDataset.

        Args:
            root_dir (str): Directory where the dataset is located (root of `train`, `validate`, `test`).
            phase (str, optional): 'train', 'validate', or 'test' to specify which subset to load. Defaults to 'train'.
            transform (callable, optional): Optional transformation to be applied on a sample. Defaults to None.
        """
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform

        # Paths to 'train', 'validate', or 'test' folders
        self.data_dir = os.path.join(self.root_dir, phase)
        self.real_dir = os.path.join(self.data_dir, 'real')
        self.fake_dir = os.path.join(self.data_dir, 'fake')

        # Get the list of images in 'real' and 'fake' subdirectories
        self.real_images = [os.path.join(self.real_dir, fname) for fname in os.listdir(self.real_dir)]
        self.fake_images = [os.path.join(self.fake_dir, fname) for fname in os.listdir(self.fake_dir)]

        # Combine the real and fake images with corresponding labels
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)  # 0 for real, 1 for fake

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns:
            int: Total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image tensor, label) where image is preprocessed and label is the class (0 for real, 1 for fake).
        """
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')  # Ensure the image is in RGB format
        
        label = self.labels[idx]

        # Apply transformations (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        return image, label
