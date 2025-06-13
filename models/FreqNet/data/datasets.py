import os
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    """
    Validate the dataset directory and ensure 'real' and 'fake' subfolders exist.

    Args:
        opt: Configuration options with dataset mode.
        root: Root directory containing 'real' and 'fake' folders.

    Returns:
        Dataset object constructed by `binary_dataset`.

    Raises:
        FileNotFoundError: If expected folders do not exist.
        ValueError: If one of the subdirectories is empty.
    """
    
    print(f"Checking dataset at: {root}")
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")

    real_path = os.path.join(root, 'real')
    fake_path = os.path.join(root, 'fake')

    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(f"Expected 'real/' and 'fake/' inside {root}, but they are missing!")

    # Check if folders contain images
    real_files = os.listdir(real_path)
    fake_files = os.listdir(fake_path)

    print(f"Files in {real_path}: {real_files[:5]} ... ({len(real_files)} total)")
    print(f"Files in {fake_path}: {fake_files[:5]} ... ({len(fake_files)} total)")

    if len(real_files) == 0 or len(fake_files) == 0:
        raise ValueError(f"One of the dataset folders ('real/' or 'fake/') is empty in {root}.")

    return binary_dataset(opt, root)


def binary_dataset(opt, root):
    """
    Apply image transformations and load images using ImageFolder.

    Args:
        opt: Configuration options containing flags for resize, crop, flip, etc.
        root: Root directory containing the image folders.

    Returns:
        A PyTorch Dataset with applied transformations.
    """
    
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    flip_func = transforms.RandomHorizontalFlip() if opt.isTrain and not opt.no_flip else transforms.Lambda(
        lambda img: img)
    rz_func = transforms.Resize((opt.loadSize, opt.loadSize)) if not (
                not opt.isTrain and opt.no_resize) else transforms.Lambda(lambda img: img)

    return datasets.ImageFolder(
        root,
        transforms.Compose([
            rz_func,
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )


class FileNameDataset(datasets.ImageFolder):
    """Custom dataset that returns file paths instead of image tensors."""
    
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    """
    Apply Gaussian blur or JPEG compression augmentation to an image.

    Args:
        img: PIL image to augment.
        opt: Configuration with probabilities and parameters for augmentation.

    Returns:
        Augmented PIL image.
    """
    
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    """Sample a continuous value within a range."""
    
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    """Randomly choose an item from a list."""
    
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    """Apply Gaussian blur to each RGB channel."""
    
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    """Apply JPEG compression using OpenCV."""
    
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    """Apply JPEG compression using PIL."""
    
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}

def jpeg_from_key(img, compress_val, key):
    """Apply JPEG compression based on the specified method key."""
    
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}

def custom_resize(img, opt):
    """
    Resize image using selected interpolation method.

    Args:
        img: PIL image to resize.
        opt: Configuration with interpolation method and target size.

    Returns:
        Resized PIL image.
    """
    
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize,opt.loadSize), interpolation=rz_dict[interp])
