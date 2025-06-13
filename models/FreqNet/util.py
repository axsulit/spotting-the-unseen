import sys
import os
import torch


def mkdirs(paths):
    """
    Create directories for a list of paths or a single path if it does not exist.

    Args:
        paths (str or list): A path or list of paths to create.
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """
    Create a single directory if it does not exist.

    Args:
        path (str): The directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Undo normalization on a tensor image.

    Args:
        tens (Tensor): Normalized tensor of shape (N, C, H, W).
        mean (list): Mean values used for normalization.
        std (list): Standard deviation values used for normalization.

    Returns:
        Tensor: Unnormalized tensor.
    """
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]




class Logger(object):
    """
    Logger that duplicates stdout to a log file.

    Attributes:
        terminal: Standard output stream.
        log: File stream to write logs.
    """

    def __init__(self, outfile):
        """
        Initialize Logger.

        Args:
            outfile (str): File path to write logs.
        """
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        """Write message to both stdout and log file."""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """Flush both stdout and log file."""
        self.terminal.flush()
        
        
def printSet(set_str):
    """
    Pretty print a set string with surrounding lines.

    Args:
        set_str (str): The string to print.
    """
    set_str = str(set_str)
    num = len(set_str)
    print("="*num*3)
    print(" "*num + set_str)
    print("="*num*3)