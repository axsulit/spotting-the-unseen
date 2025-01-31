# This file contains utility functions for the project.

import os

# Function to create a directory if it does not exist
def create_directory(path):
    """
    Creates a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)
