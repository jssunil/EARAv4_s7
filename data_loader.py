"""
Data Loader Module for CIFAR-10 Dataset

This module provides functions to download, preprocess, and create data loaders
for the CIFAR-10 dataset with support for data augmentation and GPU acceleration.
"""

import math
import numpy
import torch
from torchvision import datasets
from torchvision import transforms as tf
from transforms import *


def download_CIFAR10_dataset(data_path, train_transforms, test_transforms):
    """
    Downloads and loads the CIFAR-10 dataset with specified transforms.
    
    Args:
        data_path (str): Path where the dataset will be stored/downloaded
        train_transforms: Transformations to apply to training data
        test_transforms: Transformations to apply to test data
    
    Returns:
        tuple: (train_dataset, test_dataset) - PyTorch dataset objects
    """
    # Download and load training dataset with specified transforms
    train = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transforms)
    # Download and load test dataset with specified transforms
    test = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transforms)

    return train, test


def generate_transforms(means, stdevs):
    """
    Generates transformation pipelines for training and test data.
    
    Args:
        means (list): Mean values for normalization [R, G, B]
        stdevs (list): Standard deviation values for normalization [R, G, B]
    
    Returns:
        tuple: (train_transforms, test_transforms) - Transform objects
    """
    # Use Albumentation for advanced data augmentation on training data
    train_transforms = AlbumentationTransformations(means, stdevs)
    # Simple transforms for test data: convert to tensor and normalize
    test_transforms = tf.Compose([tf.ToTensor(), tf.Normalize(means, stdevs)])

    return train_transforms, test_transforms


def generate_train_test_loader(data_path, means, stdevs, batch_size):
    """
    Creates complete data loaders for training and testing with proper configuration.
    
    Args:
        data_path (str): Path where the dataset will be stored
        SEED (int): Random seed for reproducibility
        means (list): Mean values for normalization [R, G, B]
        stdevs (list): Standard deviation values for normalization [R, G, B]
        batch_size (int): Number of samples per batch
    
    Returns:
        tuple: (train_loader, test_loader) - PyTorch DataLoader objects
    """
    # Generate transformation pipelines for train and test data
    train_transforms, test_transforms = generate_transforms(means, stdevs)
    # Download and load the CIFAR-10 dataset
    train, test = download_CIFAR10_dataset(data_path, train_transforms, test_transforms)

    # Check if CUDA (GPU) is available for acceleration
    cuda = torch.cuda.is_available()

    # Set random seed for reproducibility across runs
    torch.manual_seed(1)

    if cuda:
        # Set CUDA seed for GPU operations reproducibility
        torch.cuda.manual_seed(1)

    # Configure dataloader arguments based on CUDA availability
    # CUDA-enabled: Use multiple workers, pin memory for faster GPU transfer
    # CPU-only: Use single worker, smaller batch size
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=128)

    # Create training data loader with configured arguments
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # Create test data loader with configured arguments
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

    return train_loader, test_loader
