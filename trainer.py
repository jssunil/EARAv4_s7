import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import math
import numpy
import glob
import torchvision
import matplotlib.pyplot as plt
import io
import glob
import os
from shutil import move, copy
from os.path import join
from os import listdir, rmdir
import time
import numpy as np
import random
from tqdm import tqdm

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor using mean and standard deviation.
    
    Args:
        tensor (torch.Tensor): Normalized tensor of shape (C, H, W) or (B, C, H, W)
        mean (list or tuple): Mean values for each channel [R, G, B]
        std (list or tuple): Standard deviation values for each channel [R, G, B]
    
    Returns:
        torch.Tensor: Denormalized tensor of the same shape as input
    
    Raises:
        ValueError: If tensor doesn't have 3 or 4 dimensions
        TypeError: If mean or std are not iterable
    """
    # Validate input tensor dimensions
    if tensor.dim() not in [3, 4]:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")
    
    # Validate mean and std are iterable
    if not hasattr(mean, '__iter__') or not hasattr(std, '__iter__'):
        raise TypeError("mean and std must be iterable (list, tuple, or tensor)")
    
    # Convert mean and std to tensors and ensure they match tensor device
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    
    # Reshape for broadcasting: (C,) -> (1, C, 1, 1) or (1, 1, C, 1, 1)
    if tensor.dim() == 3:
        # Input: (C, H, W) -> (1, C, 1, 1)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    else:
        # Input: (B, C, H, W) -> (1, 1, C, 1, 1)
        mean = mean.view(1, 1, -1, 1, 1)
        std = std.view(1, 1, -1, 1, 1)
    
    # Denormalize: x = (x_normalized * std) + mean
    denormalized = tensor * std + mean
    
    return denormalized

def imshow(img,mean, std):
	img = denormalize(img,mean, std)
	npimg = img.numpy()
	plt.imshow(numpy.transpose(npimg, (1, 2, 0)))

def count_correct_predictions(predictions, targets):
    """
    Count the number of correct predictions by comparing predictions with ground truth.
    
    Args:
        predictions (torch.Tensor): Model predictions of shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth labels of shape (batch_size,)
    
    Returns:
        int: Number of correct predictions in the batch
    
    Raises:
        ValueError: If predictions and targets have incompatible shapes
    """
    # Validate input shapes
    if predictions.dim() != 2:
        raise ValueError(f"Predictions must be 2D tensor (batch_size, num_classes), got {predictions.dim()}D")
    
    if targets.dim() != 1:
        raise ValueError(f"Targets must be 1D tensor (batch_size,), got {targets.dim()}D")
    
    if predictions.size(0) != targets.size(0):
        raise ValueError(f"Batch size mismatch: predictions {predictions.size(0)} vs targets {targets.size(0)}")
    
    # Get predicted class indices (argmax along class dimension)
    predicted_classes = predictions.argmax(dim=1)
    
    # Compare predictions with ground truth and count matches
    correct_predictions = (predicted_classes == targets).sum().item()
    
    return correct_predictions

def train(model, device, train_loader, optimizer, train_acc, train_losses):
    """Trains the model

    Args:
        model (torch.nn.Module): pytorch model
        device (_type_): cuda or cpu
        train_loader (torch.utils.data.DataLoader): data iterator on training data
        optimizer (torch.optim): optimizer function
        train_acc (list): stores accuracy of each batch
        train_losses (list): stores loss of each batch
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = F.nll_loss(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        correct += count_correct_predictions(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, test_acc, test_losses):
    """_summary_

    Args:
        model (torch.nn.Module): pytorch model
        device (_type_): cuda or cpu
        test_loader (torch.utils.data.DataLoader): data iterator on test data
        test_acc (list): stores accuracy of each batch
        test_losses (list): stores loss of each batch
    """
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            correct += count_correct_predictions(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))