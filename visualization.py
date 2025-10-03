# visualization.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from trainer import denormalize

def show_images(images, labels, class_labels, mean, std, num_images=8):
    """Display a grid of images with their labels."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        img = denormalize(images[i], mean, std)
        img = img.numpy().transpose(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(class_labels[labels[i]])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses, test_losses, train_acc, test_acc):
    """Plot training and testing metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_acc, label='Train Accuracy')
    ax2.plot(test_acc, label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
