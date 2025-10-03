# config.py
import torch

class Config:
    # Dataset parameters
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    CIFAR10_CLASS_LABELS = ("airplane", "automobile", "bird", "cat", "deer", 
                           "dog", "frog", "horse", "ship", "truck")
    
    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    DROPOUT_VALUE = 0.05
    NUM_EPOCHS = 25
    STEP_SIZE = 24
    GAMMA = 0.1
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data paths
    DATA_PATH = './data'
