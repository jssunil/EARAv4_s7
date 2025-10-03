# metrics.py
import torch
import torch.nn.functional as F

def count_correct_predictions(predictions, targets):
    """Count correct predictions."""
    if predictions.dim() != 2 or targets.dim() != 1:
        raise ValueError("Invalid tensor dimensions")
    
    predicted_classes = predictions.argmax(dim=1)
    correct_predictions = (predicted_classes == targets).sum().item()
    return correct_predictions

def calculate_accuracy(predictions, targets):
    """Calculate accuracy percentage."""
    correct = count_correct_predictions(predictions, targets)
    total = targets.size(0)
    return 100.0 * correct / total

def calculate_loss(output, target, reduction='mean'):
    """Calculate NLL loss."""
    return F.nll_loss(output, target, reduction=reduction)
