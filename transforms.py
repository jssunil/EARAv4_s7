import math
import numpy
import cv2
import torch
from torchvision import datasets
from torchvision import transforms as tf
from albumentations import *
from albumentations.pytorch import ToTensorV2

class AlbumentationTransformations():

  def __init__(self, means, stdevs):
    """
    Initialize the AlbumentationTransformations class.
    Args:
        means (list): Mean values for normalization (per channel, e.g., [0.5, 0.5, 0.5] for RGB).
        stdevs (list): Standard deviation values for normalization (per channel).
    """
    self.means = numpy.array(means)   # Store the means as a numpy array
    self.stdevs = numpy.array(stdevs) # Store the stds as a numpy array
    
    patch_size = 32  # Image patch size (not directly used here, maybe for reference)

    # Define a composition of albumentations transformations
    self.album_transforms = Compose([
      HorizontalFlip(p=0.5),   # Randomly flip the image horizontally with 50% probability
      ShiftScaleRotate(p=0.5), # Randomly shift, scale, and rotate the image with 50% probability
      CoarseDropout(           # Randomly mask out a rectangular region in the image
        max_holes=1,           # Max number of regions to drop
        max_height=16,         # Max height of the dropout region
        max_width=1,           # Max width of the dropout region (1px as specified)
        min_holes=1,           # Min number of regions to drop
        min_height=16,         # Min height of the dropout region
        min_width=16,          # Min width of the dropout region
        fill_value=self.means, # Fill the dropped-out region with the mean value
        mask_fill_value=None   # Not using a mask fill value
      ),
      Normalize(mean=self.means, std=self.stdevs), # Normalize the image using provided means and stds
      ToTensorV2()             # Convert the augmented image into a PyTorch tensor
    ])
        
  def __call__(self, img):
      """
      Make the class callable like a function.
      Args:
          img (PIL Image / ndarray): Input image
      Returns:
          Tensor: Transformed image as a PyTorch tensor
      """
      img = numpy.array(img)  # Convert PIL image to numpy array (albumentations works with numpy arrays)
      img = self.album_transforms(image=img)['image']  # Apply albumentations transformations
      return img