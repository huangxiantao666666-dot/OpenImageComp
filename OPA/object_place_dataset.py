"""
Script Name: dataset.py
Purpose: Dataset and DataLoader for OPA (Object Placement Assessment) training/testing.

This script provides:
1. Custom Dataset class that loads composite images, their masks, and rationality labels
2. Data augmentation (random horizontal flip)
3. DataLoaders for training and testing phases

The dataset format follows the OPA specification:
- Each sample: composite image + foreground mask + binary label (0/1) + bounding box
- Images and masks are stored in separate directories as specified in CSV files
"""

import csv
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from config import opt  # Import configuration options (batch_size, img_size, paths, etc.)

# Set random seed for reproducibility of data augmentation (flipping)
torch.random.manual_seed(1)


# ==================== CUSTOM DATASET CLASS ====================
class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for OPA task.
    
    Loads composite images and their corresponding:
    - Foreground masks (from the original COCO segmentation)
    - Binary rationality labels (0=unreasonable, 1=reasonable)
    - Bounding box of the foreground object in the composite image
    
    Args:
        istrain: If True, loads training set; if False, loads test set
    """
    
    def __init__(self, istrain=True):
        """
        Initialize the dataset by reading CSV file and preparing file paths.
        """
        self.istrain = istrain
        
        # Read the CSV file containing dataset information
        # opt.train_data_path or opt.test_data_path points to train_set.csv or test_set.csv
        with open(opt.train_data_path if istrain else opt.test_data_path, "r") as f:
            reader = csv.reader(f)
            reader = list(reader)
            reader = reader[1:]  # Skip header row (first line)
        
        # Initialize lists to store data from each row
        self.labels = []           # Binary labels (0 or 1)
        self.images_path = []      # Paths to composite images
        self.mask_path = []        # Paths to foreground masks
        self.target_box = []       # Bounding box coordinates [x, y, width, height]
        self.dic_name = []         # Original image filenames (for reference)
        
        # Parse each row of the CSV file
        for row in reader:
            label = int(row[-3])                    # Label is the 3rd last column
            image_path = row[-2]                    # Image path is the 2nd last column
            mask_path = row[-1]                     # Mask path is the last column
            target_box = eval(row[2])               # Bounding box is the 3rd column (stored as string)
            
            self.labels.append(label)
            # Join base path with relative image path
            self.images_path.append(os.path.join(opt.img_path, image_path))
            self.mask_path.append(os.path.join(opt.mask_path, mask_path))
            self.target_box.append(target_box)
            self.dic_name.append(image_path)
        
        # Define transformation for both images and masks:
        # 1. Resize to square of size opt.img_size (e.g., 224x224)
        # 2. Convert to Tensor (scales pixel values from [0,255] to [0,1])
        self.img_transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor()
        ])
        
        # Define horizontal flip augmentation (used during training)
        self.transforms_flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1)  # Always flip when called
        ])
    
    def __getitem__(self, index):
        """
        Retrieve one sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            img_mask: Concatenated tensor of composite image and mask [6, H, W]
            label: Binary rationality label (0 or 1)
            target_box: Bounding box coordinates [x1, y1, x2, y2]
        """
        
        # ---------- Step 1: Load and transform composite image ----------
        img = Image.open(self.images_path[index]).convert('RGB')
        w = img.width   # Original width (before resizing, needed for bounding box transformation)
        h = img.height  # Original height
        img = self.img_transform(img)  # Resize and convert to tensor, shape: [3, H, W]
        
        # ---------- Step 2: Load and transform foreground mask ----------
        mask = Image.open(self.mask_path[index]).convert('L')  # Grayscale mask (single channel)
        mask = self.img_transform(mask)  # Same transformation: resize + tensor, shape: [1, H, W]
        
        # ---------- Step 3: Data Augmentation (Horizontal Flip) ----------
        is_flip = False
        # Only apply flipping during training, with 50% probability
        if self.istrain and np.random.uniform() < 0.5:
            img = self.transforms_flip(img)   # Flip image horizontally
            mask = self.transforms_flip(mask) # Flip mask horizontally (same transformation)
            is_flip = True  # Record that flipping occurred (to adjust bounding box)
        
        # ---------- Step 4: Concatenate image and mask ----------
        # Stack image (3 channels) and mask (1 channel) along channel dimension
        # Result shape: [4, H, W] for SimOPA? Wait, mask has 1 channel, so 3+1=4 channels
        # But the paper says they concatenate image and mask. Actually mask is treated as an extra channel.
        img_mask = torch.cat([img, mask], dim=0)  # Shape: [4, H, W]
        
        # ---------- Step 5: Load and transform bounding box ----------
        label = self.labels[index]
        target_box = self.target_box[index]  # Original format: [x1, y1, bw, bh] (top-left corner + dimensions)
        x1, y1, bw, bh = target_box
        x2, y2 = x1 + bw, y1 + bh  # Convert to [x1, y1, x2, y2] format (top-left and bottom-right corners)
        
        # If the image was flipped horizontally, the bounding box needs to be transformed accordingly
        if is_flip:
            # Flip x-coordinates: new_x = width - old_x
            x1 = w - x1
            x2 = w - x2
            # Swap x1 and x2 to maintain (x1 < x2) ordering
            x1, x2 = x2, x1
        
        # Return bounding box as tensor of 4 integers
        target_box = torch.tensor([x1, y1, x2, y2])
        
        return img_mask, label, target_box
    
    def __len__(self):
        """Return total number of samples in the dataset."""
        return len(self.labels)


# ==================== DATALOADER FACTORY FUNCTIONS ====================
def get_train_dataloader():
    """
    Create and return a DataLoader for the training set.
    
    Features:
    - Shuffles data at each epoch
    - Uses drop_last=True to ensure all batches have the same size (helps with batch norm)
    - Uses pin_memory for faster GPU transfer
    """
    trainset = ImageDataset(istrain=True)
    print('Training images', len(trainset))
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=opt.batch_size,
        shuffle=True,           # Shuffle for stochastic gradient descent
        num_workers=opt.num_workers,  # Number of subprocesses for data loading
        drop_last=True,         # Drop the last incomplete batch
        pin_memory=True         # Speed up host-to-GPU transfer
    )
    return train_loader


def get_test_dataloader():
    """
    Create and return a DataLoader for the test set.
    
    Features:
    - No shuffling (evaluation order doesn't matter)
    - Larger batch size (2x training batch size) for faster inference
    - drop_last=False to keep all test samples
    """
    testset = ImageDataset(istrain=False)
    print('Testing images', len(testset))
    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size=opt.batch_size * 2,  # Double batch size for testing
        shuffle=False,                   # No need to shuffle test data
        num_workers=opt.num_workers,
        drop_last=False,                 # Keep all test samples
        pin_memory=True
    )
    return test_loader


# ==================== MAIN: VERIFICATION CODE ====================
if __name__ == "__main__":
    """
    Quick verification script to ensure the dataset loads correctly.
    Prints the shapes of batches to confirm dimensions.
    """
    # Test training data loader
    train_loader = get_train_dataloader()
    for batch_index, (img, label, target_box) in enumerate(train_loader):
        # Expected shapes:
        # img: [batch, 4, H, W] where H=W=opt.img_size (e.g., 224)
        # label: [batch] (integers 0 or 1)
        # target_box: [batch, 4] (x1, y1, x2, y2 coordinates)
        print(img.shape, label.shape, target_box.shape)
        if batch_index > 10:
            break
    
    # Test test data loader
    test_loader = get_test_dataloader()
    for batch_index, (img, label, target_box) in enumerate(test_loader):
        print(img.shape, label.shape, target_box.shape)
        if batch_index > 10:
            break