import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T  # For basic transforms if no augmentation is provided

# -----------------------------------------------------------------------------
# Define the class remapping: original COCO class IDs are mapped to 11 classes.
# -----------------------------------------------------------------------------
# Custom mapping:
# 0: Background      (0 remains 0)
# 1: Person          (1 maps to 1)
# 2: Vehicle         (2, 3, 4, 5, 6, 7, 8, 9 map to 2)
# 3: Street/Traffic  (10, 11, 13, 14, 15 map to 3)
# 4: Animal          (16, 17, 18, 19, 20, 21, 22, 23, 24, 25 map to 4)
# 5: Accessories     (27, 28, 31, 32, 33 map to 5)
# 6: Sports          (34, 35, 36, 37, 38, 39, 40, 41, 42, 43 map to 6)
# 7: Food & Kitchen  (44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61 map to 7)
# 8: Furniture       (62, 63, 64, 65, 67, 70 map to 8)
# 9: Electronics     (72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82 map to 9)
# 10: Miscellaneous  (84, 85, 86, 87, 88, 89, 90 map to 10)
# -----------------------------------------------------------------------------

coco_to_super = {
    0: 0,          # background
    1: 1,          # person
    2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2,  # Vehicle
    10: 3, 11: 3, 13: 3, 14: 3, 15: 3,             # Street/Traffic Objects
    16: 4, 17: 4, 18: 4, 19: 4, 20: 4, 21: 4, 22: 4, 23: 4, 24: 4, 25: 4,  # Animal
    27: 5, 28: 5, 31: 5, 32: 5, 33: 5,             # Accessories
    34: 6, 35: 6, 36: 6, 37: 6, 38: 6, 39: 6, 40: 6, 41: 6, 42: 6, 43: 6,  # Sports
    44: 7, 46: 7, 47: 7, 48: 7, 49: 7, 50: 7, 51: 7, 52: 7, 53: 7, 54: 7,
    55: 7, 56: 7, 57: 7, 58: 7, 59: 7, 60: 7, 61: 7,  # Food & Kitchen Items
    62: 8, 63: 8, 64: 8, 65: 8, 67: 8, 70: 8,      # Furniture
    72: 9, 73: 9, 74: 9, 75: 9, 76: 9, 77: 9, 78: 9, 79: 9, 80: 9, 81: 9, 82: 9,  # Electronics/Appliances
    84: 10, 85: 10, 86: 10, 87: 10, 88: 10, 89: 10, 90: 10  # Miscellaneous
}

# -----------------------------------------------------------------------------
# Function to remap the mask pixels using vectorized lookup.
# -----------------------------------------------------------------------------
def remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remap the original mask pixel values (COCO IDs) to new class IDs (0-10).
    This is done using a lookup table for efficiency.
    
    Args:
        mask (np.ndarray): Original mask with shape (H, W) where pixel values are COCO IDs.
    Returns:
        np.ndarray: New mask with remapped values.
    """
    # Allocate a lookup table for indices 0 to max possible id (assume 0-90)
    lookup = np.zeros(91, dtype=np.uint8)
    for orig_id, new_id in coco_to_super.items():
        lookup[orig_id] = new_id
    # Use the lookup table to remap the entire mask
    new_mask = lookup[mask]
    return new_mask

# -----------------------------------------------------------------------------
# Custom Dataset Class: COCOSegDataset
# -----------------------------------------------------------------------------
class COCOSegDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        """
        Args:
            images_dir (str): Directory with original .jpg images.
            masks_dir (str): Directory with corresponding 8-bit PNG mask images.
            transform (albumentations.Compose, optional): Data augmentation pipeline.
                Must be configured to apply the same transformation for both image and mask.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        # Ensure matching order by sorting the file lists.
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        assert len(self.image_files) == len(self.mask_files), "Mismatch between number of images and masks!"
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Build full file paths for the image and mask.
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        # Load image in RGB format and mask in its native mode.
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        
        # Remap the mask using the defined remapping function.
        mask = remap_mask(mask)
        
        # If augmentations are provided, apply them.
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            # Convert the image and mask to PyTorch tensors (if no augmentation is used)
            image = T.ToTensor()(image)  # Converts image into shape [3, H, W] in range [0,1]
            mask = torch.as_tensor(mask, dtype=torch.long)  # Mask remains [H, W] with integer class values
        return image, mask
    
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define normalization parameters (commonly used values)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Training augmentations
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # Pad images smaller than 256x256
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
    A.RandomCrop(height=256, width=256, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    # Ensure fixed size after augmentations
    A.Resize(height=256, width=256, p=1.0),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# Validation augmentations (minimal: only resizing and normalization)
val_transforms = A.Compose([
    A.Resize(height=256, width=256, p=1.0),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# -----------------------------------------------------------------------------
# Example: Testing the Dataset Class
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test: Adjust paths according to your directory structure.
    train_images_dir = "../data/train_images"  # Example: adjust accordingly
    train_masks_dir  = "../data/train_masks"
    
    # For now, no transform is used; this is for testing the remapping and DataLoader.
    dataset = COCOSegDataset(train_images_dir, train_masks_dir, transform=None)
    sample_img, sample_mask = dataset[0]
    print("Image tensor shape:", sample_img.shape)  # Should be [3, H, W]
    print("Mask tensor shape:", sample_mask.shape)    # Should be [H, W]
    print("Unique mask values:", torch.unique(sample_mask).tolist())