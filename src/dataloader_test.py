import os
import torch
from torch.utils.data import DataLoader
from dataset import COCOSegDataset, train_transforms, val_transforms

train_images_dir = "data/train_images"
train_masks_dir  = "data/train_masks"
val_images_dir   = "data/val_images"
val_masks_dir    = "data/val_masks"

# Create dataset instances
train_dataset = COCOSegDataset(train_images_dir, train_masks_dir, transform=train_transforms)
val_dataset = COCOSegDataset(val_images_dir, val_masks_dir, transform=val_transforms)

# Create DataLoaders
BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Test DataLoader: Print 10 batch shape and unique mask values
if __name__ == "__main__":
    # Try iterating over 10 training batch
    for idx in range(10):
        image, mask = train_dataset[idx]  # use train_dataset instead of dataset
        print(f"Sample {idx}:")
        print("  Image shape:", image.shape)      # Expected: [3, H, W] e.g., [3, 256, 256]
        print("  Mask shape:", mask.shape)        # Expected: [H, W] e.g., [256, 256]
        unique_vals = torch.unique(mask).tolist()
        print("  Unique mask values:", unique_vals)
