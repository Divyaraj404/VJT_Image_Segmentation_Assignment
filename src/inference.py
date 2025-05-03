from lit_module import SegmentationModule
import cv2
import numpy as np
import torchvision.transforms as T
from albumentations import Normalize, Resize
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch

model = SegmentationModule.load_from_checkpoint("coco-segmentation/k3dri1gd/checkpoints/unet-epoch=12-val_iou=0.0476.ckpt")
model.eval()
model.freeze()  # Freeze the model for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define transformation (or reuse your val_transforms)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# Here we manually build a simple transformation: resize, normalize, and convert to tensor.
def preprocess_image(image_path):
    # Load using OpenCV (BGR), then convert to RGB.
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to 256x256 (as used in training/val pipelines)
    image = cv2.resize(image, (256, 256))
    # Normalize and convert to tensor (using Albumentations)
    transform = A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    transformed = transform(image=image)
    return transformed["image"]

# Example usage:
image_tensor = preprocess_image("data/new_image.jpg")  # Replace with actual path
image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension [1, 3, 256, 256]
image_tensor = image_tensor.to(device)

with torch.no_grad():
    logits = model(image_tensor)           # Output shape: [1, 11, H, W]
    preds = torch.argmax(logits, dim=1)      # Predicted labels: [1, H, W]
    
# Convert prediction tensor to numpy for visualization or further processing:
pred_mask = preds.squeeze(0).cpu().numpy()   # Shape: [256, 256]

# ---- Save the prediction image ----
# Function to apply a colormap for visualization (using OpenCV's COLORMAP_JET)
def apply_color_map(mask):
    # Normalize mask to the range 0-255 (if mask.max() is not 0)
    if mask.max() != 0:
        mask_norm = (mask.astype(np.float32) / mask.max() * 255).astype(np.uint8)
    else:
        mask_norm = mask.astype(np.uint8)
    # Apply color map (result will be in BGR format)
    color_mask = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
    return color_mask

pred_mask_color = apply_color_map(pred_mask)
# Save the colored prediction image
cv2.imwrite("data/prediction.png", pred_mask_color)
print("Prediction image saved as data/prediction.png")
