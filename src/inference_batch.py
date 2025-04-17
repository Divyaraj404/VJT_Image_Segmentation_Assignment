import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lit_module import SegmentationModule

# Load your trained model from checkpoint.
model = SegmentationModule.load_from_checkpoint("coco-segmentation/k3dri1gd/checkpoints/unet-epoch=12-val_iou=0.0476.ckpt")
model.eval()
model.freeze()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Normalization parameters (should match training transforms)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def preprocess_image(image):
    """
    Converts a BGR image (as loaded by cv2) to RGB,
    resizes it to 256x256, normalizes, and converts it to a tensor.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    transform = A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    transformed = transform(image=image)
    return transformed["image"]

def apply_color_map(mask):
    """
    Normalize mask to 0-255 and apply a colormap.
    """
    if mask.max() != 0:
        mask_norm = (mask.astype(np.float32) / mask.max() * 255).astype(np.uint8)
    else:
        mask_norm = mask.astype(np.uint8)
    color_mask = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
    return color_mask

def overlay_segmentation(original, mask_color, alpha=0.5):
    """
    Overlay the color mask onto the original image.
    Both images should be in RGB.
    """
    overlay = cv2.addWeighted(original, 1 - alpha, mask_color, alpha, 0)
    return overlay

# Define folder paths.
input_folder = "data/new_images"          # Folder with input images.
output_folder = "data/predictions"          # Folder to save outputs.

# Create output folder if it doesn't exist.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder.
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)
        # Read the original image (BGR).
        orig_image = cv2.imread(image_path)
        # Resize original for overlay consistency.
        orig_resized = cv2.resize(orig_image, (256, 256))
        # Preprocess image for model inference.
        img_tensor = preprocess_image(orig_image)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            preds = torch.argmax(logits, dim=1)
        pred_mask = preds.squeeze(0).cpu().numpy()   # Shape: [256, 256]

        # Apply colormap to predicted mask.
        pred_mask_color = apply_color_map(pred_mask)
        
        # Create overlay image.
        # Convert resized original (BGR) to RGB for overlay.
        orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
        overlay = overlay_segmentation(orig_rgb, cv2.cvtColor(pred_mask_color, cv2.COLOR_BGR2RGB), alpha=0.5)
        
        # Save outputs:
        # Convert overlay from RGB back to BGR for saving via cv2.
        overlay_save = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        mask_save = pred_mask_color  # Already in BGR
        overlay_path = os.path.join(output_folder, f"overlay_{filename}")
        mask_path = os.path.join(output_folder, f"mask_{filename}")
        cv2.imwrite(overlay_path, overlay_save)
        cv2.imwrite(mask_path, mask_save)
        print(f"Processed {filename} and saved overlay and mask images.")

print("Batch inference and saving completed.")