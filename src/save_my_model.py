from lit_module import SegmentationModule
import torch

# Path to your trained checkpoint file
checkpoint_path = "coco-segmentation/k3dri1gd/checkpoints/unet-epoch=12-val_iou=0.0476.ckpt"

# Load your trained Lightning module
model = SegmentationModule.load_from_checkpoint(checkpoint_path)
model.freeze()  # Set model to inference mode

# Save the underlying UNet state dict to file "unet_weights.pth"
torch.save(model.model.state_dict(), "unet_weights.pth")
print("State dictionary saved successfully.")