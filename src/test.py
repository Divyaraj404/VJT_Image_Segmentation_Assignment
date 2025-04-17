import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import COCOSegDataset, val_transforms  # reuse validation transforms for test
from lit_module import SegmentationModule
from torchmetrics.classification import JaccardIndex, Accuracy
from torchmetrics.segmentation import DiceScore

def main():
    # -------------------------------
    # 1. Prepare the Test Dataset
    # -------------------------------
    test_images_dir = "data/val_images"
    test_masks_dir = "data/val_masks"
    
    test_dataset = COCOSegDataset(test_images_dir, test_masks_dir, transform=val_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # -------------------------------
    # 2. Load the Trained Model Checkpoint
    # -------------------------------
    # Replace 'best.ckpt' with the path to your best checkpoint
    checkpoint_path = "coco-segmentation/k3dri1gd/checkpoints/unet-epoch=12-val_iou=0.0476.ckpt"  # update this path accordingly
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # -------------------------------
    # 3. Initialize Evaluation Metrics
    # -------------------------------
    iou_metric = JaccardIndex(task="multiclass", num_classes=11, ignore_index=0).to(device)
    dice_metric = DiceScore(num_classes=11, include_background=False, average='micro', input_format='index').to(device)
    acc_metric = Accuracy(task="multiclass", num_classes=11).to(device)

    # -------------------------------
    # 4. Run Inference on the Test Set
    # -------------------------------
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)                  # Forward pass: [B, 11, H, W]
            preds = torch.argmax(logits, dim=1)       # Get predicted class per pixel
            
            # Update metrics
            iou_metric.update(preds, masks)
            dice_metric.update(preds, masks)
            acc_metric.update(preds, masks)
    
    # -------------------------------
    # 5. Compute and Print the Final Metrics
    # -------------------------------
    final_iou = iou_metric.compute()
    final_dice = dice_metric.compute()
    final_acc = acc_metric.compute()
    
    print("Evaluation on Test Data:")
    print(f"  Test IoU:         {final_iou.item():.4f}")
    print(f"  Test Dice Score:  {final_dice.item():.4f}")
    print(f"  Test Pixel Acc:   {final_acc.item():.4f}")

if __name__ == "__main__":
    main()