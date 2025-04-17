import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from unet import UNet  # Import the UNet model you defined in Step 4
from torchmetrics.classification import JaccardIndex, Accuracy
from torchmetrics.segmentation import DiceScore

class SegmentationModule(pl.LightningModule):
    def __init__(self, n_channels=3, n_classes=11, lr=1e-3):
        """
        Args:
            n_channels (int): Number of input channels (e.g. 3 for RGB images)
            n_classes (int): Number of output segmentation classes (11 in your case)
            lr (float): Learning rate for the optimizer
        """
        super().__init__()
        self.save_hyperparameters()  # Automatically save n_channels, n_classes, and lr

        # Instantiate the UNet model
        self.model = UNet(n_channels=n_channels, n_classes=n_classes)
        
        # Loss function: CrossEntropyLoss is common for multi-class segmentation
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Metrics for validation: using TorchMetrics
        # Optionally, ignore the background class (index 0) if desired.
        self.val_iou = JaccardIndex(task="multiclass", num_classes=n_classes, ignore_index=0)
        self.val_dice = DiceScore(num_classes=n_classes, include_background=True,
                                  average='micro', input_format='index', zero_division=0.0)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)


    def forward(self, x):
        # Forward pass through the UNet model; returns logits of shape [B, n_classes, H, W]
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch  # images: [B, 3, H, W], masks: [B, H, W]
        logits = self.forward(images)  # raw model outputs
        loss = self.loss_fn(logits, masks)
        # Log training loss; on_step=True logs per batch, on_epoch=True logs epoch-average.
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self.forward(images)
        loss = self.loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)  # Convert logits to predicted class per pixel

        # Update metrics with current batch predictions and ground truth masks
        self.val_iou.update(preds, masks)
        self.val_dice.update(preds, masks)
        self.val_accuracy.update(preds, masks)
        
        # Log validation loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}
    
    def on_validation_epoch_end(self):
        # Compute aggregated metrics over the whole validation set
        iou = self.val_iou.compute()
        dice = self.val_dice.compute()
        acc = self.val_accuracy.compute()
        self.log("val_iou", iou, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        
        # Reset the metrics to prepare for the next epoch
        self.val_iou.reset()
        self.val_dice.reset()
        self.val_accuracy.reset()
    
    def configure_optimizers(self):
        # Define an optimizer; here, we use Adam.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # Define a learning rate scheduler; for example, StepLR that decays LR every 5 epochs.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"  # For schedulers needing to monitor a metric (if applicable)
        }

# For testing, you can add a simple run:
if __name__ == "__main__":
    # Create a dummy tensor to simulate a batch of one image of size 256x256 with 3 channels
    dummy_input = torch.randn(1, 3, 256, 256)
    model = SegmentationModule(n_channels=3, n_classes=11)
    output = model(dummy_input)
    print("UNet output shape:", output.shape)  # Expected: [1, 11, 256, 256]