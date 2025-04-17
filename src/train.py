import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from torch.utils.data import DataLoader
from dataset import COCOSegDataset, train_transforms, val_transforms
from lit_module import SegmentationModule

def main():
    # -------------------------------
    # 1. Create Dataset Instances
    # -------------------------------
    # Set the paths to your training and validation data directories.
    train_images_dir = "data/train_images"
    train_masks_dir  = "data/train_masks"
    val_images_dir   = "data/val_images"
    val_masks_dir    = "data/val_masks"

    # Instantiate the datasets with corresponding augmentation pipelines.
    train_dataset = COCOSegDataset(train_images_dir, train_masks_dir, transform=train_transforms)
    val_dataset   = COCOSegDataset(val_images_dir, val_masks_dir, transform=val_transforms)

    # -------------------------------
    # 2. Create DataLoader Instances
    # -------------------------------
    BATCH_SIZE = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # -------------------------------
    # 3. Set Up the WandB Logger
    # -------------------------------
    # The WandB logger will record your metrics and hyperparameters.
    wandb_logger = WandbLogger(project="coco-segmentation", name="unet-exp1")

    # -------------------------------
    # 4. Configure Callbacks
    # -------------------------------
    # Save the best checkpoint based on validation IoU (higher is better).
    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        filename="unet-{epoch:02d}-{val_iou:.4f}"
    )

    # Early stopping to avoid overfitting if validation IoU does not improve for 'patience' epochs.
    early_stop_callback = EarlyStopping(
        monitor="val_iou",
        mode="max",
        patience=5,
        verbose=True
    )

    # Monitor learning rate over epochs.
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # -------------------------------
    # 5. Instantiate the Lightning Module
    # -------------------------------
    # This module includes your UNet architecture and all training/validation steps.
    model = SegmentationModule(n_channels=3, n_classes=11, lr=1e-3)

    # -------------------------------
    # 6. Instantiate the Trainer
    # -------------------------------
    # Configure the trainer with mixed precision for improved memory efficiency.
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        precision=16,  # Enable 16-bit (mixed) precision training
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10
    )

    # -------------------------------
    # 7. Start Training
    # -------------------------------
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()