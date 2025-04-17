# COCO UNet Segmentation Assignment

This repository contains the implementation for **Task 2: Image Segmentation** of the VJT assignment, building on the dataset you prepared in **Task 1**.

---

## 📁 Repository Structure

```
├── data/                      # (Empty; placeholder .gitkeep). Download assets into this folder before running.
├── coco-segmentation/         # (Empty; placeholder .gitkeep)
├── src/                       # Source code
│   ├── dataset.py             # Dataset class & COCO→11-class remapping
│   ├── unet.py                # UNet model definition
│   ├── lit_module.py          # PyTorch Lightning module encapsulating training/validation
│   ├── train.py               # Training script (Trainer, callbacks, WandB)
│   ├── test.py                # Evaluate on held-out test set
│   ├── inference.py           # Inference on a single image
│   ├── inference_batch.py     # Batch inference over a folder of images
│   └── dataloader_test.py     # DataLoader sanity check
├── requirements.txt           # Declared dependencies
├── uv.lock                    # Locked environment (uv)
├── .gitignore                 # Ignored files/folders (data/, model weights, etc.)
└── README.md                  # This file
```

---

## 🔗 Important Links

- **Task 1: Dataset Preparation**\
  [https://github.com/Divyaraj404/VJT\_Image\_Segmentation\_Assignment\_task1](https://github.com/Divyaraj404/VJT_Image_Segmentation_Assignment_task1)

- **Additional Assets (5 GB)**\
  Download and extract **all files** from this Google Drive folder into the project root before running any code:\
  [https://drive.google.com/drive/folders/1dE3X83SM7vdxHjRr5MxF3VaT5WFAC2uu?usp=sharing](https://drive.google.com/drive/folders/1dE3X83SM7vdxHjRr5MxF3VaT5WFAC2uu?usp=sharing)

  > The folder contains:
  >
  > - `data/train_images/` & `data/train_masks/` (training split)
  > - `data/val_images/` & `data/val_masks/` (validation split)
  > - `data/test_images/` & `data/test_masks/` (test split)
  > - Pre-generated checkpoints: `checkpoints/unet-*.ckpt`

## 🚀 Setup & Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/Divyaraj404/VJT_Image_Segmentation_Assignment.git
   cd VJT_Image_Segmentation_Assignment
   ```

2. **Download and place data & checkpoints**

   - Download the assets from the Drive link above.
   - Place the `data/` folder and `checkpoints/` folder at the project root so paths match `train.py`.

3. **Initialize environment with uv** (requires Python 3.8+):

   ```bash
   uv init --python=3.8
   uv install -r requirements.txt
   ```

4. **Activate the virtual environment**

   ```bash
   source venv/bin/activate    # or `uv shell` if configured
   ```

5. **Verify CUDA & dependencies**

   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

---

## 🏋️‍♂️ Training

Run the training script:

```bash
python src/train.py
```

**Configurable settings** (edit `src/train.py` before running):

- `train_images_dir`, `train_masks_dir`, `val_images_dir`, `val_masks_dir`: Path to data folders under `data/`.
- `BATCH_SIZE`: Batch size for training/validation DataLoaders.
- `max_epochs`: Number of epochs in `pl.Trainer(max_epochs=...)`.
- `precision`: 16 or 32 in `Trainer(precision=...)`.
- `lr`: Learning rate passed to `SegmentationModule(..., lr)`.
- WandB project/name: in `WandbLogger(project=..., name=...)`.

Training outputs the best checkpoint (by val IoU) to `lightning_logs/.../checkpoints/`.

---

## 🧪 Evaluation & Testing

Evaluate on the held‑out test set:

```bash
python src/test.py
```

Prints Test IoU, Dice score, and pixel accuracy.

---

## 🔍 Inference

### Single‑image inference

```bash
python src/inference.py --input data/new_images/img1.jpg --output data/prediction.png
```

### Batch inference

```bash
python src/inference_batch.py
```

Saves colored masks (`mask_<filename>`) and overlays (`overlay_<filename>`) in `data/predictions/`.

---