# COCO UNet Segmentation Assignment – Task 2

This repository contains the implementation for **Task 2: Image Segmentation** of the VJT assignment, building on the dataset prepared in **[Task 1](https://github.com/Divyaraj404/VJT_Image_Segmentation_Assignment_task1)**.

---

## 📁 Repository Structure
```
├── data/                      # (Empty; download assets here – see below)
├── coco-segmentation/         # (Empty placeholder)
├── src/                       # Source code
│   ├── dataset.py             # Dataset & COCO→11‑class remapping
│   ├── unet.py                # UNet architecture
│   ├── lit_module.py          # LightningModule (loss + metrics)
│   ├── train.py               # Training script (Trainer, callbacks, WandB)
│   ├── test.py                # Evaluation on unseen test set
│   ├── inference.py           # Single‑image inference
│   ├── inference_batch.py     # Batch inference over a folder
│   └── dataloader_test.py     # DataLoader sanity check
├── requirements.txt           # Dependencies
├── uv.lock                    # Locked env (uv)
├── .gitignore                 # Ignore rules (data/, checkpoints, etc.)
└── README.md                  # You’re reading it
```

---

## 🔗 Links & Assets

| Purpose | Link |
| ------- | ---- |
| **Task 1 code** | <https://github.com/Divyaraj404/VJT_Image_Segmentation_Assignment_task1> |
| **Dataset & checkpoints (Google Drive)**<br>Download & extract into project root *before running* | <https://drive.google.com/drive/folders/1dE3X83SM7vdxHjRr5MxF3VaT5WFAC2uu?usp=sharing> |
| **Final model training runs (WandB)** | <https://wandb.ai/chundawatdivyaraj01-indian-institute-of-science/coco-segmentation?nw=nwuserchundawatdivyaraj01> |
| **Earlier experiment logs (baseline model)** | <https://wandb.ai/chundawatdivyaraj01-indian-institute-of-science/segmentation-training?nw=nwuserchundawatdivyaraj01> |

> **Important:** The Drive folder contains `data/train_*`, `data/val_*`, `data/test_*` and optional pre‑trained checkpoints. Place these folders directly under the repo root so their paths match the defaults in `src/train.py`.

---

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/Divyaraj404/VJT_Image_Segmentation_Assignment.git
cd VJT_Image_Segmentation_Assignment

# Download dataset/checkpoints from Drive → put in ./data/

# Create & install env with uv
uv init --python=3.8
uv install -r requirements.txt
source venv/bin/activate   # or `uv shell`
```
Verify GPU:
```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
PY
```

---

## 🏋️‍♂️ Training

```bash
python src/train.py
```
The script:
* Loads data from `data/train_*` / `data/val_*`
* Logs all metrics to **Weights & Biases** (links above)
* Saves best checkpoint (by val IoU) under `lightning_logs/.../checkpoints/`

### Editable settings (`src/train.py`)
| Variable | Meaning |
| -------- | ------- |
| `train_images_dir`, `train_masks_dir`, etc. | Input folders |
| `BATCH_SIZE` | Batch size (default 4) |
| `max_epochs` | Total epochs for `pl.Trainer` |
| `precision` | 16 (mixed) or 32 |
| `lr` | Learning rate |
| `early_stop_patience` | Epochs with no val IoU improvement before stopping (default 5) |
| WandB `project` & `name` | Change run grouping |

You can edit these directly in `src/train.py` before running.

---

## 🧪 Evaluate

```bash
python src/test.py
```
Prints **Test IoU, Dice, pixel accuracy**.

---

## 🔍 Inference

Single image:
```bash
python src/inference.py --input path/to.jpg --output out.png
```
Batch folder:
```bash
python src/inference_batch.py
```
Results saved to `data/predictions/`.

---

## 🔭 Experiment Tracking (WandB)
* All training/validation metrics, LR curves, and example predictions are logged automatically.
* Final runs: see [coco‑segmentation workspace](https://wandb.ai/chundawatdivyaraj01-indian-institute-of-science/coco-segmentation?nw=nwuserchundawatdivyaraj01).
* Early baseline runs: see [segmentation‑training workspace](https://wandb.ai/chundawatdivyaraj01-indian-institute-of-science/segmentation-training?nw=nwuserchundawatdivyaraj01).

> To run your own logging, install WandB (`pip install wandb`) and `wandb login` with your API key.

---

## 📋 License & Citation
This code is for educational use. Please cite this repo or the WandB runs if you build upon it.

