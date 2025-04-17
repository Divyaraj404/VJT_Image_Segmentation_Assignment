# COCO UNet Segmentation Assignment – **Task 2 (Model Training & Inference)**

**Task 1 (Dataset Preparation)** lives here → **<https://github.com/Divyaraj404/VJT_Image_Segmentation_Assignment_task1>**.  
That repository shows how the COCO masks were remapped into 11 super‑classes and exported for 5 000 images. **You must prepare that dataset first (or download it from Drive below) before running Task 2.**

---

## 📁 Repository Structure
```
├── data/                      # <‑‑ Put downloaded images & masks here
├── coco-segmentation/         # Placeholder (.gitkeep)
├── src/                       # All source code
│   ├── dataset.py             # Dataset + remap logic
│   ├── unet.py                # UNet architecture
│   ├── lit_module.py          # LightningModule (loss + metrics)
│   ├── train.py               # Training entry point
│   ├── test.py                # Evaluation script
│   ├── inference.py           # Single‑img inference
│   └── inference_batch.py     # Batch inference
└── …
```

---

## 🔗 Links & Assets
| Purpose | URL |
| ------- | --- |
| **Task 1 repo (dataset scripts & README)** | <https://github.com/Divyaraj404/VJT_Image_Segmentation_Assignment_task1> |
| **Dataset & checkpoints (Google Drive)** | <https://drive.google.com/drive/folders/1dE3X83SM7vdxHjRr5MxF3VaT5WFAC2uu?usp=sharing> |
| **Final WandB dashboard (UNet)** | <https://wandb.ai/chundawatdivyaraj01-indian-institute-of-science/coco-segmentation?nw=nwuserchundawatdivyaraj01> |
| **Baseline WandB runs (old model)** | <https://wandb.ai/chundawatdivyaraj01-indian-institute-of-science/segmentation-training?nw=nwuserchundawatdivyaraj01> |

> **Download the Drive archive first!** Extract it so you have `data/train_images/`, `data/train_masks/`, … at repo root. Otherwise `train.py` will raise *file not found* errors.

---

## 🚀 Quick Start (condensed)
```bash
# clone & cd
git clone https://github.com/Divyaraj404/VJT_Image_Segmentation_Assignment.git
cd VJT_Image_Segmentation_Assignment

# download + place data/

# env with uv
python3 -m venv venv && source venv/bin/activate
pip install uv
uv init --python=3.8
uv install -r requirements.txt

# (optional) WandB login
pip install wandb && wandb login
```

---

## 🏋️ Training (`src/train.py`)
```bash
python src/train.py
```
What happens:
1. **Data loading** – reads `data/train_*` & `data/val_*` using `COCOSegDataset` with on‑the‑fly Albumentations.
2. **Model** – UNet (3⟶64 base channels, mixed‑precision) defined in `unet.py`.
3. **Callbacks**  
   • `ModelCheckpoint` → saves best checkpoint by **val IoU**.  
   • `EarlyStopping` (patience =`early_stop_patience`, default 5) – increase or set to `None` to run longer.  
   • `LearningRateMonitor` → logs LR every epoch.
4. **Logging** – metrics (loss, IoU, Dice, pixel‑acc), LR curve, example masks are streamed to the **final WandB project** linked above.
5. **Outputs** – best checkpoint in `lightning_logs/version_*/checkpoints/`.

**Change hyper‑parameters** by editing the variables at the top of `train.py`:
- paths (`train_images_dir`, …)  
- `BATCH_SIZE`, `max_epochs`, `precision` (16/32), `lr`  
- `early_stop_patience` if you want to disable early stop or train longer.

---

## 🧪 Evaluate (`src/test.py`)
```bash
python src/test.py
```
- Loads the **best checkpoint** automatically (edit path in script if needed).  
- Computes mIoU, Dice, and pixel‑accuracy on `data/test_*`.  
- Prints metrics to console and (if WandB enabled) logs to the run.

---

## 🔍 Inference
| Command | Description |
| ------- | ----------- |
| `python src/inference.py --input path/to.jpg --output pred.png` | Run on one image; saves colored mask overlay. |
| `python src/inference_batch.py` | Processes **all** JPG/PNG files in `data/new_images/` and writes overlays & masks into `data/predictions/`. |

Both scripts rely on the same preprocessing pipeline used in validation to ensure consistency.

---

## 🐧 Reproducing on Linux / WSL2 (full log)
<details><summary>Show step‑by‑step shell</summary>

```bash
# install system deps (Ubuntu 20.04)
sudo apt update && sudo apt install -y git python3.8 python3.8-venv build-essential

# NVIDIA driver / CUDA toolkit if GPU desired

# clone repo & enter
git clone https://github.com/Divyaraj404/VJT_Image_Segmentation_Assignment.git
cd VJT_Image_Segmentation_Assignment

# download Drive assets -> ensure ./data/ exists with subfolders

# create venv + install packages via uv
python3.8 -m venv venv && source venv/bin/activate
pip install uv
uv init --python=3.8
uv install -r requirements.txt

# (optional) verify CUDA\python - <<'PY'
import torch, os; print('CUDA available:', torch.cuda.is_available())
PY

# (optional) WandB
pip install wandb && wandb login

# train
python src/train.py

# evaluate
python src/test.py
```
</details>

---

## ⚙️ Key Settings Quick‑reference
| Variable in `train.py` | Purpose |
|-----------------------|---------|
| `BATCH_SIZE` | memory vs. speed trade‑off |
| `max_epochs` | upper bound training duration |
| `early_stop_patience` | set high (e.g. 50) or `None` to disable early stop |
| `precision` | 16→mixed FP16, 32→full FP32 |
| `lr` | learning‑rate search |
| WandB project/name | experiment grouping on dashboard |

---

## 🔭 WandB Tracking
Every run is logged to WandB. Public dashboards:
- **UNet final runs** → <https://wandb.ai/chundawatdivyaraj01-indian-institute-of-science/coco-segmentation?nw=nwuserchundawatdivyaraj01>
- **Legacy baseline** → <https://wandb.ai/chundawatdivyaraj01-indian-institute-of-science/segmentation-training?nw=nwuserchundawatdivyaraj01>

You can compare your reproduction metrics directly with these runs.

---

## License
Educational / research use only. Please cite this repository and associated WandB runs if used in your work.

