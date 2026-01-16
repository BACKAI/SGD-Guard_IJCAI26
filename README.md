# Robust, Generalizable Proactive Face-swapping Defense via Semantic Gradient Divergence


## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (Not mandatory bur recommended)
- Python 3


### Installation
- Dependencies:
	1. lpips
	2. wandb
	3. pytorch
	4. torchvision
	5. matplotlib
	6. dlib
- All dependencies can be installed using *pip install* and the package name


## Pretrained Models
Please download the pretrained models from the following link:

|[ArcFace](https://drive.google.com/drive/folders/1jV6_0FIMPC53FZ2HzZNJZGMe55bbu17R) | used to build the generalized feature gallery. Download the weights and place them in the `pretrained_model/` directory.

|[FaceNet](https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view) | used to build the generalized feature gallery. Download the weights and place them in the `pretrained_model/` directory.

|[FaRL (CLIP model)](https://github.com/FacePerceiver/FaRL) | Used for text-image embedding. Download weights and place them in `pretrained_model/` as described in the FaRL repository.


---

## Training & Inference Pipeline

Follow the steps below to train the model components and run inference.

### Step 1: Train Generalized Feature Gallery

Train the model to build a robust, generalizable feature gallery for face representation.

```bash
python feature_gallery/siir_train.py
```

> **Note**: This step trains the feature extractor that will be used to construct the generalized feature gallery. Ensure your dataset paths are correctly configured before running.

---

### Step 2: Build Generalized Feature Gallery

Generate the feature gallery using the trained model from Step 1.

```bash
python feature_gallery/siir_inference.py
```

> **Output**: The generated feature gallery will be saved and used for semantic-aware adversarial perturbation generation.

---

### Step 3: Fine-tune One-Step Diffusion Purifier with LoRA

Fine-tune the diffusion-based purification model using Low-Rank Adaptation (LoRA) for efficient one-step purification.

```bash
python train_lora.py
```

> **Note**: This step adapts a pretrained diffusion model for fast, single-step image purification while preserving facial features.

---

### Step 4: Run SGD-Guard

After completing the above steps, place the generated files in the `pretrained/` directory and run the main protection script.

```bash
# Ensure the following files are in place:
# - pretrained/feature_gallery.h5 (or equivalent)
# - pretrained/adapter_model.safetensors (LoRA weights)

python sgd_guard.py --data_path <your_data_path> --output_dir <output_path>
```

---

### 📋 Quick Reference

| Step | Script | Description |
|:----:|--------|-------------|
| 1 | `siir_train.py` | Train generalized feature gallery |
| 2 | `siir_inference.py` | Build feature gallery |
| 3 | `train_lora.py` | Fine-tune diffusion purifier with LoRA |
| 4 | `sgd_guard.py` | Run SGD-Guard protection |

