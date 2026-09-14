# SGD-Guard: Robust, Generalizable Proactive Face-Swapping Defense via Semantic Gradient Divergence

This directory is an independent, runnable implementation of:

> Seung-hyeok Back, Do-Hyun Ki, Juwan Kim, and Seok-Bong Yoo, “Robust, Generalizable Proactive Face-Swapping Defense via Semantic Gradient Divergence.” **International Joint Conference on Artificial Intelligence (IJCAI), 2026.**

**Publication status:** the paper is **accepted to IJCAI 2026**. The official proceedings version is forthcoming.


## Overview

SGD-Guard is an image-level proactive face-swapping defense. It adds a small pixel perturbation to a face image so that identity-critical facial attributes are disrupted in downstream face-swapping models while maintaining visual quality. The perturbation is optimized to remain effective under diffusion purification, JPEG compression, and geometric or photometric transformations.

The implementation follows Sections 3.1--3.6 and Algorithms 1--2 of the paper and includes the CI-JES semantic distortion attack, semantic-direction EOT, semantic-robust JPEG, SIIR generalized identity gallery, and diffusion-guided optimization.



<p align="center">
  <img src="assets/overview.jpg" width="75%">
</p>

<!-- <p align="center">
  <b>Overview of the proposed SGD-Guard framework.</b>
</p> -->


<p align="center">
  <img src="assets/results.jpg" width="55%">
</p>

<!-- <p align="center">
  <b>Qualitative results of SGD-Guard.</b>
</p> -->


## What is implemented

SGD-Guard consists of the following components:

1. **CI-JES semantic distortion attack (SD-attack).** FaRL CLIP image features and identity features are joined into a normalized CLIP--identity joint embedding space. For each of `eyebrows`, `eyes`, `nose`, and `lips`, the top-`k` and bottom-`k` gallery samples are retrieved by CLIP/text similarity. Their CLIP and generalized identity centroids form four pairs of semantic anchors. The offset loss uses consensus weights from the cosine agreement between CLIP and identity directions.

2. **Semantic direction EOT (SD-EOT).** For each differentiable transform, the mean joint-feature shift is precomputed on FFHQ. During protection, transforms whose shift conflicts with the current attack direction receive larger softmax weights.

3. **Semantic robust JPEG (SR-JPEG).** The forward value is produced by the real PIL JPEG codec. The backward pass uses a differentiable JPEG-like low-pass surrogate through the BPDA straight-through construction described in Section 3.4.

4. **SIIR generalized identity gallery.** ArcFace and FaceNet embeddings are refined with two four-path residual bottleneck transformation (RBT) modules. Channel-wise disagreement is thresholded by its harmonic mean; stable channels are accumulated and heterogeneous channels are iterated. The final 512-D code is trained with the statistical, cross-path contrastive, and frozen-head classification losses.

5. **Diffusion-guided optimization.** A Stable Diffusion v1.5 image-to-image pipeline with an LCM-LoRA UNet is invoked once per attack iteration. Its forward output is used in the loss, while an identity straight-through estimator sends gradients back to the pixels. The default setting uses four LCM denoising steps and `strength=0.4`, matching the released implementation's default.


## Repository layout

```text
SGD-Guard_IJCAI26/
├── build_gallery.py          # ArcFace/FaceNet cache, classifier heads, SIIR, HDF5 gallery
├── estimate_directions.py    # FFHQ mean CI-JES shifts for SD-EOT
├── train_lora.py             # SD1.5 UNet LoRA purifier training
├── protect.py                # Online Algorithm 2 inference
├── sgd_guard/
│   ├── attack.py             # SD-attack, SD-EOT, SR-JPEG objectives
│   ├── augmentations.py      # Differentiable transform bank
│   ├── data.py               # ImageFolder-style scanning and image I/O
│   ├── diffusion.py          # LCM-LoRA purifier and straight-through wrapper
│   ├── gallery.py            # HDF5 gallery, anchors, CI-JES offset loss
│   ├── jpeg.py               # Real JPEG + BPDA surrogate
│   ├── models.py             # ArcFace, FaceNet, and FaRL loaders
│   └── siir.py               # SIIR/RBT and gallery losses
├── tools/
│   └── evaluate.py           # Original/protected PSNR and SSIM evaluation
├── environment.yml
└── requirements.txt
```


## Environment

Linux with an NVIDIA GPU and CUDA is recommended. At least 24 GB of GPU memory is recommended for SD1.5 at 512×512. CPU execution is suitable for code inspection and small tests, but full gallery construction, FFHQ direction estimation, and diffusion-based protection are not practical on CPU.

```bash
git clone <REPOSITORY_URL>
cd SGD-Guard_IJCAI26

conda env create -f environment.yml
conda activate sgd-guard

# Or, in an existing environment:
python -m pip install -r requirements.txt
```

The first use of OpenAI CLIP, FaceNet, and Stable Diffusion may download model files from their respective hubs. For offline execution, download all required weights in advance and pass local paths to the commands.


## Pretrained models

Download or prepare the following artifacts and pass their paths explicitly to the scripts.

| Component | Required artifact | Use |
|---|---|---|
| ArcFace | Serialized `nn.Module` checkpoint, e.g. `arcface_checkpoint.tar` | Online identity encoder and one SIIR branch |
| FaceNet | `facenet-pytorch` VGGFace2 weights | Second SIIR branch; loaded by `InceptionResnetV1(pretrained="vggface2")` |
| FaRL | `FaRL-Base-Patch16-LAIONFace20M-ep64.pth` | CLIP ViT-B/16 image/text features |
| Stable Diffusion | `runwayml/stable-diffusion-v1-5` | Image-to-image purifier backbone |
| LCM-LoRA | Directory or `.safetensors` file produced by `train_lora.py` | Fast purifier adaptation |

The ArcFace checkpoint must contain a complete serialized PyTorch module directly or under the `model` key. If that module imports classes from an external reference implementation, pass its root with `--models-root <REFERENCE_ROOT>`.


## Data organization

All image commands accept an ImageFolder-style directory:

```text
<DATA_ROOT>/
├── identity_000001/
│   ├── face_01.jpg
│   └── face_02.jpg
├── identity_000002/
│   └── face_01.jpg
└── ...
```

The paper uses the following data roles:

| Role | Dataset / protocol |
|---|---|
| Offline gallery | One face image per identity, with identity subdirectories. The implementation writes `gallery.h5`. |
| SD-EOT directions | Full FFHQ face dataset, as specified in Section 4.1. |
| Purifier training | CelebA-HQ and VGGFace2-HQ 512×512 face images. |
| Protection evaluation | CelebA-HQ and VGGFace2-HQ. The paper evaluates 1,000 source images, 1,000 random source--target swaps per set, and averages 10 sets. |
| Face-swapping backbones | SimSwap, FaceDancer, DiffSwap, DiffFace, and REFace. These are evaluation-time attack models, not dependencies of the protector. |

The gallery builder selects the first lexicographically sorted image in each identity directory when one-image-per-identity behavior is used internally. A flat directory is interpreted as one unique identity per image, which is useful for smoke tests but is not recommended for paper-scale experiments.


## Training / offline preparation

The full preparation pipeline contains three offline stages before online protection.


### 1. Build the generalized feature gallery

This stage extracts ArcFace, FaceNet, and FaRL features, trains the two frozen classifier heads, trains SIIR, and writes the paired gallery.

```bash
python build_gallery.py \
  --gallery-root <DATA_ROOT>/VGGFace2-HQ/gallery_one_per_identity \
  --arcface-ckpt <MODEL_ROOT>/arcface_checkpoint.tar \
  --farl-ckpt <MODEL_ROOT>/FaRL-Base-Patch16-LAIONFace20M-ep64.pth \
  --out-dir <MODEL_ROOT>/sgd_guard_gallery \
  --models-root <REFERENCE_ROOT> \
  --batch-size 128 \
  --head-epochs 10 \
  --siir-epochs 10 \
  --siir-batch-size 64 \
  --siir-lr 1e-3
```

Outputs include:

```text
<MODEL_ROOT>/sgd_guard_gallery/
├── gallery.h5
├── arcface.npy
├── facenet.npy
├── labels.npy
├── arcface_head.pt
├── facenet_head.pt
├── siir_module_best.pt
└── siir_module_last.pt
```


### 2. Estimate SD-EOT directions on FFHQ

The default transform bank is `brightness`, `color`, `contrast`, `crop`, `gamma`, `hue`, `rotate`, `saturation`, `scale`, `sharpness`, `translateX`, and `translateY`. The resulting vectors have shape `(12, 1024)` and are stored in normalized form.

```bash
python estimate_directions.py \
  --data-root <DATA_ROOT>/FFHQ/images \
  --arcface-ckpt <MODEL_ROOT>/arcface_checkpoint.tar \
  --farl-ckpt <MODEL_ROOT>/FaRL-Base-Patch16-LAIONFace20M-ep64.pth \
  --out <MODEL_ROOT>/sgd_guard_gallery/transform_directions.npz \
  --models-root <REFERENCE_ROOT> \
  --batch-size 64
```

Use all FFHQ images for the reported setting. `--max-images 256` is available for a quick pipeline check.


### 3. Train the LCM-LoRA purifier

The purifier is trained from Stable Diffusion v1.5 on the two high-quality face datasets used by the paper. Only UNet LoRA parameters are updated; the VAE, text encoder, and teacher UNet remain frozen.

```bash
python train_lora.py \
  --data-roots <DATA_ROOT>/CelebA-HQ/images,<DATA_ROOT>/VGGFace2-HQ/images \
  --pretrained-model runwayml/stable-diffusion-v1-5 \
  --output-dir <MODEL_ROOT>/sgd_guard_lora \
  --resolution 512 \
  --batch-size 4 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --lora-rank 4 \
  --lora-alpha 4 \
  --min-timestep 400 \
  --max-timestep 800 \
  --amp fp16
```

The output `unet_lora/` directory can be passed directly to `protect.py`. For a short validation run, add `--max-samples 128 --max-steps 20`; these are not the paper-scale training settings.


## Inference

Protect a directory of images with:

```bash
python protect.py \
  --input <DATA_ROOT>/to_protect \
  --output-dir <OUTPUT_ROOT>/sgd_guard_protected \
  --gallery <MODEL_ROOT>/sgd_guard_gallery/gallery.h5 \
  --directions <MODEL_ROOT>/sgd_guard_gallery/transform_directions.npz \
  --arcface-ckpt <MODEL_ROOT>/arcface_checkpoint.tar \
  --farl-ckpt <MODEL_ROOT>/FaRL-Base-Patch16-LAIONFace20M-ep64.pth \
  --lora-path <MODEL_ROOT>/sgd_guard_lora/unet_lora \
  --models-root <REFERENCE_ROOT> \
  --iterations 5 \
  --step-size 0.0078431373 \
  --epsilon 0.0117647059 \
  --retrieval-k 10 \
  --purifier-steps 4 \
  --purifier-strength 0.4 \
  --jpeg-qualities 30,50,70
```

For a single image:

```bash
python protect.py \
  --input <DATA_ROOT>/face.jpg \
  --output-dir <OUTPUT_ROOT>/sgd_guard_protected \
  --gallery <MODEL_ROOT>/sgd_guard_gallery/gallery.h5 \
  --directions <MODEL_ROOT>/sgd_guard_gallery/transform_directions.npz \
  --arcface-ckpt <MODEL_ROOT>/arcface_checkpoint.tar \
  --farl-ckpt <MODEL_ROOT>/FaRL-Base-Patch16-LAIONFace20M-ep64.pth \
  --lora-path <MODEL_ROOT>/sgd_guard_lora/unet_lora \
  --models-root <REFERENCE_ROOT>
```

The output directory mirrors the input tree and stores protected PNG files together with `metrics.json`. Using `--save-tensors` additionally saves the original and protected tensors for debugging.

The default normalized perturbation budget is `epsilon=3/255`, and the PGD step size is `2/255`. These values are expressed in `[0,1]` image units. The final image is returned as `clip(x + delta, 0, 1)`, as in Algorithm 2.


## Ablations and debugging

```bash
# Remove the diffusion purifier from the online objective.
python protect.py ... --disable-purifier

# SD-attack only.
python protect.py ... --disable-purifier --disable-sd-eot --disable-sr-jpeg

# SD-attack + SD-EOT.
python protect.py ... --disable-purifier --disable-sr-jpeg
```

`--allow-base-clip` is an explicit debugging fallback when a FaRL checkpoint is unavailable and should not be used for the paper setting.

`--projection-ckpt` optionally loads separate 512-to-256-to-256 directional projection heads. The paper describes these lightweight heads but does not release a projection checkpoint; the default therefore uses the shared-space identity map for the already 512-D features.

<!--
## Evaluation

The included evaluator computes source-image PSNR and SSIM for matching original and protected pairs:

```bash
python tools/evaluate.py \
  --original-dir <DATA_ROOT>/CelebA-HQ/eval_original \
  --protected-dir <OUTPUT_ROOT>/sgd_guard_protected \
  --json-out <OUTPUT_ROOT>/sgd_guard_protected/source_metrics.json
```

For the paper's deepfake metrics, generate swaps with SimSwap, FaceDancer, DiffSwap, DiffFace, and REFace using the same random source--target protocol. Then measure PSNR/SSIM on swap outputs, ISM for identity similarity, DSR for source-identity failure, and PSR for protected-source recognition failure.

The face-swapping implementations and their pretrained weights are intentionally external because they are evaluation backbones rather than components of SGD-Guard.


## Reproducibility notes

- Gallery retrieval uses exact top-`k`/bottom-`k` cosine ranking (`k=10` by default).
- The four semantic anchors are built from matched CLIP and generalized SIIR identity centroids, not from a single identity encoder.
- `SD-EOT` weights are based on `1 - cosine(g_adv, mean_direction)` and are detached before the transform losses, matching the paper's online adaptive weighting.
- `SR-JPEG` keeps the actual JPEG output in the forward computation and uses only the surrogate in the backward pass.
- The purifier is invoked once per online iteration. Its PIL/diffusion forward path is wrapped with a straight-through gradient to keep pixel optimization well-defined.
- The current method is single-image and does not enforce temporal consistency for video, which is also listed as a limitation in the paper appendix.
-->

## Citation

```bibtex
@inproceedings{back2026sgdguard,
  author    = {Seung-hyeok Back and Do-Hyun Ki and Juwan Kim and Seok-Bong Yoo},
  title     = {Robust, Generalizable Proactive Face-Swapping Defense via Semantic Gradient Divergence},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2026},
  note      = {Accepted; proceedings version forthcoming}
}
```
