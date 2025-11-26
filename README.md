# LoRA Fine-Tuning (Naruto Style – SDXL)

This project fine-tunes **Stable Diffusion XL Base 1.0** using **LoRA** adapters to reproduce imagery in the style of the *Naruto* anime series.

---

## Checkpoints & Logs

| Version | Link |
|----------|------|
| 1024 px Model | [Google Drive](https://drive.google.com/drive/folders/1Fke4VZ5_IJyIH0vOB6VMx4cAxspXcRNw?usp=sharing) |
| 256 px Model | [Google Drive](https://drive.google.com/drive/folders/1wqJb4Xo0R5nVFMrxCPTCQvhIiGBDUenx?usp=sharing) |
| Inference & Comparisons | [Google Drive](https://drive.google.com/drive/folders/1etQR-2ir_iUklRpibzzWU6NaDlqKGep4?usp=sharing) |

---

## High-Level Idea

Stable Diffusion is made of several parts that work together:

| Component | Purpose | Type |
|------------|----------|------|
| **VAE (Variational Autoencoder)** | Turns images into smaller latent codes and back. | Autoencoder (CNN) |
| **U-Net (Denoiser)** | Learns to remove noise and form the final image. | UNet2DConditionModel |
| **Text Encoder** | Converts text prompts into embeddings that guide the image. | CLIP Encoder |
| **Scheduler** | Controls how noise is reduced step by step. | DDIM, Euler, etc. |

We only train a few layers of the U-Net using **LoRA (Low-Rank Adaptation)**.  
LoRA adds small trainable matrices inside the attention layers while keeping the main model frozen.  
This keeps the training lightweight and fast, which worked perfectly on a free-tier **T4 GPU (16 GB)**.

---

## Training Workflow

### Setup
Used Hugging Face Diffusers official SDXL LoRA training script:
```bash
!pip install -qqq git+https://github.com/huggingface/diffusers.git
!wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
```

### Model & Dataset
- **Base model:** `stabilityai/stable-diffusion-xl-base-1.0`  
- **Dataset:** `lambdalabs/naruto-blip-captions`  
- **Optimizer:** 8-bit Adam  
- **Precision:** mixed FP16 (faster on T4 than BF16)  
- **Rank:** 16 (LoRA rank)

---

## Training Progress

### 1024 px Training
- Batch = 1, Grad Accum = 8 → effective 8 images/step  
- Trained ~10 steps (initial), then optimized T4 setup (8-bit Adam + FP16)  
- Continued training ≈ 300 steps @ LR = 0.0001  
- Reduced LR to 0.00003 (cosine scheduler) for stability → total ≈ 600 steps  
- Final run to 765 steps (~5 epochs)

### 256 px Training
- Noticed dataset images are simple and low res  
- Trained fast LoRA with Batch = 12, Grad Accum = 8 (~96 images per step)  
- 200 steps @ LR = 1e-4  
- Provided quick style adaptation baseline

---

## Intermediate Notes

```text
# train_batch_size = 12
# gradient_accumulation_steps = 8
# -> 1 optimizer step = 12 * 8 = 96 images
# dataset size = 1221
# -> ~13 steps per epoch
# later: batch=8 -> 1 step = 64 images -> ~20 steps/epoch

# 1024px setup:
# batch=1, grad_accum=8 -> 8 images/step
# -> ~153 steps per epoch (~1221 images)
# trained ~270 steps @ lr=0.0001 then reduced to 0.00003 (cosine)
```

FP16 ran ~2× faster than BF16 on Colab T4.

---

## Evaluation

Compared:
- **Base SDXL vs 256 LoRA**
- **Base SDXL vs 1024 LoRA**
- **1024 vs 256 LoRA**  

All comparisons and inference outputs are available in the shared Drive folder above.

---
## Some Visuals
***Base VS 1024 Model***
![Image](https://github.com/user-attachments/assets/6222abcd-b77b-4c05-b549-a84e254691b9)

***Base VS 256 Model***
![Image](https://github.com/user-attachments/assets/0bc17594-78db-4323-a5ee-3ddd89f301b5)

***1024 Model VS 256 Model***
![Image](https://github.com/user-attachments/assets/ef9f90d5-e2c6-4d0a-8b51-48c2c82c5e45)


## Notes

- All code was executed on **free-tier Google Colab (T4 GPU, 16 GB VRAM)**.  
- Training used **gradient checkpointing**, **mixed precision (fp16)**, and **8-bit Adam** for memory efficiency.  
- Checkpoints saved every 30 steps (`checkpoint-<n>`).  
- Logs and intermediate weights included in Drive links.

---

## References

- Hugging Face Diffusers LoRA Example: [train_text_to_image_lora_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py)
- Dataset: [lambdalabs/naruto-blip-captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)
- Base Model: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

---

*Author: Tamal Mallick*  
*Date: November 2025*

