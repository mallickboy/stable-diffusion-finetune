import torch
import os
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler

class LoraInference:
    def __init__(self, seed=42):
        self.generator = torch.Generator("cuda").manual_seed(seed)
        self.negative_prompt = "low quality, blurry, worst quality, noise, bad anatomy, deformed"

    def init_pipe(self, lora_path=None):
        dtype = torch.float16
        model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_model_name = "madebyollin/sdxl-vae-fp16-fix"
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")

        vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=dtype).to("cuda")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name, vae=vae, scheduler=scheduler, torch_dtype=dtype, variant="fp16"
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()

        if lora_path and os.path.exists(lora_path):
            pipe.load_lora_weights(lora_path)
        return pipe

    def inference(self, inf_pipe, prompts, save=None, show=False, seed=42):
        pipe = inf_pipe
        results = []
        for i, prompt in enumerate(prompts, 1):
            self.generator.manual_seed(seed)
            image = pipe(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                generator=self.generator,
                num_inference_steps=30,
                guidance_scale=7.5,
                output_type="pil"
            ).images[0]

            results.append((prompt, image))
            if save:
                os.makedirs(save, exist_ok=True)
                filename = os.path.join(save, f"inf_{i}_{prompt[:30].replace(' ', '_').replace(',', '')}.png")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                image.save(filename)

        if show:
            plt.figure(figsize=(15, len(results) * 5))
            for i, (prompt, image) in enumerate(results, 1):
                plt.subplot(len(results), 1, i)
                plt.imshow(image)
                plt.axis("off")
                plt.title(prompt, fontsize=12)
            plt.tight_layout()
            plt.show()

        return results

    @classmethod
    def compare_results(cls, res1, res2, title1="Base Model", title2="LoRA Model", save=None):
        n = len(res1)
        plt.figure(figsize=(10, n * 5))
        for i, ((prompt1, img1), (prompt2, img2)) in enumerate(zip(res1, res2), 1):
            plt.subplot(n, 2, (i * 2) - 1)
            plt.imshow(img1)
            plt.axis("off")
            plt.title(f"{title1}\n{prompt1[:50]}", fontsize=10)

            plt.subplot(n, 2, i * 2)
            plt.imshow(img2)
            plt.axis("off")
            plt.title(f"{title2}\n{prompt2[:50]}", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)
            plt.savefig(save)
        plt.show()

if __name__=="__main__":
    lrf = LoraInference("")
    base_pipe = lrf.init_pipe()
    ckpt_pipe = lrf.init_pipe("/content/drive/MyDrive/naruto_sdxl_lora_output/checkpoint-600/pytorch_lora_weights.safetensors")
    prompts = [
        "Naruto Uzumaki eating ramen",
        "Bill Gates in Naruto style",
        "A boy with blue eyes in Naruto style",
    ]
    base_res = lrf.inference(base_pipe, prompts)
    ckpt_res = lrf.inference(ckpt_pipe, prompts)
    lrf.compare_results(base_res, ckpt_res)
