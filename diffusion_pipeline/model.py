from diffusers import StableDiffusionXLPipeline
import torch

def load_model():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"  # required for CUDA float16 variant
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe
