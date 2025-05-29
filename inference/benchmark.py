import time
from diffusion_pipeline.generate import generate_image

def benchmark(pipe, prompt: str, output_path="output.png"):
    start = time.time()
    image = generate_image(pipe, prompt, output_path)
    pytorch_time = time.time() - start
    print(f"🕒 PyTorch inference time: {pytorch_time:.2f} seconds")
    return image
