import torch

def generate_image(pipe, prompt, output_path="output.png"):
    generator = torch.Generator(device="cuda").manual_seed(42)

    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted",
        guidance_scale=7.5,  # Recommended range: 5.0–8.5
        num_inference_steps=40,  # 30–50 is usually sufficient
        generator=generator
    )

    image = result.images[0]
    image.save(output_path)
    return image


