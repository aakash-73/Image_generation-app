from diffusion_pipeline.model import load_model
from inference.benchmark import benchmark
import gradio as gr

# Load the SDXL model once at startup
pipe = load_model()

# Gradio callback function
def gradio_generate(prompt):
    print("⏳ Generating image...")
    image = benchmark(pipe, prompt)
    return image

# Launch Gradio interface
def run_gradio():
    gr.Interface(
        fn=gradio_generate,
        inputs=gr.Textbox(label="Enter your scene description"),
        outputs=gr.Image(type="pil"),
        title="🌍 WorldSketch - Text-to-Image Generator (SDXL)",
        description="Enter a creative scene or concept. This model uses SDXL for high-quality generation.",
    ).launch(share=False)

if __name__ == "__main__":
    run_gradio()
