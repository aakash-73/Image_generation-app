# Image_generation-app

# Name:- WorldSketch - Text-to-Image Generator (SDXL)

**WorldSketch** is a PyTorch-based application for high-quality text-to-image generation using **Stable Diffusion XL (SDXL)**. It features a simple Gradio interface and supports deterministic image generation for consistent results.

---

## 🚀 Features

- 🧠 Uses `stabilityai/stable-diffusion-xl-base-1.0`
- 🎛️ Configured with `float16` + attention slicing for speed & memory efficiency
- 🖼️ Generates 1024x1024 resolution images with 40 inference steps
- 🖱️ Simple Gradio interface for interactive prompt input
- ✅ Includes a test suite using `pytest`

---

## 🛠 Requirements

- Python 3.9 or newer
- NVIDIA GPU with CUDA support (8GB VRAM minimum recommended)

Install dependencies:

```bash
pip install -r requirements.txt
