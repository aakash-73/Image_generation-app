import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion_pipeline.model import load_model
from diffusion_pipeline.generate import generate_image

def test_image_generation():
    pipe = load_model()
    prompt = "A scenic waterfall in the forest"
    output_path = "test_output.png"
    generate_image(pipe, prompt, output_path)
    assert os.path.exists(output_path)
