"""Patent My Idea: Gradio demo for the patent-drawing style LoRA.

Runs on a Hugging Face Space with ZeroGPU hardware (the @spaces.GPU decorator
attaches a GPU only while generate() runs). Locally it runs on any CUDA GPU;
set OFFLOAD=1 for 8GB cards.

Shows the same prompt and seed without and with the LoRA, side by side.
"""
import os

import gradio as gr
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

try:
    import spaces  # available on Hugging Face Spaces
    # ZeroGPU charges the requested duration against each visitor's daily quota, so keep it tight:
    # two 1024px SDXL images at <=30 steps take well under 40 s on ZeroGPU hardware.
    gpu = spaces.GPU(duration=40)
except ImportError:  # local run
    def gpu(fn):
        return fn

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
LORA = os.environ.get("LORA_REPO", "sisodiajatin/patent-style-lora")
TRIGGER = "ptnt_style"
NEG = "color, gray background, photo, photorealistic, 3D render, blurry, text, watermark"
VIEWS = ["perspective view", "front view", "side view", "top view", "exploded view"]
MAX_OBJECT_CHARS = 80

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(BASE, vae=vae, torch_dtype=torch.float16,
                                                 variant="fp16", use_safetensors=True)
pipe.load_lora_weights(LORA, adapter_name="patent")
if os.environ.get("OFFLOAD") == "1":
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")


@gpu
def generate(thing: str, view: str, seed: float, strength: float, steps: float):
    thing = (thing or "").strip()[:MAX_OBJECT_CHARS] or "a quadcopter drone"
    view = view if view in VIEWS else VIEWS[0]
    strength = max(0.0, min(float(strength), 1.2))
    steps = int(max(10, min(int(steps), 30)))
    out = []
    for weight, prompt in [(0.0, f"{view} of {thing}"), (strength, f"{TRIGGER}, {view} of {thing}")]:
        pipe.set_adapters(["patent"], adapter_weights=[weight])
        g = torch.Generator("cpu").manual_seed(int(seed))
        out.append(pipe(prompt, negative_prompt=NEG, num_inference_steps=steps,
                        guidance_scale=6.0, generator=g).images[0])
    return out


with gr.Blocks(title="Patent My Idea") as demo:
    gr.Markdown("# Patent My Idea\nType any gadget and see it as a US patent figure. "
                "Same prompt and seed, without and with the LoRA.")
    with gr.Row():
        thing = gr.Textbox(value="a quadcopter drone", label="Your invention", max_lines=1)
        view = gr.Dropdown(VIEWS, value=VIEWS[0], label="View")
    with gr.Row():
        seed = gr.Number(value=42, precision=0, label="Seed")
        strength = gr.Slider(0, 1.2, value=0.7, step=0.05, label="LoRA strength")
        steps = gr.Slider(10, 30, value=25, step=1, label="Steps")
    btn = gr.Button("Generate", variant="primary")
    with gr.Row():
        without_lora = gr.Image(label="Without LoRA")
        with_lora = gr.Image(label="With LoRA")
    gr.Examples([["a smartwatch", "front view"], ["a robot arm", "perspective view"],
                 ["an electric scooter", "side view"]], [thing, view])
    btn.click(generate, [thing, view, seed, strength, steps], [without_lora, with_lora])

if __name__ == "__main__":
    demo.queue(max_size=10).launch()
