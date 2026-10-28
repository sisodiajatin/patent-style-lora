---
base_model: stabilityai/stable-diffusion-xl-base-1.0
library_name: diffusers
license: openrail++
tags:
  - lora
  - text-to-image
  - stable-diffusion-xl
  - line-art
instance_prompt: ptnt_style
---

# Patent-Drawing Style LoRA (SDXL)

Turns any object into a black-and-white US patent figure: black lines on white, line-based shading, standard views.

**Trigger word:** `ptnt_style`. Put it first, then the view and the object:

```
ptnt_style, perspective view of a quadcopter drone
```

Recommended LoRA strength: **0.7**. Negative prompt that helps:
`color, gray background, photo, photorealistic, 3D render, blurry, text, watermark`

## Usage

```python
import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("sisodiajatin/patent-style-lora")
image = pipe("ptnt_style, side view of an electric scooter",
             negative_prompt="color, photo, 3D render, text").images[0]
```

## Training

- Base: SDXL base 1.0, with the madebyollin/sdxl-vae-fp16-fix VAE
- Data: 24 drawing sheets from US design patents (via the DeepPatent dataset), one figure each,
  labels removed, padded to 1024×1024. Patent numbers are listed in `manifest.csv` in the project repo.
- LoRA rank 16 (alpha 16) on the UNet and both text encoders, LR 1e-4 constant, AdamW 8-bit, batch 1, bf16,
  noise offset 0.1
- Steps: 1,500 of a 3,000-step run (later checkpoints overfit)
- Hardware: one NVIDIA L4 (Google Colab)

## Evaluation

10 subjects × 3 seeds per condition, fixed sampler, steps, CFG and negative prompt. Unseen gadgets (drone, smartwatch,
VR headset, scooter, robot arm, foldable phone):

| Condition | White border ↑ | Gray share ↓ | CLIP subject ↑ |
| --- | --- | --- | --- |
| Prompt only ("patent drawing … line art") | 0.440 | 0.241 | 0.301 |
| LoRA @ 0.7 | 0.998 | 0.038 | 0.316 |

The LoRA's gain is the page: pure white, crisp black lines, no beige paper, with the subject preserved.

## Limitations

- Generated figures are illustrations, not valid patent drawings, and carry no legal meaning.
- Text and reference numbers are not reliable; the training data had labels removed on purpose.
- Mechanical plausibility is not guaranteed: parts may not connect sensibly.

## Data credits

US patent drawings are generally not subject to copyright (USPTO). Images were sourced through
DeepPatent (Kucer et al., WACV 2022).
