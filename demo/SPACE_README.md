---
title: Patent My Idea
emoji: 📐
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: openrail++
short_description: Turn any gadget into a US patent-style figure (SDXL LoRA)
---

# Patent My Idea

Type a gadget, pick a view, and compare SDXL without and with the patent-drawing style LoRA.

Deploy (account must be 30+ days old with a verified email to host ZeroGPU for free): create a Space (SDK: Gradio, hardware: ZeroGPU), then upload `app.py`, `requirements.txt`
and this file renamed to `README.md`. Set the `LORA_REPO` variable in the Space settings if your
LoRA lives at a different Hub repo.
